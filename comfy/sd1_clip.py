from __future__ import annotations

import copy
import importlib.resources
import logging
import numbers
import os
import re
import traceback
import zipfile
from pathlib import Path

try:
    from importlib.resources.abc import Traversable  # pylint: disable=no-name-in-module
except ImportError:
    from importlib.abc import Traversable  # pylint: disable=no-name-in-module
from typing import Tuple, Sequence, TypeVar, Callable

import torch
from transformers import CLIPTokenizer, PreTrainedTokenizerBase

from . import clip_model
from . import model_management
from . import ops
from .component_model import files
from .component_model.files import get_path_as_dict, get_package_as_path
from .text_encoders.spiece_tokenizer import SPieceTokenizer


def gen_empty_tokens(special_tokens, length):
    start_token = special_tokens.get("start", None)
    end_token = special_tokens.get("end", None)
    pad_token = special_tokens.get("pad")
    output = []
    if start_token is not None:
        output.append(start_token)
    if end_token is not None:
        output.append(end_token)
    output += [pad_token] * (length - len(output))
    return output


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0 and hasattr(self, "special_tokens"):
            if hasattr(self, "gen_empty_tokens"):
                to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
            else:
                to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))  # pylint: disable=no-member

        assert hasattr(self, "encode")
        assert isinstance(self.encode, Callable)  # pylint: disable=no-member
        o = self.encode(to_encode)  # pylint: disable=no-member
        out, pooled = o[:2]

        if pooled is not None:
            first_pooled = pooled[0:1].to(model_management.intermediate_device())
        else:
            first_pooled = pooled

        output = []
        for k in range(0, sections):
            z = out[k:k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if (len(output) == 0):
            r = (out[-1:].to(model_management.intermediate_device()), first_pooled)
        else:
            r = (torch.cat(output, dim=-2).to(model_management.intermediate_device()), first_pooled)

        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == "attention_mask":
                    v = v[:sections].flatten().unsqueeze(dim=0).to(model_management.intermediate_device())
                extra[k] = v

            r = r + (extra,)
        return r


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    LAYERS = [
        "last",
        "pooled",
        "hidden",
        "all"
    ]

    def __init__(self, device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None, textmodel_json_config: str | dict | None = None, dtype=None, model_class=clip_model.CLIPTextModel,
                 special_tokens=None, layer_norm_hidden_state=True, enable_attention_masks=False, zero_out_masked=False,
                 return_projected_pooled=True, return_attention_masks=False, model_options=None):  # clip-vit-base-patch32
        super().__init__()
        if model_options is None:
            model_options = {}
        if special_tokens is None:
            special_tokens = {"start": 49406, "end": 49407, "pad": 49407}
        assert layer in self.LAYERS

        if textmodel_json_config is None and "model_name" not in model_options:
            model_options = {**model_options, "model_name": "clip_l"}

        config = get_path_as_dict(textmodel_json_config, "sd1_clip_config.json", package=__package__)

        te_model_options = model_options.get("{}_model_config".format(model_options.get("model_name", "")), {})
        for k, v in te_model_options.items():
            config[k] = v

        operations = model_options.get("custom_operations", None)
        scaled_fp8 = None

        if operations is None:
            scaled_fp8 = model_options.get("scaled_fp8", None)
            if scaled_fp8 is not None:
                operations = ops.scaled_fp8_ops(fp8_matrix_mult=False, override_dtype=scaled_fp8)
            else:
                operations = ops.manual_cast

        self.operations = operations
        self.transformer = model_class(config, dtype, device, self.operations)
        if scaled_fp8 is not None:
            self.transformer.scaled_fp8 = torch.nn.Parameter(torch.tensor([], dtype=scaled_fp8))

        self.num_layers = self.transformer.num_layers

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens

        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks
        self.zero_out_masked = zero_out_masked

        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        self.return_attention_masks = return_attention_masks

        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        if self.layer == "all":
            pass
        elif layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_options(self):
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]

    def process_tokens(self, tokens, device):
        end_token = self.special_tokens.get("end", None)
        if end_token is None:
            cmp_token = self.special_tokens.get("pad", -1)
        else:
            cmp_token = end_token

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for x in tokens:
            attention_mask = []
            tokens_temp = []
            other_embeds = []
            eos = False
            index = 0
            for y in x:
                if isinstance(y, numbers.Integral):
                    if eos:
                        attention_mask.append(0)
                    else:
                        attention_mask.append(1)
                    token = int(y)
                    tokens_temp += [token]
                    if not eos and token == cmp_token:
                        if end_token is None:
                            attention_mask[-1] = 0
                        eos = True
                else:
                    other_embeds.append((index, y))
                index += 1

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.transformer.get_input_embeddings()(tokens_embed, out_dtype=torch.float32)
            index = 0
            pad_extra = 0
            for o in other_embeds:
                emb = o[1]
                if torch.is_tensor(emb):
                    emb = {"type": "embedding", "data": emb}

                emb_type = emb.get("type", None)
                if emb_type == "embedding":
                    emb = emb.get("data", None)
                else:
                    if hasattr(self.transformer, "preprocess_embed"):
                        emb = self.transformer.preprocess_embed(emb, device=device)
                    else:
                        emb = None

                if emb is None:
                    index += -1
                    continue

                ind = index + o[0]
                emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
                emb_shape = emb.shape[1]
                if emb.shape[-1] == tokens_embed.shape[-1]:
                    tokens_embed = torch.cat([tokens_embed[:, :ind], emb, tokens_embed[:, ind:]], dim=1)
                    attention_mask = attention_mask[:ind] + [1] * emb_shape + attention_mask[ind:]
                    index += emb_shape - 1
                else:
                    index += -1
                    pad_extra += emb_shape
                    logging.warning("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored {} != {}".format(emb.shape[-1], tokens_embed.shape[-1]))

            if pad_extra > 0:
                padd_embed = self.transformer.get_input_embeddings()(torch.tensor([[self.special_tokens["pad"]] * pad_extra], device=device, dtype=torch.long), out_dtype=torch.float32)
                tokens_embed = torch.cat([tokens_embed, padd_embed], dim=1)
                attention_mask = attention_mask + [0] * pad_extra

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens

    def forward(self, tokens):
        device = self.transformer.get_input_embeddings().weight.device
        embeds, attention_mask, num_tokens = self.process_tokens(tokens, device)

        attention_mask_model = None
        if self.enable_attention_masks:
            attention_mask_model = attention_mask

        if self.layer == "all":
            intermediate_output = "all"
        else:
            intermediate_output = self.layer_idx

        outputs = self.transformer(None, attention_mask_model, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=self.layer_norm_hidden_state, dtype=torch.float32)

        if self.layer == "last":
            z = outputs[0].float()
        else:
            z = outputs[1].float()

        if self.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()

        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        extra = {}
        if self.return_attention_masks:
            extra["attention_mask"] = attention_mask

        if len(extra) > 0:
            return z, pooled_output, extra

        return z, pooled_output

    def encode(self, tokens):
        return self(tokens)

    def encode_token_weights(self, token_weight_pairs):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        o = self.encode(to_encode)
        out, pooled = o[:2]

        if pooled is not None:
            first_pooled = pooled[0:1].to(model_management.intermediate_device())
        else:
            first_pooled = pooled

        output = []
        for k in range(0, sections):
            z = out[k:k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if (len(output) == 0):
            r = (out[-1:].to(model_management.intermediate_device()), first_pooled)
        else:
            r = (torch.cat(output, dim=-2).to(model_management.intermediate_device()), first_pooled)

        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == "attention_mask":
                    v = v[:sections].flatten().unsqueeze(dim=0).to(model_management.intermediate_device())
                extra[k] = v

            r = r + (extra,)
        return r

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=False)


def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def token_weights(string, current_weight):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ')' and x[0] == '(':
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1:])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out


def escape_important(text):
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text


def unescape_important(text):
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text


def safe_load_embed_zip(embed_path):
    with zipfile.ZipFile(embed_path) as myzip:
        names = list(filter(lambda a: "data/" in a, myzip.namelist()))
        names.reverse()
        for n in names:
            with myzip.open(n) as myfile:
                data = myfile.read()
                number = len(data) // 4
                length_embed = 1024  # sd2.x
                if number < 768:
                    continue
                if number % 768 == 0:
                    length_embed = 768  # sd1.x
                num_embeds = number // length_embed
                embed = torch.frombuffer(data, dtype=torch.float)
                out = embed.reshape((num_embeds, length_embed)).clone()
                del embed
                return out


def expand_directory_list(directories):
    dirs = set()
    for x in directories:
        dirs.add(x)
        for root, subdir, file in os.walk(x, followlinks=True):
            dirs.add(root)
    return list(dirs)


def bundled_embed(embed, prefix, suffix):  # bundled embedding in lora format
    out_list = []
    for k in embed:
        if k.startswith(prefix) and k.endswith(suffix):
            out_list.append(embed[k])
    if len(out_list) == 0:
        return None

    return torch.cat(out_list, dim=0)


def load_embed(embedding_name, embedding_directory, embedding_size, embed_key=None):
    if isinstance(embedding_directory, str):
        embedding_directory = [embedding_directory]

    embedding_directory = expand_directory_list(embedding_directory)

    valid_file = None
    for embed_dir in embedding_directory:
        embed_path = os.path.abspath(os.path.join(embed_dir, embedding_name))
        embed_dir = os.path.abspath(embed_dir)
        try:
            if os.path.commonpath((embed_dir, embed_path)) != embed_dir:
                continue
        except:
            continue
        if not os.path.isfile(embed_path):
            extensions = ['.safetensors', '.pt', '.bin']
            for x in extensions:
                t = embed_path + x
                if os.path.isfile(t):
                    valid_file = t
                    break
        else:
            valid_file = embed_path
        if valid_file is not None:
            break

    if valid_file is None:
        return None

    embed_path = valid_file

    embed_out = None

    try:
        if embed_path.lower().endswith(".safetensors"):
            import safetensors.torch
            embed = safetensors.torch.load_file(embed_path, device="cpu")
        else:
            try:
                embed = torch.load(embed_path, weights_only=True, map_location="cpu")
            except:
                embed_out = safe_load_embed_zip(embed_path)
    except Exception:
        logging.warning("{}\n\nerror loading embedding, skipping loading: {}".format(traceback.format_exc(), embedding_name))
        return None

    if embed_out is None:
        if 'string_to_param' in embed:
            values = embed['string_to_param'].values()
            embed_out = next(iter(values))
        elif isinstance(embed, list):
            out_list = []
            for x in range(len(embed)):
                for k in embed[x]:
                    t = embed[x][k]
                    if t.shape[-1] != embedding_size:
                        continue
                    out_list.append(t.reshape(-1, t.shape[-1]))
            embed_out = torch.cat(out_list, dim=0)
        elif embed_key is not None and embed_key in embed:
            embed_out = embed[embed_key]
        else:
            embed_out = bundled_embed(embed, 'bundle_emb.', '.string_to_param.*')
            if embed_out is None:
                embed_out = bundled_embed(embed, 'bundle_emb.', '.{}'.format(embed_key))
            if embed_out is None:
                values = embed.values()
                embed_out = next(iter(values))
    return embed_out


SDTokenizerT = TypeVar('SDTokenizerT', bound='SDTokenizer')


class SDTokenizer:
    def __init__(self, tokenizer_path: torch.Tensor | bytes | bytearray | memoryview | str | Path | Traversable = None, max_length=77, pad_with_end=True, embedding_directory=None, embedding_size=768, embedding_key='clip_l', tokenizer_class=CLIPTokenizer, has_start_token=True, has_end_token=True, pad_to_max_length=True, min_length=None, pad_token=None, end_token=None, min_padding=None, tokenizer_data=None, tokenizer_args=None):
        if tokenizer_data is None:
            tokenizer_data = dict()
        if tokenizer_args is None:
            tokenizer_args = dict()
        if tokenizer_path is None:
            tokenizer_path = files.get_package_as_path("comfy.sd1_tokenizer")
        if isinstance(tokenizer_path, Path):
            tokenizer_path = str(tokenizer_path)
        if isinstance(tokenizer_path, Traversable):
            contextlib_path = importlib.resources.as_file(tokenizer_path)
            tokenizer_path = contextlib_path.__enter__()
        if issubclass(tokenizer_class, CLIPTokenizer) and not os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
            # assumes sd1_tokenizer
            tokenizer_path = get_package_as_path('comfy.sd1_tokenizer')
        self.tokenizer_class = tokenizer_class
        self.tokenizer_path = tokenizer_path
        self.tokenizer: PreTrainedTokenizerBase | SPieceTokenizer = tokenizer_class.from_pretrained(tokenizer_path, **tokenizer_args)
        self.max_length = tokenizer_data.get("{}_max_length".format(embedding_key), max_length)
        self.min_length = tokenizer_data.get("{}_min_length".format(embedding_key), min_length)
        self.end_token = None
        self.min_padding = min_padding

        empty = self.tokenizer('')["input_ids"]
        self.tokenizer_adds_end_token = has_end_token
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            if end_token is not None:
                self.end_token = end_token
            else:
                if has_end_token:
                    self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            if end_token is not None:
                self.end_token = end_token
            else:
                if has_end_token:
                    self.end_token = empty[0]

        if pad_token is not None:
            self.pad_token = pad_token
        elif pad_with_end:
            self.pad_token = self.end_token
        else:
            self.pad_token = 0

        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        self.additional_tokens: Tuple[str, ...] = ()
        self.add_tokens([])
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def clone(self) -> SDTokenizerT:
        sd_tokenizer = copy.copy(self)
        # correctly copy additional vocab
        sd_tokenizer.tokenizer = self.tokenizer_class.from_pretrained(self.tokenizer_path, legacy=True)
        sd_tokenizer.add_tokens(sd_tokenizer.additional_tokens)
        return sd_tokenizer

    def add_tokens(self, tokens: Sequence[str]):
        self.additional_tokens += tuple(tokens)
        if len(tokens) > 0:
            self.tokenizer.add_tokens(list(tokens))
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def _try_get_embedding(self, embedding_name: str):
        '''
        Takes a potential embedding name and tries to retrieve it.
        Returns a Tuple consisting of the embedding and any leftover string, embedding can be None.
        '''
        split_embed = embedding_name.split()
        embedding_name = split_embed[0]
        leftover = ' '.join(split_embed[1:])
        embed = load_embed(embedding_name, self.embedding_directory, self.embedding_size, self.embedding_key)
        if embed is None:
            stripped = embedding_name.strip(',')
            if len(stripped) < len(embedding_name):
                embed = load_embed(stripped, self.embedding_directory, self.embedding_size, self.embedding_key)
                return (embed, "{} {}".format(embedding_name[len(stripped):], leftover))
        return (embed, leftover)

    def tokenize_with_weights(self, text: str, return_word_ids=False, tokenizer_options={}, **kwargs):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed CLIP tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of CLIP
        '''
        min_length = tokenizer_options.get("{}_min_length".format(self.embedding_key), self.min_length)
        min_padding = tokenizer_options.get("{}_min_padding".format(self.embedding_key), self.min_padding)

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)
        vocab = self.tokenizer.get_vocab()

        # tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment)
            split = re.split(' {0}|\n{0}'.format(self.embedding_identifier), to_tokenize)
            to_tokenize = [split[0]]
            for i in range(1, len(split)):
                to_tokenize.append("{}{}".format(self.embedding_identifier, split[i]))

            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                # if we find an embedding, deal with the embedding
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        logging.warning(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                    # if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                end = 999999999999
                if self.tokenizer_adds_end_token:
                    end = -1
                # parse word
                exact_word = f"{word}</w>"
                if hasattr(self.tokenizer, "eos_token") and word == self.tokenizer.eos_token:
                    tokenizer_result = [self.tokenizer.eos_token_id]
                elif exact_word in vocab:
                    tokenizer_result = [vocab[exact_word]]
                else:
                    tokenizer_result = self.tokenizer(word)["input_ids"][self.tokens_start:end]
                tokens.append([(t, weight) for t in tokenizer_result])

        # reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            # determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length
            if self.end_token is not None:
                has_end_token = 1
            else:
                has_end_token = 0

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - has_end_token:
                    remaining_length = self.max_length - len(batch) - has_end_token
                    # break word in two and add end token
                    if is_large:
                        batch.extend([(t, w, i + 1) for t, w in t_group[:remaining_length]])
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    # add end token and pad
                    else:
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(self.pad_token, 1.0, 0)] * (remaining_length))
                    # start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t, w, i + 1) for t, w in t_group])
                    t_group = []

        # fill last batch
        if self.end_token is not None:
            batch.append((self.end_token, 1.0, 0))
        if min_padding is not None:
            batch.extend([(self.pad_token, 1.0, 0)] * min_padding)
        if self.pad_to_max_length and len(batch) < self.max_length:
            batch.extend([(self.pad_token, 1.0, 0)] * (self.max_length - len(batch)))
        if min_length is not None and len(batch) < min_length:
            batch.extend([(self.pad_token, 1.0, 0)] * (min_length - len(batch)))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w, _ in x] for x in batched_tokens]

        return batched_tokens

    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))

    def state_dict(self):
        return {}


SD1TokenizerT = TypeVar("SD1TokenizerT", bound="SD1Tokenizer")


class SD1Tokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data: dict=None, clip_name="l", tokenizer=SDTokenizer, name=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        if name is not None:
            self.clip_name = name
            self.clip = "{}".format(self.clip_name)
        else:
            self.clip_name = clip_name
            self.clip = "clip_{}".format(self.clip_name)
        tokenizer = tokenizer_data.get("{}_tokenizer_class".format(self.clip), tokenizer)
        self.sd_tokenizer = tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        out = {}
        out[self.clip_name] = self.sd_tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.sd_tokenizer.untokenize(token_weight_pair)

    @property
    def sd_tokenizer(self) -> SDTokenizer:
        return getattr(self, self.clip)

    @sd_tokenizer.setter
    def sd_tokenizer(self, value):
        setattr(self, self.clip, value)

    def clone(self) -> SD1TokenizerT:
        sd1_tokenizer = copy.copy(self)
        sd1_tokenizer.sd_tokenizer = self.sd_tokenizer.clone()
        return sd1_tokenizer

    def state_dict(self):
        return getattr(self, self.clip).state_dict()

class SD1CheckpointClipModel(SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options=None, textmodel_json_config=None):
        super().__init__(device=device, return_projected_pooled=False, dtype=dtype, model_options=model_options, textmodel_json_config=textmodel_json_config)
        if model_options is None:
            model_options = {}


class SD1ClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options=None, clip_name="l", clip_model=SD1CheckpointClipModel, textmodel_json_config=None, name=None, **kwargs):
        super().__init__()

        if model_options is None:
            model_options = {}
        if name is not None:
            self.clip_name = name
            self.clip = "{}".format(self.clip_name)
        else:
            self.clip_name = clip_name
            self.clip = "clip_{}".format(self.clip_name)

        clip_model = model_options.get("{}_class".format(self.clip), clip_model)
        model_options = {**model_options, "model_name": self.clip}
        setattr(self, self.clip, clip_model(device=device, dtype=dtype, model_options=model_options, textmodel_json_config=textmodel_json_config, **kwargs))

        self.dtypes = set()
        if dtype is not None:
            self.dtypes.add(dtype)

    def set_clip_options(self, options):
        getattr(self, self.clip).set_clip_options(options)

    def reset_clip_options(self):
        getattr(self, self.clip).reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs = token_weight_pairs[self.clip_name]
        out = getattr(self, self.clip).encode_token_weights(token_weight_pairs)
        return out

    def load_sd(self, sd):
        return getattr(self, self.clip).load_sd(sd)
