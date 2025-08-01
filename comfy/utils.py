"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import contextlib
import contextvars
import itertools
import json
import logging
import math
import os
import random
import struct
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from pickle import UnpicklingError
from typing import Optional, Any, Literal

import numpy as np
import safetensors.torch
import torch
from PIL import Image
from einops import rearrange
from torch.nn.functional import interpolate
from tqdm import tqdm
from typing_extensions import TypedDict, NotRequired

from comfy_execution.progress import get_progress_state
from . import interruption, checkpoint_pickle
from .cli_args import args
from .component_model import files
from .component_model.deprecation import _deprecate_method
from .component_model.executor_types import ExecutorToClientProgress, ProgressMessage
from .execution_context import current_execution_context, ExecutionContext
from .gguf import gguf_sd_loader

MMAP_TORCH_FILES = args.mmap_torch_files
DISABLE_MMAP = args.disable_mmap

logger = logging.getLogger(__name__)

ALWAYS_SAFE_LOAD = False
if hasattr(torch.serialization, "add_safe_globals"):  # TODO: this was added in pytorch 2.4, the unsafe path should be removed once earlier versions are deprecated
    class ModelCheckpoint:
        pass


    ModelCheckpoint.__module__ = "pytorch_lightning.callbacks.model_checkpoint"

    from numpy.core.multiarray import scalar  # pylint: disable=no-name-in-module
    from numpy import dtype
    from numpy.dtypes import Float64DType  # pylint: disable=no-name-in-module
    from _codecs import encode

    torch.serialization.add_safe_globals([ModelCheckpoint, scalar, dtype, Float64DType, encode])
    ALWAYS_SAFE_LOAD = True
    logger.debug("Checkpoint files will always be loaded safely.")
else:
    logger.debug("Warning, you are using an old pytorch version and some ckpt/pt files might be loaded unsafely. Upgrading to 2.4 or above is recommended.")


# deprecate PROGRESS_BAR_ENABLED
def _get_progress_bar_enabled():
    warnings.warn(
        "The global variable 'PROGRESS_BAR_ENABLED' is deprecated and will be removed in a future version. Use current_execution_context().server.receive_all_progress_notifications instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return current_execution_context().server.receive_all_progress_notifications


setattr(sys.modules[__name__], 'PROGRESS_BAR_ENABLED', property(_get_progress_bar_enabled))


class FileMetadata(TypedDict):
    format: NotRequired[Literal["gguf"]]


def load_torch_file(ckpt: str, safe_load=False, device=None, return_metadata=False) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], Optional[FileMetadata]]:
    if device is None:
        device = torch.device("cpu")
    if ckpt is None:
        raise FileNotFoundError("the checkpoint was not found")
    metadata: Optional[dict[str, str]] = None
    sd: dict[str, torch.Tensor] = None
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        try:
            with safetensors.safe_open(Path(ckpt).resolve(strict=True), framework="pt", device=device.type) as f:
                sd = {}
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    if DISABLE_MMAP:  # TODO: Not sure if this is the best way to bypass the mmap issues
                        tensor = tensor.to(device=device, copy=True)
                    sd[k] = tensor
                if return_metadata:
                    metadata = f.metadata()
        except Exception as e:
            if len(e.args) > 0:
                message = e.args[0]
                if "HeaderTooLarge" in message:
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(message, ckpt))
                if "MetadataIncompleteBuffer" in message:
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(message, ckpt))
            raise e
    elif ckpt.lower().endswith("index.json"):
        # from accelerate
        index_filename = ckpt
        checkpoint_folder = os.path.split(index_filename)[0]
        with open(index_filename) as f:
            index = json.loads(f.read())

        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]
        sd: dict[str, torch.Tensor] = {}
        for checkpoint_file in checkpoint_files:
            sd.update(safetensors.torch.load_file(str(checkpoint_file), device=device.type))
    elif ckpt.lower().endswith(".gguf"):
        # from gguf
        sd = gguf_sd_loader(ckpt)
        metadata = {"format": "gguf"}
    else:
        try:
            torch_args = {}
            if MMAP_TORCH_FILES:
                torch_args["mmap"] = True

            if safe_load or ALWAYS_SAFE_LOAD:
                pl_sd = torch.load(ckpt, map_location=device, weights_only=True, **torch_args)
            else:
                logging.warning("WARNING: loading {} unsafely, upgrade your pytorch to 2.4 or newer to load this file safely.".format(ckpt))
                pl_sd = torch.load(ckpt, map_location=device, pickle_module=checkpoint_pickle)
            if "state_dict" in pl_sd:
                sd = pl_sd["state_dict"]
            else:
                if len(pl_sd) == 1:
                    key = list(pl_sd.keys())[0]
                    sd = pl_sd[key]
                    if not isinstance(sd, dict):
                        sd = pl_sd
                else:
                    sd = pl_sd
        except UnpicklingError as exc_info:
            try:
                # wrong extension is most likely, try to load as safetensors anyway
                sd = safetensors.torch.load_file(Path(ckpt).resolve(strict=True), device=device.type)
            except Exception:
                msg = f"The checkpoint at {ckpt} could not be loaded as a safetensor nor a torch checkpoint. The file at the path is corrupted or unexpected. Try deleting it and downloading it again"
                if hasattr(exc_info, "add_note"):
                    exc_info.add_note(msg)
                else:
                    logger.error(msg, exc_info=exc_info)
            if sd is None:
                raise exc_info
    return (sd, metadata) if return_metadata else sd


def save_torch_file(sd, ckpt, metadata=None):
    if metadata is not None:
        safetensors.torch.save_file(sd, ckpt, metadata=metadata)
    else:
        safetensors.torch.save_file(sd, ckpt)


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            w = sd[k]
            params += w.nelement()
    return params


def weight_dtype(sd, prefix=""):
    dtypes = {}
    for k in sd.keys():
        if k.startswith(prefix):
            w = sd[k]
            dtypes[w.dtype] = dtypes.get(w.dtype, 0) + w.numel()

    if len(dtypes) == 0:
        return None

    return max(dtypes, key=dtypes.get)


def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def transformers_convert(sd, prefix_from, prefix_to, number):
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from * x:shape_from * (x + 1)]

    return sd


def clip_text_transformers_convert(sd, prefix_from, prefix_to):
    sd = transformers_convert(sd, prefix_from, "{}text_model.".format(prefix_to), 32)

    tp = "{}text_projection.weight".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = sd.pop(tp)

    tp = "{}text_projection".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = sd.pop(tp).transpose(0, 1).contiguous()
    return sd


UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
}

UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias")
}


def unet_to_diffusers(unet_config):
    if "num_res_blocks" not in unet_config:
        return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)

    i = 0
    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    return diffusers_unet_map


def swap_scale_shift(weight):
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


MMDIT_MAP_BASIC = {
    ("context_embedder.bias", "context_embedder.bias"),
    ("context_embedder.weight", "context_embedder.weight"),
    ("t_embedder.mlp.0.bias", "time_text_embed.timestep_embedder.linear_1.bias"),
    ("t_embedder.mlp.0.weight", "time_text_embed.timestep_embedder.linear_1.weight"),
    ("t_embedder.mlp.2.bias", "time_text_embed.timestep_embedder.linear_2.bias"),
    ("t_embedder.mlp.2.weight", "time_text_embed.timestep_embedder.linear_2.weight"),
    ("x_embedder.proj.bias", "pos_embed.proj.bias"),
    ("x_embedder.proj.weight", "pos_embed.proj.weight"),
    ("y_embedder.mlp.0.bias", "time_text_embed.text_embedder.linear_1.bias"),
    ("y_embedder.mlp.0.weight", "time_text_embed.text_embedder.linear_1.weight"),
    ("y_embedder.mlp.2.bias", "time_text_embed.text_embedder.linear_2.bias"),
    ("y_embedder.mlp.2.weight", "time_text_embed.text_embedder.linear_2.weight"),
    ("pos_embed", "pos_embed.pos_embed"),
    ("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias", swap_scale_shift),
    ("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight", swap_scale_shift),
    ("final_layer.linear.bias", "proj_out.bias"),
    ("final_layer.linear.weight", "proj_out.weight"),
}

MMDIT_MAP_BLOCK = {
    ("context_block.adaLN_modulation.1.bias", "norm1_context.linear.bias"),
    ("context_block.adaLN_modulation.1.weight", "norm1_context.linear.weight"),
    ("context_block.attn.proj.bias", "attn.to_add_out.bias"),
    ("context_block.attn.proj.weight", "attn.to_add_out.weight"),
    ("context_block.mlp.fc1.bias", "ff_context.net.0.proj.bias"),
    ("context_block.mlp.fc1.weight", "ff_context.net.0.proj.weight"),
    ("context_block.mlp.fc2.bias", "ff_context.net.2.bias"),
    ("context_block.mlp.fc2.weight", "ff_context.net.2.weight"),
    ("context_block.attn.ln_q.weight", "attn.norm_added_q.weight"),
    ("context_block.attn.ln_k.weight", "attn.norm_added_k.weight"),
    ("x_block.adaLN_modulation.1.bias", "norm1.linear.bias"),
    ("x_block.adaLN_modulation.1.weight", "norm1.linear.weight"),
    ("x_block.attn.proj.bias", "attn.to_out.0.bias"),
    ("x_block.attn.proj.weight", "attn.to_out.0.weight"),
    ("x_block.attn.ln_q.weight", "attn.norm_q.weight"),
    ("x_block.attn.ln_k.weight", "attn.norm_k.weight"),
    ("x_block.attn2.proj.bias", "attn2.to_out.0.bias"),
    ("x_block.attn2.proj.weight", "attn2.to_out.0.weight"),
    ("x_block.attn2.ln_q.weight", "attn2.norm_q.weight"),
    ("x_block.attn2.ln_k.weight", "attn2.norm_k.weight"),
    ("x_block.mlp.fc1.bias", "ff.net.0.proj.bias"),
    ("x_block.mlp.fc1.weight", "ff.net.0.proj.weight"),
    ("x_block.mlp.fc2.bias", "ff.net.2.bias"),
    ("x_block.mlp.fc2.weight", "ff.net.2.weight"),
}


def mmdit_to_diffusers(mmdit_config, output_prefix=""):
    key_map = {}

    depth = mmdit_config.get("depth", 0)
    num_blocks = mmdit_config.get("num_blocks", depth)
    for i in range(num_blocks):
        block_from = "transformer_blocks.{}".format(i)
        block_to = "{}joint_blocks.{}".format(output_prefix, i)

        offset = depth * 64

        for end in ("weight", "bias"):
            k = "{}.attn.".format(block_from)
            qkv = "{}.x_block.attn.qkv.{}".format(block_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, offset))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, offset, offset))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, offset * 2, offset))

            qkv = "{}.context_block.attn.qkv.{}".format(block_to, end)
            key_map["{}add_q_proj.{}".format(k, end)] = (qkv, (0, 0, offset))
            key_map["{}add_k_proj.{}".format(k, end)] = (qkv, (0, offset, offset))
            key_map["{}add_v_proj.{}".format(k, end)] = (qkv, (0, offset * 2, offset))

            k = "{}.attn2.".format(block_from)
            qkv = "{}.x_block.attn2.qkv.{}".format(block_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, offset))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, offset, offset))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, offset * 2, offset))

        for k in MMDIT_MAP_BLOCK:
            key_map["{}.{}".format(block_from, k[1])] = "{}.{}".format(block_to, k[0])

    map_basic = MMDIT_MAP_BASIC.copy()
    map_basic.add(("joint_blocks.{}.context_block.adaLN_modulation.1.bias".format(depth - 1), "transformer_blocks.{}.norm1_context.linear.bias".format(depth - 1), swap_scale_shift))
    map_basic.add(("joint_blocks.{}.context_block.adaLN_modulation.1.weight".format(depth - 1), "transformer_blocks.{}.norm1_context.linear.weight".format(depth - 1), swap_scale_shift))

    for k in map_basic:
        if len(k) > 2:
            key_map[k[1]] = ("{}{}".format(output_prefix, k[0]), None, k[2])
        else:
            key_map[k[1]] = "{}{}".format(output_prefix, k[0])

    return key_map


PIXART_MAP_BASIC = {
    ("csize_embedder.mlp.0.weight", "adaln_single.emb.resolution_embedder.linear_1.weight"),
    ("csize_embedder.mlp.0.bias", "adaln_single.emb.resolution_embedder.linear_1.bias"),
    ("csize_embedder.mlp.2.weight", "adaln_single.emb.resolution_embedder.linear_2.weight"),
    ("csize_embedder.mlp.2.bias", "adaln_single.emb.resolution_embedder.linear_2.bias"),
    ("ar_embedder.mlp.0.weight", "adaln_single.emb.aspect_ratio_embedder.linear_1.weight"),
    ("ar_embedder.mlp.0.bias", "adaln_single.emb.aspect_ratio_embedder.linear_1.bias"),
    ("ar_embedder.mlp.2.weight", "adaln_single.emb.aspect_ratio_embedder.linear_2.weight"),
    ("ar_embedder.mlp.2.bias", "adaln_single.emb.aspect_ratio_embedder.linear_2.bias"),
    ("x_embedder.proj.weight", "pos_embed.proj.weight"),
    ("x_embedder.proj.bias", "pos_embed.proj.bias"),
    ("y_embedder.y_embedding", "caption_projection.y_embedding"),
    ("y_embedder.y_proj.fc1.weight", "caption_projection.linear_1.weight"),
    ("y_embedder.y_proj.fc1.bias", "caption_projection.linear_1.bias"),
    ("y_embedder.y_proj.fc2.weight", "caption_projection.linear_2.weight"),
    ("y_embedder.y_proj.fc2.bias", "caption_projection.linear_2.bias"),
    ("t_embedder.mlp.0.weight", "adaln_single.emb.timestep_embedder.linear_1.weight"),
    ("t_embedder.mlp.0.bias", "adaln_single.emb.timestep_embedder.linear_1.bias"),
    ("t_embedder.mlp.2.weight", "adaln_single.emb.timestep_embedder.linear_2.weight"),
    ("t_embedder.mlp.2.bias", "adaln_single.emb.timestep_embedder.linear_2.bias"),
    ("t_block.1.weight", "adaln_single.linear.weight"),
    ("t_block.1.bias", "adaln_single.linear.bias"),
    ("final_layer.linear.weight", "proj_out.weight"),
    ("final_layer.linear.bias", "proj_out.bias"),
    ("final_layer.scale_shift_table", "scale_shift_table"),
}

PIXART_MAP_BLOCK = {
    ("scale_shift_table", "scale_shift_table"),
    ("attn.proj.weight", "attn1.to_out.0.weight"),
    ("attn.proj.bias", "attn1.to_out.0.bias"),
    ("mlp.fc1.weight", "ff.net.0.proj.weight"),
    ("mlp.fc1.bias", "ff.net.0.proj.bias"),
    ("mlp.fc2.weight", "ff.net.2.weight"),
    ("mlp.fc2.bias", "ff.net.2.bias"),
    ("cross_attn.proj.weight", "attn2.to_out.0.weight"),
    ("cross_attn.proj.bias", "attn2.to_out.0.bias"),
}


def pixart_to_diffusers(mmdit_config, output_prefix=""):
    key_map = {}

    depth = mmdit_config.get("depth", 0)
    offset = mmdit_config.get("hidden_size", 1152)

    for i in range(depth):
        block_from = "transformer_blocks.{}".format(i)
        block_to = "{}blocks.{}".format(output_prefix, i)

        for end in ("weight", "bias"):
            s = "{}.attn1.".format(block_from)
            qkv = "{}.attn.qkv.{}".format(block_to, end)
            key_map["{}to_q.{}".format(s, end)] = (qkv, (0, 0, offset))
            key_map["{}to_k.{}".format(s, end)] = (qkv, (0, offset, offset))
            key_map["{}to_v.{}".format(s, end)] = (qkv, (0, offset * 2, offset))

            s = "{}.attn2.".format(block_from)
            q = "{}.cross_attn.q_linear.{}".format(block_to, end)
            kv = "{}.cross_attn.kv_linear.{}".format(block_to, end)

            key_map["{}to_q.{}".format(s, end)] = q
            key_map["{}to_k.{}".format(s, end)] = (kv, (0, 0, offset))
            key_map["{}to_v.{}".format(s, end)] = (kv, (0, offset, offset))

        for k in PIXART_MAP_BLOCK:
            key_map["{}.{}".format(block_from, k[1])] = "{}.{}".format(block_to, k[0])

    for k in PIXART_MAP_BASIC:
        key_map[k[1]] = "{}{}".format(output_prefix, k[0])

    return key_map


def auraflow_to_diffusers(mmdit_config, output_prefix=""):
    n_double_layers = mmdit_config.get("n_double_layers", 0)
    n_layers = mmdit_config.get("n_layers", 0)

    key_map = {}
    for i in range(n_layers):
        if i < n_double_layers:
            index = i
            prefix_from = "joint_transformer_blocks"
            prefix_to = "{}double_layers".format(output_prefix)
            block_map = {
                "attn.to_q.weight": "attn.w2q.weight",
                "attn.to_k.weight": "attn.w2k.weight",
                "attn.to_v.weight": "attn.w2v.weight",
                "attn.to_out.0.weight": "attn.w2o.weight",
                "attn.add_q_proj.weight": "attn.w1q.weight",
                "attn.add_k_proj.weight": "attn.w1k.weight",
                "attn.add_v_proj.weight": "attn.w1v.weight",
                "attn.to_add_out.weight": "attn.w1o.weight",
                "ff.linear_1.weight": "mlpX.c_fc1.weight",
                "ff.linear_2.weight": "mlpX.c_fc2.weight",
                "ff.out_projection.weight": "mlpX.c_proj.weight",
                "ff_context.linear_1.weight": "mlpC.c_fc1.weight",
                "ff_context.linear_2.weight": "mlpC.c_fc2.weight",
                "ff_context.out_projection.weight": "mlpC.c_proj.weight",
                "norm1.linear.weight": "modX.1.weight",
                "norm1_context.linear.weight": "modC.1.weight",
            }
        else:
            index = i - n_double_layers
            prefix_from = "single_transformer_blocks"
            prefix_to = "{}single_layers".format(output_prefix)

            block_map = {
                "attn.to_q.weight": "attn.w1q.weight",
                "attn.to_k.weight": "attn.w1k.weight",
                "attn.to_v.weight": "attn.w1v.weight",
                "attn.to_out.0.weight": "attn.w1o.weight",
                "norm1.linear.weight": "modCX.1.weight",
                "ff.linear_1.weight": "mlp.c_fc1.weight",
                "ff.linear_2.weight": "mlp.c_fc2.weight",
                "ff.out_projection.weight": "mlp.c_proj.weight"
            }

        for k in block_map:
            key_map["{}.{}.{}".format(prefix_from, index, k)] = "{}.{}.{}".format(prefix_to, index, block_map[k])

    MAP_BASIC = {
        ("positional_encoding", "pos_embed.pos_embed"),
        ("register_tokens", "register_tokens"),
        ("t_embedder.mlp.0.weight", "time_step_proj.linear_1.weight"),
        ("t_embedder.mlp.0.bias", "time_step_proj.linear_1.bias"),
        ("t_embedder.mlp.2.weight", "time_step_proj.linear_2.weight"),
        ("t_embedder.mlp.2.bias", "time_step_proj.linear_2.bias"),
        ("cond_seq_linear.weight", "context_embedder.weight"),
        ("init_x_linear.weight", "pos_embed.proj.weight"),
        ("init_x_linear.bias", "pos_embed.proj.bias"),
        ("final_linear.weight", "proj_out.weight"),
        ("modF.1.weight", "norm_out.linear.weight", swap_scale_shift),
    }

    for k in MAP_BASIC:
        if len(k) > 2:
            key_map[k[1]] = ("{}{}".format(output_prefix, k[0]), None, k[2])
        else:
            key_map[k[1]] = "{}{}".format(output_prefix, k[0])

    return key_map


def flux_to_diffusers(mmdit_config, output_prefix=""):
    n_double_layers = mmdit_config.get("depth", 0)
    n_single_layers = mmdit_config.get("depth_single_blocks", 0)
    hidden_size = mmdit_config.get("hidden_size", 0)

    key_map = {}
    for index in range(n_double_layers):
        prefix_from = "transformer_blocks.{}".format(index)
        prefix_to = "{}double_blocks.{}".format(output_prefix, index)

        for end in ("weight", "bias"):
            k = "{}.attn.".format(prefix_from)
            qkv = "{}.img_attn.qkv.{}".format(prefix_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, hidden_size))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, hidden_size, hidden_size))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, hidden_size * 2, hidden_size))

            k = "{}.attn.".format(prefix_from)
            qkv = "{}.txt_attn.qkv.{}".format(prefix_to, end)
            key_map["{}add_q_proj.{}".format(k, end)] = (qkv, (0, 0, hidden_size))
            key_map["{}add_k_proj.{}".format(k, end)] = (qkv, (0, hidden_size, hidden_size))
            key_map["{}add_v_proj.{}".format(k, end)] = (qkv, (0, hidden_size * 2, hidden_size))

        block_map = {
            "attn.to_out.0.weight": "img_attn.proj.weight",
            "attn.to_out.0.bias": "img_attn.proj.bias",
            "norm1.linear.weight": "img_mod.lin.weight",
            "norm1.linear.bias": "img_mod.lin.bias",
            "norm1_context.linear.weight": "txt_mod.lin.weight",
            "norm1_context.linear.bias": "txt_mod.lin.bias",
            "attn.to_add_out.weight": "txt_attn.proj.weight",
            "attn.to_add_out.bias": "txt_attn.proj.bias",
            "ff.net.0.proj.weight": "img_mlp.0.weight",
            "ff.net.0.proj.bias": "img_mlp.0.bias",
            "ff.net.2.weight": "img_mlp.2.weight",
            "ff.net.2.bias": "img_mlp.2.bias",
            "ff_context.net.0.proj.weight": "txt_mlp.0.weight",
            "ff_context.net.0.proj.bias": "txt_mlp.0.bias",
            "ff_context.net.2.weight": "txt_mlp.2.weight",
            "ff_context.net.2.bias": "txt_mlp.2.bias",
            "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
            "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
            "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
            "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
        }

        for k in block_map:
            key_map["{}.{}".format(prefix_from, k)] = "{}.{}".format(prefix_to, block_map[k])

    for index in range(n_single_layers):
        prefix_from = "single_transformer_blocks.{}".format(index)
        prefix_to = "{}single_blocks.{}".format(output_prefix, index)

        for end in ("weight", "bias"):
            k = "{}.attn.".format(prefix_from)
            qkv = "{}.linear1.{}".format(prefix_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, hidden_size))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, hidden_size, hidden_size))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, hidden_size * 2, hidden_size))
            key_map["{}.proj_mlp.{}".format(prefix_from, end)] = (qkv, (0, hidden_size * 3, hidden_size * 4))

        block_map = {
            "norm.linear.weight": "modulation.lin.weight",
            "norm.linear.bias": "modulation.lin.bias",
            "proj_out.weight": "linear2.weight",
            "proj_out.bias": "linear2.bias",
            "attn.norm_q.weight": "norm.query_norm.scale",
            "attn.norm_k.weight": "norm.key_norm.scale",
        }

        for k in block_map:
            key_map["{}.{}".format(prefix_from, k)] = "{}.{}".format(prefix_to, block_map[k])

    MAP_BASIC = {
        ("final_layer.linear.bias", "proj_out.bias"),
        ("final_layer.linear.weight", "proj_out.weight"),
        ("img_in.bias", "x_embedder.bias"),
        ("img_in.weight", "x_embedder.weight"),
        ("time_in.in_layer.bias", "time_text_embed.timestep_embedder.linear_1.bias"),
        ("time_in.in_layer.weight", "time_text_embed.timestep_embedder.linear_1.weight"),
        ("time_in.out_layer.bias", "time_text_embed.timestep_embedder.linear_2.bias"),
        ("time_in.out_layer.weight", "time_text_embed.timestep_embedder.linear_2.weight"),
        ("txt_in.bias", "context_embedder.bias"),
        ("txt_in.weight", "context_embedder.weight"),
        ("vector_in.in_layer.bias", "time_text_embed.text_embedder.linear_1.bias"),
        ("vector_in.in_layer.weight", "time_text_embed.text_embedder.linear_1.weight"),
        ("vector_in.out_layer.bias", "time_text_embed.text_embedder.linear_2.bias"),
        ("vector_in.out_layer.weight", "time_text_embed.text_embedder.linear_2.weight"),
        ("guidance_in.in_layer.bias", "time_text_embed.guidance_embedder.linear_1.bias"),
        ("guidance_in.in_layer.weight", "time_text_embed.guidance_embedder.linear_1.weight"),
        ("guidance_in.out_layer.bias", "time_text_embed.guidance_embedder.linear_2.bias"),
        ("guidance_in.out_layer.weight", "time_text_embed.guidance_embedder.linear_2.weight"),
        ("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias", swap_scale_shift),
        ("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight", swap_scale_shift),
        ("pos_embed_input.bias", "controlnet_x_embedder.bias"),
        ("pos_embed_input.weight", "controlnet_x_embedder.weight"),
    }

    for k in MAP_BASIC:
        if len(k) > 2:
            key_map[k[1]] = ("{}{}".format(output_prefix, k[0]), None, k[2])
        else:
            key_map[k[1]] = "{}{}".format(output_prefix, k[0])

    return key_map


def repeat_to_batch_size(tensor, batch_size, dim=0):
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor


def resize_to_batch_size(tensor, batch_size):
    in_batch_size = tensor.shape[0]
    if in_batch_size == batch_size:
        return tensor

    if batch_size <= 1:
        return tensor[:batch_size]

    output = torch.empty([batch_size] + list(tensor.shape)[1:], dtype=tensor.dtype, device=tensor.device)
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output[i] = tensor[min(round(i * scale), in_batch_size - 1)]
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            output[i] = tensor[min(math.floor((i + 0.5) * scale), in_batch_size - 1)]

    return output


def resize_list_to_batch_size(l, batch_size):
    in_batch_size = len(l)
    if in_batch_size == batch_size or in_batch_size == 0:
        return l

    if batch_size <= 1:
        return l[:batch_size]

    output = []
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output.append(l[min(round(i * scale), in_batch_size - 1)])
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            output.append(l[min(math.floor((i + 0.5) * scale), in_batch_size - 1)])

    return output


def convert_sd_to(state_dict, dtype):
    keys = list(state_dict.keys())
    for k in keys:
        state_dict[k] = state_dict[k].to(dtype)
    return state_dict


def safetensors_header(safetensors_path, max_size=100 * 1024 * 1024):
    with open(safetensors_path, "rb") as f:
        header = f.read(8)
        length_of_header = struct.unpack('<Q', header)[0]
        if length_of_header > max_size:
            return None
        return f.read(length_of_header)


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], value)
    return prev


def set_attr_param(obj, attr, value):
    return set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))


def copy_to_param(obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def get_attr(obj, attr: str):
    """Retrieves a nested attribute from an object using dot notation.

    Args:
        obj: The object to get the attribute from
        attr (str): The attribute path using dot notation (e.g. "model.layer.weight")

    Returns:
        The value of the requested attribute

    Example:
        model = MyModel()
        weight = get_attr(model, "layer1.conv.weight")
        # Equivalent to: model.layer1.conv.weight

    Important:
        Always prefer `comfy.model_patcher.ModelPatcher.get_model_object` when
        accessing nested model objects under `ModelPatcher.model`.
    """
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        '''slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC'''

        c = b1.shape[-1]

        # norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        # normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        # zero when norms are zero
        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

        # slerp
        dot = (b1_normalized * b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        # technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0 - r.squeeze(1)) * omega) / so).unsqueeze(1) * b1_normalized + (torch.sin(r.squeeze(1) * omega) / so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c)

        # edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1, 1, 1, -1))
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1, 1, 1, -1)) + 1
        coords_2[:, :, :, -1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n, c, h, w = samples.shape
    h_new, w_new = (height, width)

    # linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    # linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1, 1, -1, 1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)


def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)


def common_upscale(samples, width, height, upscale_method, crop):
    orig_shape = tuple(samples.shape)
    if len(orig_shape) > 4:
        samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
    if crop == "center":
        old_width = samples.shape[-1]
        old_height = samples.shape[-2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    else:
        s = samples

    if upscale_method == "bislerp":
        out = bislerp(s, width, height)
    elif upscale_method == "lanczos":
        out = lanczos(s, width, height)
    else:
        out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

    if len(orig_shape) == 4:
        return out

    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols


@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", downscale=False, index_formulas=None, pbar=None):
    dims = len(tile)

    if not (isinstance(upscale_amount, (tuple, list))):
        upscale_amount = [upscale_amount] * dims

    if not (isinstance(overlap, (tuple, list))):
        overlap = [overlap] * dims

    if index_formulas is None:
        index_formulas = upscale_amount

    if not (isinstance(index_formulas, (tuple, list))):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    def get_upscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    if downscale:
        get_scale = get_downscale
        get_pos = get_downscale_pos
    else:
        get_scale = get_upscale
        get_pos = get_upscale_pos

    def mult_list_upscale(a):
        out = []
        for i in range(len(a)):
            out.append(round(get_scale(i, a[i])))
        return out

    output = torch.empty([samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), device=output_device)

    for b in range(samples.shape[0]):
        s = samples[b:b + 1]

        # handle entire input fitting in a single tile
        if all(s.shape[d + 2] <= tile[d] for d in range(dims)):
            output[b:b + 1] = function(s).to(output_device)
            if pbar is not None:
                pbar.update(1)
            continue
        out = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)
        out_div = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)

        positions = [range(0, s.shape[d + 2] - overlap[d], tile[d] - overlap[d]) if s.shape[d + 2] > tile[d] else [0] for d in range(dims)]

        for it in itertools.product(*positions):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(get_pos(d, pos)))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)

            for d in range(2, dims + 2):
                feather = round(get_scale(d - 2, overlap[d - 2]))
                if feather >= mask.shape[d]:
                    continue
                for t in range(feather):
                    a = (t + 1) / feather
                    mask.narrow(d, t, 1).mul_(a)
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o.add_(ps * mask)
            o_d.add_(mask)

            if pbar is not None:
                pbar.update(1)

        output[b:b + 1] = out / out_div
    return output


def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", pbar=None):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap=overlap, upscale_amount=upscale_amount, out_channels=out_channels, output_device=output_device, pbar=pbar)


def _progress_bar_update(value: float, total: float, preview_image_or_data: Optional[Any] = None, client_id: Optional[str] = None, server: Optional[ExecutorToClientProgress] = None, node_id: str = None, prompt_id: str = None):
    context = current_execution_context()
    server = server or context.server
    executing_context = context
    prompt_id = prompt_id or executing_context.task_id or server.last_prompt_id
    node_id = node_id or executing_context.node_id or server.last_node_id
    interruption.throw_exception_if_processing_interrupted()

    progress: ProgressMessage = {"value": value, "max": total, "prompt_id": prompt_id, "node": node_id}
    # todo: is this still necessary?
    if isinstance(preview_image_or_data, dict):
        progress["output"] = preview_image_or_data

    # this is responsible for encoding the image
    get_progress_state().update_progress(node_id, value, total, preview_image_or_data)
    server.send_sync("progress", progress, client_id)


def set_progress_bar_enabled(enabled: bool):
    warnings.warn(
        "The global method 'set_progress_bar_enabled' is deprecated and will be removed in a future version. Use current_execution_context().server.receive_all_progress_notifications instead.",
        DeprecationWarning,
        stacklevel=2
    )

    current_execution_context().server.receive_all_progress_notifications = enabled
    pass


def get_progress_bar_enabled() -> bool:
    warnings.warn(
        "The global method 'get_progress_bar_enabled' is deprecated and will be removed in a future version. Use current_execution_context().server.receive_all_progress_notifications instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return current_execution_context().server.receive_all_progress_notifications


class _DisabledProgressBar:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def update_absolute(self, *args, **kwargs):
        pass


class ProgressBar:
    def __init__(self, total: float, node_id: Any = None):
        self.total: float = total
        self.current: float = 0.0
        self.server = current_execution_context().server
        self.node_id = node_id

    def update_absolute(self, value, total=None, preview_image_or_output=None):
        if total is not None:
            self.total = total
        if value > self.total:
            value = self.total
        self.current = value
        _progress_bar_update(self.current, self.total, preview_image_or_output, server=self.server, node_id=self.node_id)

    def update(self, value):
        self.update_absolute(self.current + value)


@_deprecate_method(version="1.0.0", message="The root project directory isn't valid when the application is installed as a package. Use os.getcwd() instead.")
def get_project_root() -> str:
    return files.get_package_as_path("comfy")


@contextmanager
def comfy_tqdm():
    """
    Monky patches child calls to tqdm and sends the progress to the UI
    :return:
    """
    _original_init = tqdm.__init__
    _original_call = tqdm.__call__
    _original_update = tqdm.update
    context = contextvars.copy_context()
    try:
        def __init(self, *args, **kwargs):
            context.run(lambda: _original_init(self, *args, **kwargs))
            self._progress_bar = context.run(lambda: ProgressBar(self.total))

        def __update(self, n=1):
            assert self._progress_bar is not None
            context.run(lambda: _original_update(self, n))
            context.run(lambda: self._progress_bar.update(n))

        def __call(self, *args, **kwargs):
            # When TQDM is called to wrap an iterable, ensure the instance is created
            # with the captured context
            instance = context.run(lambda: _original_call(self, *args, **kwargs))
            return instance

        tqdm.__init__ = __init
        tqdm.__call__ = __call
        tqdm.update = __update
        # todo: modify the tqdm class here to correctly copy the context into the function that tqdm is passed
        yield
    finally:
        # Restore original tqdm
        tqdm.__init__ = _original_init
        tqdm.__call__ = _original_call
        tqdm.update = _original_update
        # todo: restore the context copying away


@contextmanager
def comfy_progress(total: float) -> ProgressBar:
    ctx = current_execution_context()
    if ctx.server.receive_all_progress_notifications:
        yield ProgressBar(total)
    else:
        yield _DisabledProgressBar()


@contextlib.contextmanager
def seed_for_block(seed):
    # Save the current random state
    torch_rng_state = torch.get_rng_state()
    random_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    # todo: investigate with torch.random.fork_rng(devices=(device,))
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state_all()
    else:
        cuda_rng_state = None

    # Set the new seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        # Restore the previous random state
        torch.set_rng_state(torch_rng_state)
        random.setstate(random_state)
        np.random.set_state(numpy_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)


def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask


def reshape_mask(input_mask, output_shape):
    dims = len(output_shape) - 2

    if dims == 1:
        scale_mode = "linear"
    elif dims == 2:
        input_mask = input_mask.reshape((-1, 1, input_mask.shape[-2], input_mask.shape[-1]))
        scale_mode = "bilinear"
    elif dims == 3:
        if len(input_mask.shape) < 5:
            input_mask = input_mask.reshape((1, 1, -1, input_mask.shape[-2], input_mask.shape[-1]))
        scale_mode = "trilinear"
    else:
        raise ValueError(f"invalid dims={dims}")

    mask = torch.nn.functional.interpolate(input_mask, size=output_shape[2:], mode=scale_mode)
    if mask.shape[1] < output_shape[1]:
        mask = mask.repeat((1, output_shape[1]) + (1,) * dims)[:, :output_shape[1]]
    mask = repeat_to_batch_size(mask, output_shape[0])
    return mask


def upscale_dit_mask(mask: torch.Tensor, img_size_in, img_size_out):
    hi, wi = img_size_in
    ho, wo = img_size_out
    # if it's already the correct size, no need to do anything
    if (hi, wi) == (ho, wo):
        return mask
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Got a mask of shape {list(mask.shape)}, expected [b, q, k] or [q, k]")
    txt_tokens = mask.shape[1] - (hi * wi)
    # quadrants of the mask
    txt_to_txt = mask[:, :txt_tokens, :txt_tokens]
    txt_to_img = mask[:, :txt_tokens, txt_tokens:]
    img_to_img = mask[:, txt_tokens:, txt_tokens:]
    img_to_txt = mask[:, txt_tokens:, :txt_tokens]

    # convert to 1d x 2d, interpolate, then back to 1d x 1d
    txt_to_img = rearrange(txt_to_img, "b t (h w) -> b t h w", h=hi, w=wi)
    txt_to_img = interpolate(txt_to_img, size=img_size_out, mode="bilinear")
    txt_to_img = rearrange(txt_to_img, "b t h w -> b t (h w)")
    # this one is hard because we have to do it twice
    # convert to 1d x 2d, interpolate, then to 2d x 1d, interpolate, then 1d x 1d
    img_to_img = rearrange(img_to_img, "b hw (h w) -> b hw h w", h=hi, w=wi)
    img_to_img = interpolate(img_to_img, size=img_size_out, mode="bilinear")
    img_to_img = rearrange(img_to_img, "b (hk wk) hq wq -> b (hq wq) hk wk", hk=hi, wk=wi)
    img_to_img = interpolate(img_to_img, size=img_size_out, mode="bilinear")
    img_to_img = rearrange(img_to_img, "b (hq wq) hk wk -> b (hk wk) (hq wq)", hq=ho, wq=wo)
    # convert to 2d x 1d, interpolate, then back to 1d x 1d
    img_to_txt = rearrange(img_to_txt, "b (h w) t -> b t h w", h=hi, w=wi)
    img_to_txt = interpolate(img_to_txt, size=img_size_out, mode="bilinear")
    img_to_txt = rearrange(img_to_txt, "b t h w -> b (h w) t")

    # reassemble the mask from blocks
    out = torch.cat([
        torch.cat([txt_to_txt, txt_to_img], dim=2),
        torch.cat([img_to_txt, img_to_img], dim=2)],
        dim=1
    )
    return out
