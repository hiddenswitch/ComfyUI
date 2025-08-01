import json
import logging
import math
from typing import Optional

import torch

from . import supported_models, utils
from . import supported_models_base
from .gguf import GGMLOps


def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count


def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    context_dim = None
    use_linear_in_transformer = False

    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
        context_dim = state_dict['{}0.attn2.to_k.weight'.format(transformer_prefix)].shape[1]
        use_linear_in_transformer = len(state_dict['{}1.proj_in.weight'.format(prefix)].shape) == 2
        time_stack = '{}1.time_stack.0.attn1.to_q.weight'.format(prefix) in state_dict or '{}1.time_mix_blocks.0.attn1.to_q.weight'.format(prefix) in state_dict
        time_stack_cross = '{}1.time_stack.0.attn2.to_q.weight'.format(prefix) in state_dict or '{}1.time_mix_blocks.0.attn2.to_q.weight'.format(prefix) in state_dict
        return last_transformer_depth, context_dim, use_linear_in_transformer, time_stack, time_stack_cross
    return None


def detect_unet_config(state_dict, key_prefix, metadata=None):
    state_dict_keys = list(state_dict.keys())

    if '{}joint_blocks.0.context_block.attn.qkv.weight'.format(key_prefix) in state_dict_keys:  # mmdit model
        unet_config = {}
        unet_config["in_channels"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[1]
        patch_size = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[2]
        unet_config["patch_size"] = patch_size
        final_layer = '{}final_layer.linear.weight'.format(key_prefix)
        if final_layer in state_dict:
            unet_config["out_channels"] = state_dict[final_layer].shape[0] // (patch_size * patch_size)

        unet_config["depth"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[0] // 64
        unet_config["input_size"] = None
        y_key = '{}y_embedder.mlp.0.weight'.format(key_prefix)
        if y_key in state_dict_keys:
            unet_config["adm_in_channels"] = state_dict[y_key].shape[1]

        context_key = '{}context_embedder.weight'.format(key_prefix)
        if context_key in state_dict_keys:
            in_features = state_dict[context_key].shape[1]
            out_features = state_dict[context_key].shape[0]
            unet_config["context_embedder_config"] = {"target": "torch.nn.Linear", "params": {"in_features": in_features, "out_features": out_features}}
        num_patches_key = '{}pos_embed'.format(key_prefix)
        if num_patches_key in state_dict_keys:
            num_patches = state_dict[num_patches_key].shape[1]
            unet_config["num_patches"] = num_patches
            unet_config["pos_embed_max_size"] = round(math.sqrt(num_patches))

        rms_qk = '{}joint_blocks.0.context_block.attn.ln_q.weight'.format(key_prefix)
        if rms_qk in state_dict_keys:
            unet_config["qk_norm"] = "rms"

        unet_config["pos_embed_scaling_factor"] = None  # unused for inference
        context_processor = '{}context_processor.layers.0.attn.qkv.weight'.format(key_prefix)
        if context_processor in state_dict_keys:
            unet_config["context_processor_layers"] = count_blocks(state_dict_keys, '{}context_processor.layers.'.format(key_prefix) + '{}.')
        unet_config["x_block_self_attn_layers"] = []
        for key in state_dict_keys:
            if key.startswith('{}joint_blocks.'.format(key_prefix)) and key.endswith('.x_block.attn2.qkv.weight'):
                layer = key[len('{}joint_blocks.'.format(key_prefix)):-len('.x_block.attn2.qkv.weight')]
                unet_config["x_block_self_attn_layers"].append(int(layer))
        return unet_config

    if '{}clf.1.weight'.format(key_prefix) in state_dict_keys:  # stable cascade
        unet_config = {}
        text_mapper_name = '{}clip_txt_mapper.weight'.format(key_prefix)
        if text_mapper_name in state_dict_keys:
            unet_config['stable_cascade_stage'] = 'c'
            w = state_dict[text_mapper_name]
            if w.shape[0] == 1536:  # stage c lite
                unet_config['c_cond'] = 1536
                unet_config['c_hidden'] = [1536, 1536]
                unet_config['nhead'] = [24, 24]
                unet_config['blocks'] = [[4, 12], [12, 4]]
            elif w.shape[0] == 2048:  # stage c full
                unet_config['c_cond'] = 2048
        elif '{}clip_mapper.weight'.format(key_prefix) in state_dict_keys:
            unet_config['stable_cascade_stage'] = 'b'
            w = state_dict['{}down_blocks.1.0.channelwise.0.weight'.format(key_prefix)]
            if w.shape[-1] == 640:
                unet_config['c_hidden'] = [320, 640, 1280, 1280]
                unet_config['nhead'] = [-1, -1, 20, 20]
                unet_config['blocks'] = [[2, 6, 28, 6], [6, 28, 6, 2]]
                unet_config['block_repeat'] = [[1, 1, 1, 1], [3, 3, 2, 2]]
            elif w.shape[-1] == 576:  # stage b lite
                unet_config['c_hidden'] = [320, 576, 1152, 1152]
                unet_config['nhead'] = [-1, 9, 18, 18]
                unet_config['blocks'] = [[2, 4, 14, 4], [4, 14, 4, 2]]
                unet_config['block_repeat'] = [[1, 1, 1, 1], [2, 2, 2, 2]]
        return unet_config

    if '{}transformer.rotary_pos_emb.inv_freq'.format(key_prefix) in state_dict_keys:  # stable audio dit
        unet_config = {}
        unet_config["audio_model"] = "dit1.0"
        return unet_config

    if '{}double_layers.0.attn.w1q.weight'.format(key_prefix) in state_dict_keys:  # aura flow dit
        unet_config = {}
        unet_config["max_seq"] = state_dict['{}positional_encoding'.format(key_prefix)].shape[1]
        unet_config["cond_seq_dim"] = state_dict['{}cond_seq_linear.weight'.format(key_prefix)].shape[1]
        double_layers = count_blocks(state_dict_keys, '{}double_layers.'.format(key_prefix) + '{}.')
        single_layers = count_blocks(state_dict_keys, '{}single_layers.'.format(key_prefix) + '{}.')
        unet_config["n_double_layers"] = double_layers
        unet_config["n_layers"] = double_layers + single_layers
        return unet_config

    if '{}mlp_t5.0.weight'.format(key_prefix) in state_dict_keys:  # Hunyuan DiT
        unet_config = {}
        unet_config["image_model"] = "hydit"
        unet_config["depth"] = count_blocks(state_dict_keys, '{}blocks.'.format(key_prefix) + '{}.')
        unet_config["hidden_size"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[0]
        if unet_config["hidden_size"] == 1408 and unet_config["depth"] == 40:  # DiT-g/2
            unet_config["mlp_ratio"] = 4.3637
        if state_dict['{}extra_embedder.0.weight'.format(key_prefix)].shape[1] == 3968:
            unet_config["size_cond"] = True
            unet_config["use_style_cond"] = True
            unet_config["image_model"] = "hydit1"
        return unet_config

    if '{}txt_in.individual_token_refiner.blocks.0.norm1.weight'.format(key_prefix) in state_dict_keys:  # Hunyuan Video
        dit_config = {}
        dit_config["image_model"] = "hunyuan_video"
        dit_config["in_channels"] = state_dict['{}img_in.proj.weight'.format(key_prefix)].shape[1]  # SkyReels img2video has 32 input channels
        dit_config["patch_size"] = [1, 2, 2]
        dit_config["out_channels"] = 16
        dit_config["vec_in_dim"] = 768
        dit_config["context_in_dim"] = 4096
        dit_config["hidden_size"] = 3072
        dit_config["mlp_ratio"] = 4.0
        dit_config["num_heads"] = 24
        dit_config["depth"] = count_blocks(state_dict_keys, '{}double_blocks.'.format(key_prefix) + '{}.')
        dit_config["depth_single_blocks"] = count_blocks(state_dict_keys, '{}single_blocks.'.format(key_prefix) + '{}.')
        dit_config["axes_dim"] = [16, 56, 56]
        dit_config["theta"] = 256
        dit_config["qkv_bias"] = True
        guidance_keys = list(filter(lambda a: a.startswith("{}guidance_in.".format(key_prefix)), state_dict_keys))
        dit_config["guidance_embed"] = len(guidance_keys) > 0
        return dit_config

    if '{}double_blocks.0.img_attn.norm.key_norm.scale'.format(key_prefix) in state_dict_keys and '{}img_in.weight'.format(key_prefix) in state_dict_keys:  # Flux
        dit_config = {}
        dit_config["image_model"] = "flux"
        dit_config["in_channels"] = 16
        patch_size = 2
        dit_config["patch_size"] = patch_size
        in_key = "{}img_in.weight".format(key_prefix)
        if in_key in state_dict_keys:
            dit_config["in_channels"] = state_dict[in_key].shape[1] // (patch_size * patch_size)
        dit_config["out_channels"] = 16
        vec_in_key = '{}vector_in.in_layer.weight'.format(key_prefix)
        if vec_in_key in state_dict_keys:
            dit_config["vec_in_dim"] = state_dict[vec_in_key].shape[1]
        dit_config["context_in_dim"] = 4096
        dit_config["hidden_size"] = 3072
        dit_config["mlp_ratio"] = 4.0
        dit_config["num_heads"] = 24
        dit_config["depth"] = count_blocks(state_dict_keys, '{}double_blocks.'.format(key_prefix) + '{}.')
        dit_config["depth_single_blocks"] = count_blocks(state_dict_keys, '{}single_blocks.'.format(key_prefix) + '{}.')
        dit_config["axes_dim"] = [16, 56, 56]
        dit_config["theta"] = 10000
        dit_config["qkv_bias"] = True
        if '{}distilled_guidance_layer.0.norms.0.scale'.format(key_prefix) in state_dict_keys or '{}distilled_guidance_layer.norms.0.scale'.format(key_prefix) in state_dict_keys: #Chroma
            dit_config["image_model"] = "chroma"
            dit_config["in_channels"] = 64
            dit_config["out_channels"] = 64
            dit_config["in_dim"] = 64
            dit_config["out_dim"] = 3072
            dit_config["hidden_dim"] = 5120
            dit_config["n_layers"] = 5
        else:
            dit_config["guidance_embed"] = "{}guidance_in.in_layer.weight".format(key_prefix) in state_dict_keys
        return dit_config

    if '{}t5_yproj.weight'.format(key_prefix) in state_dict_keys:  # Genmo mochi preview
        dit_config = {}
        dit_config["image_model"] = "mochi_preview"
        dit_config["depth"] = 48
        dit_config["patch_size"] = 2
        dit_config["num_heads"] = 24
        dit_config["hidden_size_x"] = 3072
        dit_config["hidden_size_y"] = 1536
        dit_config["mlp_ratio_x"] = 4.0
        dit_config["mlp_ratio_y"] = 4.0
        dit_config["learn_sigma"] = False
        dit_config["in_channels"] = 12
        dit_config["qk_norm"] = True
        dit_config["qkv_bias"] = False
        dit_config["out_bias"] = True
        dit_config["attn_drop"] = 0.0
        dit_config["patch_embed_bias"] = True
        dit_config["posenc_preserve_area"] = True
        dit_config["timestep_mlp_bias"] = True
        dit_config["attend_to_padding"] = False
        dit_config["timestep_scale"] = 1000.0
        dit_config["use_t5"] = True
        dit_config["t5_feat_dim"] = 4096
        dit_config["t5_token_length"] = 256
        dit_config["rope_theta"] = 10000.0
        return dit_config

    if '{}adaln_single.emb.timestep_embedder.linear_1.bias'.format(key_prefix) in state_dict_keys and '{}pos_embed.proj.bias'.format(key_prefix) in state_dict_keys:
        # PixArt diffusers
        return None

    if '{}adaln_single.emb.timestep_embedder.linear_1.bias'.format(key_prefix) in state_dict_keys:  # Lightricks ltxv
        dit_config = {}
        dit_config["image_model"] = "ltxv"
        dit_config["num_layers"] = count_blocks(state_dict_keys, '{}transformer_blocks.'.format(key_prefix) + '{}.')
        shape = state_dict['{}transformer_blocks.0.attn2.to_k.weight'.format(key_prefix)].shape
        dit_config["attention_head_dim"] = shape[0] // 32
        dit_config["cross_attention_dim"] = shape[1]
        if metadata is not None and "config" in metadata:
            dit_config.update(json.loads(metadata["config"]).get("transformer", {}))
        return dit_config

    if '{}genre_embedder.weight'.format(key_prefix) in state_dict_keys: #ACE-Step model
        dit_config = {}
        dit_config["audio_model"] = "ace"
        dit_config["attention_head_dim"] = 128
        dit_config["in_channels"] = 8
        dit_config["inner_dim"] = 2560
        dit_config["max_height"] = 16
        dit_config["max_position"] = 32768
        dit_config["max_width"] = 32768
        dit_config["mlp_ratio"] = 2.5
        dit_config["num_attention_heads"] = 20
        dit_config["num_layers"] = 24
        dit_config["out_channels"] = 8
        dit_config["patch_size"] = [16, 1]
        dit_config["rope_theta"] = 1000000.0
        dit_config["speaker_embedding_dim"] = 512
        dit_config["text_embedding_dim"] = 768

        dit_config["ssl_encoder_depths"] = [8, 8]
        dit_config["ssl_latent_dims"] = [1024, 768]
        dit_config["ssl_names"] = ["mert", "m-hubert"]
        dit_config["lyric_encoder_vocab_size"] = 6693
        dit_config["lyric_hidden_size"] = 1024
        return dit_config

    if '{}t_block.1.weight'.format(key_prefix) in state_dict_keys:  # PixArt
        patch_size = 2
        dit_config = {}
        dit_config["num_heads"] = 16
        dit_config["patch_size"] = patch_size
        dit_config["hidden_size"] = 1152
        dit_config["in_channels"] = 4
        dit_config["depth"] = count_blocks(state_dict_keys, '{}blocks.'.format(key_prefix) + '{}.')

        y_key = "{}y_embedder.y_embedding".format(key_prefix)
        if y_key in state_dict_keys:
            dit_config["model_max_length"] = state_dict[y_key].shape[0]

        pe_key = "{}pos_embed".format(key_prefix)
        if pe_key in state_dict_keys:
            dit_config["input_size"] = int(math.sqrt(state_dict[pe_key].shape[1])) * patch_size
            dit_config["pe_interpolation"] = dit_config["input_size"] // (512 // 8)  # guess

        ar_key = "{}ar_embedder.mlp.0.weight".format(key_prefix)
        if ar_key in state_dict_keys:
            dit_config["image_model"] = "pixart_alpha"
            dit_config["micro_condition"] = True
        else:
            dit_config["image_model"] = "pixart_sigma"
            dit_config["micro_condition"] = False
        return dit_config

    if '{}blocks.block0.blocks.0.block.attn.to_q.0.weight'.format(key_prefix) in state_dict_keys:  # Cosmos
        dit_config = {}
        dit_config["image_model"] = "cosmos"
        dit_config["max_img_h"] = 240
        dit_config["max_img_w"] = 240
        dit_config["max_frames"] = 128
        concat_padding_mask = True
        dit_config["in_channels"] = (state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[1] // 4) - int(concat_padding_mask)
        dit_config["out_channels"] = 16
        dit_config["patch_spatial"] = 2
        dit_config["patch_temporal"] = 1
        dit_config["model_channels"] = state_dict['{}blocks.block0.blocks.0.block.attn.to_q.0.weight'.format(key_prefix)].shape[0]
        dit_config["block_config"] = "FA-CA-MLP"
        dit_config["concat_padding_mask"] = concat_padding_mask
        dit_config["pos_emb_cls"] = "rope3d"
        dit_config["pos_emb_learnable"] = False
        dit_config["pos_emb_interpolation"] = "crop"
        dit_config["block_x_format"] = "THWBD"
        dit_config["affline_emb_norm"] = True
        dit_config["use_adaln_lora"] = True
        dit_config["adaln_lora_dim"] = 256

        if dit_config["model_channels"] == 4096:
            # 7B
            dit_config["num_blocks"] = 28
            dit_config["num_heads"] = 32
            dit_config["extra_per_block_abs_pos_emb"] = True
            dit_config["rope_h_extrapolation_ratio"] = 1.0
            dit_config["rope_w_extrapolation_ratio"] = 1.0
            dit_config["rope_t_extrapolation_ratio"] = 2.0
            dit_config["extra_per_block_abs_pos_emb_type"] = "learnable"
        else:  # 5120
            # 14B
            dit_config["num_blocks"] = 36
            dit_config["num_heads"] = 40
            dit_config["extra_per_block_abs_pos_emb"] = True
            dit_config["rope_h_extrapolation_ratio"] = 2.0
            dit_config["rope_w_extrapolation_ratio"] = 2.0
            dit_config["rope_t_extrapolation_ratio"] = 2.0
            dit_config["extra_h_extrapolation_ratio"] = 2.0
            dit_config["extra_w_extrapolation_ratio"] = 2.0
            dit_config["extra_t_extrapolation_ratio"] = 2.0
            dit_config["extra_per_block_abs_pos_emb_type"] = "learnable"
        return dit_config

    if '{}cap_embedder.1.weight'.format(key_prefix) in state_dict_keys:  # Lumina 2
        dit_config = {}
        dit_config["image_model"] = "lumina2"
        dit_config["patch_size"] = 2
        dit_config["in_channels"] = 16
        dit_config["dim"] = 2304
        dit_config["cap_feat_dim"] = 2304
        dit_config["n_layers"] = 26
        dit_config["n_heads"] = 24
        dit_config["n_kv_heads"] = 8
        dit_config["qk_norm"] = True
        dit_config["axes_dims"] = [32, 32, 32]
        dit_config["axes_lens"] = [300, 512, 512]
        return dit_config

    if '{}head.modulation'.format(key_prefix) in state_dict_keys:  # Wan 2.1
        dit_config = {}
        dit_config["image_model"] = "wan2.1"
        dim = state_dict['{}head.modulation'.format(key_prefix)].shape[-1]
        out_dim = state_dict['{}head.head.weight'.format(key_prefix)].shape[0] // 4
        dit_config["dim"] = dim
        dit_config["out_dim"] = out_dim
        dit_config["num_heads"] = dim // 128
        dit_config["ffn_dim"] = state_dict['{}blocks.0.ffn.0.weight'.format(key_prefix)].shape[0]
        dit_config["num_layers"] = count_blocks(state_dict_keys, '{}blocks.'.format(key_prefix) + '{}.')
        dit_config["patch_size"] = (1, 2, 2)
        dit_config["freq_dim"] = 256
        dit_config["window_size"] = (-1, -1)
        dit_config["qk_norm"] = True
        dit_config["cross_attn_norm"] = True
        dit_config["eps"] = 1e-6
        dit_config["in_dim"] = state_dict['{}patch_embedding.weight'.format(key_prefix)].shape[1]
        if '{}vace_patch_embedding.weight'.format(key_prefix) in state_dict_keys:
            dit_config["model_type"] = "vace"
            dit_config["vace_in_dim"] = state_dict['{}vace_patch_embedding.weight'.format(key_prefix)].shape[1]
            dit_config["vace_layers"] = count_blocks(state_dict_keys, '{}vace_blocks.'.format(key_prefix) + '{}.')
        elif '{}control_adapter.conv.weight'.format(key_prefix) in state_dict_keys:
            dit_config["model_type"] = "camera"
        else:
            if '{}img_emb.proj.0.bias'.format(key_prefix) in state_dict_keys:
                dit_config["model_type"] = "i2v"
            else:
                dit_config["model_type"] = "t2v"
        flf_weight = state_dict.get('{}img_emb.emb_pos'.format(key_prefix))
        if flf_weight is not None:
            dit_config["flf_pos_embed_token_number"] = flf_weight.shape[1]
        return dit_config

    if '{}latent_in.weight'.format(key_prefix) in state_dict_keys:  # Hunyuan 3D
        in_shape = state_dict['{}latent_in.weight'.format(key_prefix)].shape
        dit_config = {}
        dit_config["image_model"] = "hunyuan3d2"
        dit_config["in_channels"] = in_shape[1]
        dit_config["context_in_dim"] = state_dict['{}cond_in.weight'.format(key_prefix)].shape[1]
        dit_config["hidden_size"] = in_shape[0]
        dit_config["mlp_ratio"] = 4.0
        dit_config["num_heads"] = 16
        dit_config["depth"] = count_blocks(state_dict_keys, '{}double_blocks.'.format(key_prefix) + '{}.')
        dit_config["depth_single_blocks"] = count_blocks(state_dict_keys, '{}single_blocks.'.format(key_prefix) + '{}.')
        dit_config["qkv_bias"] = True
        dit_config["guidance_embed"] = "{}guidance_in.in_layer.weight".format(key_prefix) in state_dict_keys
        return dit_config

    if '{}caption_projection.0.linear.weight'.format(key_prefix) in state_dict_keys:  # HiDream
        dit_config = {}
        dit_config["image_model"] = "hidream"
        dit_config["attention_head_dim"] = 128
        dit_config["axes_dims_rope"] = [64, 32, 32]
        dit_config["caption_channels"] = [4096, 4096]
        dit_config["max_resolution"] = [128, 128]
        dit_config["in_channels"] = 16
        dit_config["llama_layers"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31]
        dit_config["num_attention_heads"] = 20
        dit_config["num_routed_experts"] = 4
        dit_config["num_activated_experts"] = 2
        dit_config["num_layers"] = 16
        dit_config["num_single_layers"] = 32
        dit_config["out_channels"] = 16
        dit_config["patch_size"] = 2
        dit_config["text_emb_dim"] = 2048
        return dit_config

    if '{}blocks.0.mlp.layer1.weight'.format(key_prefix) in state_dict_keys:  # Cosmos predict2
        dit_config = {}
        dit_config["image_model"] = "cosmos_predict2"
        dit_config["max_img_h"] = 240
        dit_config["max_img_w"] = 240
        dit_config["max_frames"] = 128
        concat_padding_mask = True
        dit_config["in_channels"] = (state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[1] // 4) - int(concat_padding_mask)
        dit_config["out_channels"] = 16
        dit_config["patch_spatial"] = 2
        dit_config["patch_temporal"] = 1
        dit_config["model_channels"] = state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[0]
        dit_config["concat_padding_mask"] = concat_padding_mask
        dit_config["crossattn_emb_channels"] = 1024
        dit_config["pos_emb_cls"] = "rope3d"
        dit_config["pos_emb_learnable"] = True
        dit_config["pos_emb_interpolation"] = "crop"
        dit_config["min_fps"] = 1
        dit_config["max_fps"] = 30

        dit_config["use_adaln_lora"] = True
        dit_config["adaln_lora_dim"] = 256
        if dit_config["model_channels"] == 2048:
            dit_config["num_blocks"] = 28
            dit_config["num_heads"] = 16
        elif dit_config["model_channels"] == 5120:
            dit_config["num_blocks"] = 36
            dit_config["num_heads"] = 40

        if dit_config["in_channels"] == 16:
            dit_config["extra_per_block_abs_pos_emb"] = False
            dit_config["rope_h_extrapolation_ratio"] = 4.0
            dit_config["rope_w_extrapolation_ratio"] = 4.0
            dit_config["rope_t_extrapolation_ratio"] = 1.0
        elif dit_config["in_channels"] == 17: # img to video
            if dit_config["model_channels"] == 2048:
                dit_config["extra_per_block_abs_pos_emb"] = False
                dit_config["rope_h_extrapolation_ratio"] = 3.0
                dit_config["rope_w_extrapolation_ratio"] = 3.0
                dit_config["rope_t_extrapolation_ratio"] = 1.0
            elif dit_config["model_channels"] == 5120:
                dit_config["rope_h_extrapolation_ratio"] = 2.0
                dit_config["rope_w_extrapolation_ratio"] = 2.0
                dit_config["rope_t_extrapolation_ratio"] = 0.8333333333333334

        dit_config["extra_h_extrapolation_ratio"] = 1.0
        dit_config["extra_w_extrapolation_ratio"] = 1.0
        dit_config["extra_t_extrapolation_ratio"] = 1.0
        dit_config["rope_enable_fps_modulation"] = False

        return dit_config

    if '{}time_caption_embed.timestep_embedder.linear_1.bias'.format(key_prefix) in state_dict_keys:  # Omnigen2
        dit_config = {}
        dit_config["image_model"] = "omnigen2"
        dit_config["axes_dim_rope"] = [40, 40, 40]
        dit_config["axes_lens"] = [1024, 1664, 1664]
        dit_config["ffn_dim_multiplier"] = None
        dit_config["hidden_size"] = 2520
        dit_config["in_channels"] = 16
        dit_config["multiple_of"] = 256
        dit_config["norm_eps"] = 1e-05
        dit_config["num_attention_heads"] = 21
        dit_config["num_kv_heads"] = 7
        dit_config["num_layers"] = 32
        dit_config["num_refiner_layers"] = 2
        dit_config["out_channels"] = None
        dit_config["patch_size"] = 2
        dit_config["text_feat_dim"] = 2048
        dit_config["timestep_scale"] = 1000.0
        return dit_config

    if '{}input_blocks.0.0.weight'.format(key_prefix) not in state_dict_keys:
        return None

    unet_config = {
        "use_checkpoint": False,
        "image_size": 32,
        "use_spatial_transformer": True,
        "legacy": False
    }

    y_input = '{}label_emb.0.0.weight'.format(key_prefix)
    if y_input in state_dict_keys:
        unet_config["num_classes"] = "sequential"
        unet_config["adm_in_channels"] = state_dict[y_input].shape[1]
    else:
        unet_config["adm_in_channels"] = None

    model_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[0]
    in_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[1]

    out_key = '{}out.2.weight'.format(key_prefix)
    if out_key in state_dict:
        out_channels = state_dict[out_key].shape[0]
    else:
        out_channels = 4

    num_res_blocks = []
    channel_mult = []
    transformer_depth = []
    transformer_depth_output = []
    context_dim = None
    use_linear_in_transformer = False

    video_model = False
    video_model_cross = False

    current_res = 1
    count = 0

    last_res_blocks = 0
    last_channel_mult = 0

    input_block_count = count_blocks(state_dict_keys, '{}input_blocks'.format(key_prefix) + '.{}.')
    for count in range(input_block_count):
        prefix = '{}input_blocks.{}.'.format(key_prefix, count)
        prefix_output = '{}output_blocks.{}.'.format(key_prefix, input_block_count - count - 1)

        block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))
        if len(block_keys) == 0:
            break

        block_keys_output = sorted(list(filter(lambda a: a.startswith(prefix_output), state_dict_keys)))

        if "{}0.op.weight".format(prefix) in block_keys:  # new layer
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)

            current_res *= 2
            last_res_blocks = 0
            last_channel_mult = 0
            out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
            if out is not None:
                transformer_depth_output.append(out[0])
            else:
                transformer_depth_output.append(0)
        else:
            res_block_prefix = "{}0.in_layers.0.weight".format(prefix)
            if res_block_prefix in block_keys:
                last_res_blocks += 1
                last_channel_mult = state_dict["{}0.out_layers.3.weight".format(prefix)].shape[0] // model_channels

                out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth.append(out[0])
                    if context_dim is None:
                        context_dim = out[1]
                        use_linear_in_transformer = out[2]
                        video_model = out[3]
                        video_model_cross = out[4]
                else:
                    transformer_depth.append(0)

            res_block_prefix = "{}0.in_layers.0.weight".format(prefix_output)
            if res_block_prefix in block_keys_output:
                out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth_output.append(out[0])
                else:
                    transformer_depth_output.append(0)

    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)
    if "{}middle_block.1.proj_in.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = count_blocks(state_dict_keys, '{}middle_block.1.transformer_blocks.'.format(key_prefix) + '{}')
    elif "{}middle_block.0.in_layers.0.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = -1
    else:
        transformer_depth_middle = -2

    unet_config["in_channels"] = in_channels
    unet_config["out_channels"] = out_channels
    unet_config["model_channels"] = model_channels
    unet_config["num_res_blocks"] = num_res_blocks
    unet_config["transformer_depth"] = transformer_depth
    unet_config["transformer_depth_output"] = transformer_depth_output
    unet_config["channel_mult"] = channel_mult
    unet_config["transformer_depth_middle"] = transformer_depth_middle
    unet_config['use_linear_in_transformer'] = use_linear_in_transformer
    unet_config["context_dim"] = context_dim

    if video_model:
        unet_config["extra_ff_mix_layer"] = True
        unet_config["use_spatial_context"] = True
        unet_config["merge_strategy"] = "learned_with_images"
        unet_config["merge_factor"] = 0.0
        unet_config["video_kernel_size"] = [3, 1, 1]
        unet_config["use_temporal_resblock"] = True
        unet_config["use_temporal_attention"] = True
        unet_config["disable_temporal_crossattention"] = not video_model_cross
    else:
        unet_config["use_temporal_resblock"] = False
        unet_config["use_temporal_attention"] = False

    return unet_config


def model_config_from_unet_config(unet_config, state_dict=None):
    for model_config in supported_models.models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)

    logging.error("no match {}".format(unet_config))
    return None


def model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False, metadata:Optional[dict]=None):
    unet_config = detect_unet_config(state_dict, unet_key_prefix, metadata=metadata)
    if unet_config is None:
        return None
    model_config = model_config_from_unet_config(unet_config, state_dict)
    if model_config is None and use_base_if_no_match:
        model_config = supported_models_base.BASE(unet_config)

    scaled_fp8_key = "{}scaled_fp8".format(unet_key_prefix)
    if scaled_fp8_key in state_dict:
        scaled_fp8_weight = state_dict.pop(scaled_fp8_key)
        model_config.scaled_fp8 = scaled_fp8_weight.dtype
        if model_config.scaled_fp8 == torch.float32:
            model_config.scaled_fp8 = torch.float8_e4m3fn
        if scaled_fp8_weight.nelement() == 2:
            model_config.optimizations["fp8"] = False
        else:
            model_config.optimizations["fp8"] = True

    if metadata is not None and "format" in metadata and metadata["format"] == "gguf":
        model_config.custom_operations = GGMLOps

    return model_config


def unet_prefix_from_state_dict(state_dict):
    candidates = ["model.diffusion_model.",  # ldm/sgm models
                  "model.model.",  # audio models
                  "net.",  # cosmos
                  ]
    counts = {k: 0 for k in candidates}
    for k in state_dict:
        for c in candidates:
            if k.startswith(c):
                counts[c] += 1
                break

    top = max(counts, key=counts.get)
    if counts[top] > 5:
        return top
    else:
        return "model."  # aura flow and others


def convert_config(unet_config):
    new_config = unet_config.copy()
    num_res_blocks = new_config.get("num_res_blocks", None)
    channel_mult = new_config.get("channel_mult", None)

    if isinstance(num_res_blocks, int):
        num_res_blocks = len(channel_mult) * [num_res_blocks]

    if "attention_resolutions" in new_config:
        attention_resolutions = new_config.pop("attention_resolutions")
        transformer_depth = new_config.get("transformer_depth", None)
        transformer_depth_middle = new_config.get("transformer_depth_middle", None)

        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        if transformer_depth_middle is None:
            transformer_depth_middle = transformer_depth[-1]
        t_in = []
        t_out = []
        s = 1
        for i in range(len(num_res_blocks)):
            res = num_res_blocks[i]
            d = 0
            if s in attention_resolutions:
                d = transformer_depth[i]

            t_in += [d] * res
            t_out += [d] * (res + 1)
            s *= 2
        transformer_depth = t_in
        new_config["transformer_depth"] = t_in
        new_config["transformer_depth_output"] = t_out
        new_config["transformer_depth_middle"] = transformer_depth_middle

    new_config["num_res_blocks"] = num_res_blocks
    return new_config


def unet_config_from_diffusers_unet(state_dict, dtype=None):
    if "conv_in.weight" not in state_dict:
        return None

    match = {}
    transformer_depth = []

    attn_res = 1
    down_blocks = count_blocks(state_dict, "down_blocks.{}")
    for i in range(down_blocks):
        attn_blocks = count_blocks(state_dict, "down_blocks.{}.attentions.".format(i) + '{}')
        res_blocks = count_blocks(state_dict, "down_blocks.{}.resnets.".format(i) + '{}')
        for ab in range(attn_blocks):
            transformer_count = count_blocks(state_dict, "down_blocks.{}.attentions.{}.transformer_blocks.".format(i, ab) + '{}')
            transformer_depth.append(transformer_count)
            if transformer_count > 0:
                match["context_dim"] = state_dict["down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_k.weight".format(i, ab)].shape[1]

        attn_res *= 2
        if attn_blocks == 0:
            for i in range(res_blocks):
                transformer_depth.append(0)

    match["transformer_depth"] = transformer_depth

    match["model_channels"] = state_dict["conv_in.weight"].shape[0]
    match["in_channels"] = state_dict["conv_in.weight"].shape[1]
    match["adm_in_channels"] = None
    if "class_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["class_embedding.linear_1.weight"].shape[1]
    elif "add_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["add_embedding.linear_1.weight"].shape[1]

    SDXL = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
            'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
            'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_refiner = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 2560, 'dtype': dtype, 'in_channels': 4, 'model_channels': 384,
                    'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [0, 0, 4, 4, 4, 4, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 4,
                    'use_linear_in_transformer': True, 'context_dim': 1280, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
                    'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD21 = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'adm_in_channels': None, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2],
            'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': True,
            'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD21_uncliph = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 2048, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                    'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1,
                    'use_linear_in_transformer': True, 'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD21_unclipl = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 1536, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                    'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1,
                    'use_linear_in_transformer': True, 'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD15 = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,
            'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
            'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, 'num_heads': 8,
            'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_mid_cnet = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                     'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                     'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 0, 0, 1, 1], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 1,
                     'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 0, 0, 0, 1, 1, 1],
                     'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_small_cnet = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                       'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                       'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 0, 0, 0, 0], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 0,
                       'use_linear_in_transformer': True, 'num_head_channels': 64, 'context_dim': 1, 'transformer_depth_output': [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_diffusers_inpaint = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 9, 'model_channels': 320,
                              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
                              'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
                              'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_diffusers_ip2p = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                           'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 8, 'model_channels': 320,
                           'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
                           'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
                           'use_temporal_attention': False, 'use_temporal_resblock': False}

    SSD_1B = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 4, 4], 'transformer_depth_output': [0, 0, 0, 1, 1, 2, 10, 4, 4],
              'channel_mult': [1, 2, 4], 'transformer_depth_middle': -1, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
              'use_temporal_attention': False, 'use_temporal_resblock': False}

    Segmind_Vega = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                    'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 1, 1, 2, 2], 'transformer_depth_output': [0, 0, 0, 1, 1, 1, 2, 2, 2],
                    'channel_mult': [1, 2, 4], 'transformer_depth_middle': -1, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
                    'use_temporal_attention': False, 'use_temporal_resblock': False}

    KOALA_700M = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                  'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                  'num_res_blocks': [1, 1, 1], 'transformer_depth': [0, 2, 5], 'transformer_depth_output': [0, 0, 2, 2, 5, 5],
                  'channel_mult': [1, 2, 4], 'transformer_depth_middle': -2, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
                  'use_temporal_attention': False, 'use_temporal_resblock': False}

    KOALA_1B = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                'num_res_blocks': [1, 1, 1], 'transformer_depth': [0, 2, 6], 'transformer_depth_output': [0, 0, 2, 2, 6, 6],
                'channel_mult': [1, 2, 4], 'transformer_depth_middle': 6, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
                'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD09_XS = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
               'adm_in_channels': None, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [1, 1, 1],
               'transformer_depth': [1, 1, 1], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': -2, 'use_linear_in_transformer': True,
               'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1],
               'use_temporal_attention': False, 'use_temporal_resblock': False, 'disable_self_attentions': [True, False, False]}

    SD_XS = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
             'adm_in_channels': None, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [1, 1, 1],
             'transformer_depth': [0, 1, 1], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': -2, 'use_linear_in_transformer': False,
             'context_dim': 768, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 1, 1, 1, 1],
             'use_temporal_attention': False, 'use_temporal_resblock': False}
    SD15_diffusers_inpaint = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,
                              'dtype': dtype, 'in_channels': 9, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
                              'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, 'num_heads': 8,
                              'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                              'use_temporal_attention': False, 'use_temporal_resblock': False}

    LotusD = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': 4,
                'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
                'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': True, 'context_dim': 1024, 'num_heads': 8,
                'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                'use_temporal_attention': False, 'use_temporal_resblock': False}

    supported_models = [LotusD, SDXL, SDXL_refiner, SD21, SD15, SD21_uncliph, SD21_unclipl, SDXL_mid_cnet, SDXL_small_cnet, SDXL_diffusers_inpaint, SSD_1B, Segmind_Vega, KOALA_700M, KOALA_1B, SD09_XS, SD_XS, SDXL_diffusers_ip2p, SD15_diffusers_inpaint]

    for unet_config in supported_models:
        matches = True
        for k in match:
            if match[k] != unet_config[k]:
                matches = False
                break
        if matches:
            return convert_config(unet_config)
    return None


def model_config_from_diffusers_unet(state_dict):
    unet_config = unet_config_from_diffusers_unet(state_dict)
    if unet_config is not None:
        return model_config_from_unet_config(unet_config)
    return None


def convert_diffusers_mmdit(state_dict, output_prefix=""):
    out_sd = {}

    if 'joint_transformer_blocks.0.attn.add_k_proj.weight' in state_dict:  # AuraFlow
        num_joint = count_blocks(state_dict, 'joint_transformer_blocks.{}.')
        num_single = count_blocks(state_dict, 'single_transformer_blocks.{}.')
        sd_map = utils.auraflow_to_diffusers({"n_double_layers": num_joint, "n_layers": num_joint + num_single}, output_prefix=output_prefix)
    elif 'adaln_single.emb.timestep_embedder.linear_1.bias' in state_dict and 'pos_embed.proj.bias' in state_dict:  # PixArt
        num_blocks = count_blocks(state_dict, 'transformer_blocks.{}.')
        sd_map = utils.pixart_to_diffusers({"depth": num_blocks}, output_prefix=output_prefix)
    elif 'x_embedder.weight' in state_dict:  # Flux
        depth = count_blocks(state_dict, 'transformer_blocks.{}.')
        depth_single_blocks = count_blocks(state_dict, 'single_transformer_blocks.{}.')
        hidden_size = state_dict["x_embedder.bias"].shape[0]
        sd_map = utils.flux_to_diffusers({"depth": depth, "depth_single_blocks": depth_single_blocks, "hidden_size": hidden_size}, output_prefix=output_prefix)
    elif 'transformer_blocks.0.attn.add_q_proj.weight' in state_dict:  # SD3
        num_blocks = count_blocks(state_dict, 'transformer_blocks.{}.')
        depth = state_dict["pos_embed.proj.weight"].shape[0] // 64
        sd_map = utils.mmdit_to_diffusers({"depth": depth, "num_blocks": num_blocks}, output_prefix=output_prefix)
    else:
        return None

    for k in sd_map:
        weight = state_dict.get(k, None)
        if weight is not None:
            t = sd_map[k]

            if not isinstance(t, str):
                if len(t) > 2:
                    fun = t[2]
                else:
                    fun = lambda a: a
                offset = t[1]
                if offset is not None:
                    old_weight = out_sd.get(t[0], None)
                    if old_weight is None:
                        old_weight = torch.empty_like(weight)
                    if old_weight.shape[offset[0]] < offset[1] + offset[2]:
                        exp = list(weight.shape)
                        exp[offset[0]] = offset[1] + offset[2]
                        new = torch.empty(exp, device=weight.device, dtype=weight.dtype)
                        new[:old_weight.shape[0]] = old_weight
                        old_weight = new

                    w = old_weight.narrow(offset[0], offset[1], offset[2])
                else:
                    old_weight = weight
                    w = weight
                w[:] = fun(weight)
                t = t[0]
                out_sd[t] = old_weight
            else:
                out_sd[t] = weight
            state_dict.pop(k)

    return out_sd
