from .spiece_tokenizer import SPieceTokenizer
from .t5 import T5
from .. import sd1_clip
from ..component_model.files import get_path_as_dict


class UMT5XXlModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options=None, textmodel_json_config=None):
        if model_options is None:
            model_options = {}
        textmodel_json_config = get_path_as_dict(textmodel_json_config, "umt5_config_xxl.json", package=__package__)
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=T5, enable_attention_masks=True, zero_out_masked=True, model_options=model_options)


class UMT5XXlTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        tokenizer = tokenizer_data.get("spiece_model", None)
        super().__init__(tokenizer, pad_with_end=False, embedding_size=4096, embedding_key='umt5xxl', tokenizer_class=SPieceTokenizer, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=512, pad_token=0, tokenizer_data=tokenizer_data, embedding_directory=embedding_directory)

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}


class WanT5Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="umt5xxl", tokenizer=UMT5XXlTokenizer)


class WanT5Model(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options=None, **kwargs):
        if model_options is None:
            model_options = {}
        super().__init__(device=device, dtype=dtype, model_options=model_options, name="umt5xxl", clip_model=UMT5XXlModel, **kwargs)


def te(dtype_t5=None, t5xxl_scaled_fp8=None):
    class WanTEModel(WanT5Model):
        def __init__(self, device="cpu", dtype=None, model_options=None):
            if model_options is None:
                model_options = {}
            if t5xxl_scaled_fp8 is not None and "scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["scaled_fp8"] = t5xxl_scaled_fp8
            if dtype_t5 is not None:
                dtype = dtype_t5
            super().__init__(device=device, dtype=dtype, model_options=model_options)

    return WanTEModel
