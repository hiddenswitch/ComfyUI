import json
import os
from io import BytesIO
from itertools import chain
from typing import Tuple, Dict, Any, Literal

import requests
import torch
from PIL import Image

from comfy.component_model.tensor_types import RGBImageBatch, MaskBatch, ImageBatch
from comfy.nodes.package_typing import CustomNode, Seed
from comfy.utils import pil2tensor, tensor2pil
from comfy_extras.constants.resolutions import IDEOGRAM_RESOLUTIONS
from comfy_extras.nodes.nodes_mask import MaskToImage

ASPECT_RATIOS = [(10, 6), (16, 10), (9, 16), (3, 2), (4, 3)]
ASPECT_RATIO_ENUM = ["ASPECT_1_1"] + list(chain.from_iterable(
    [f"ASPECT_{a}_{b}", f"ASPECT_{b}_{a}"]
    for a, b in ASPECT_RATIOS
))
V2_MODELS = ["V_2", "V_2_TURBO"]
MODELS_ENUM = V2_MODELS + ["V_3"]
AUTO_PROMPT_ENUM = ["AUTO", "ON", "OFF"]
STYLES_ENUM = ["AUTO", "GENERAL", "REALISTIC", "DESIGN", "RENDER_3D", "ANIME"]
RESOLUTION_ENUM = [f"RESOLUTION_{w}_{h}" for w, h in IDEOGRAM_RESOLUTIONS]

def to_v3_resolution(resolution:str) -> str:
    return resolution[len("RESOLUTION_"):].replace("_", "x")

def api_key_in_env_or_workflow(api_key_from_workflow: str):
    from comfy.cli_args import args
    if api_key_from_workflow is not None and "" != api_key_from_workflow.strip():
        return api_key_from_workflow

    return os.environ.get("IDEOGRAM_API_KEY", args.ideogram_api_key)


class IdeogramGenerate(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "resolution": (RESOLUTION_ENUM, {"default": RESOLUTION_ENUM[0]}),
                "model": (MODELS_ENUM, {"default": MODELS_ENUM[0]}),
                "magic_prompt_option": (AUTO_PROMPT_ENUM, {"default": AUTO_PROMPT_ENUM[0]}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": Seed,
                "style_type": (STYLES_ENUM, {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "ideogram"

    def generate(self, prompt: str, resolution: str, model: str, magic_prompt_option: str,
                 api_key: str = "", negative_prompt: str = "", num_images: int = 1, seed: int = 0, style_type: str = "AUTO") -> Tuple[torch.Tensor]:
        api_key = api_key_in_env_or_workflow(api_key)
        headers = {"Api-Key": api_key, "Content-Type": "application/json"}

        if model in V2_MODELS:
            payload = {
                "image_request": {
                    "prompt": prompt,
                    "resolution": resolution,
                    "model": model,
                    "magic_prompt_option": magic_prompt_option,
                    "num_images": num_images,
                    "style_type": style_type,
                }
            }

            if negative_prompt:
                payload["image_request"]["negative_prompt"] = negative_prompt
            if seed:
                payload["image_request"]["seed"] = seed

            response = requests.post("https://api.ideogram.ai/generate", headers=headers, json=payload)
        elif model == "V_3":
            payload = {
                "prompt": prompt,
                "resolution": to_v3_resolution(resolution),
                "model": model,
                "magic_prompt": magic_prompt_option,
                "num_images": num_images,
                "style_type": style_type,
            }

            if negative_prompt:
                payload["negative_prompt"] = negative_prompt
            if seed:
                payload["seed"] = seed
            response = requests.post("https://api.ideogram.ai/v1/ideogram-v3/generate", headers=headers, json=payload)
        else:
            raise ValueError(f"Invalid model={model}")

        response.raise_for_status()

        images = []
        for item in response.json()["data"]:
            img_response = requests.get(item["url"])
            img_response.raise_for_status()

            pil_image = Image.open(BytesIO(img_response.content))
            images.append(pil2tensor(pil_image))

        return (torch.cat(images, dim=0),)


class IdeogramEdit(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (MODELS_ENUM, {"default": MODELS_ENUM[0]}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "magic_prompt_option": (AUTO_PROMPT_ENUM, {"default": AUTO_PROMPT_ENUM[0]}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0}),
                "style_type": (STYLES_ENUM, {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit"
    CATEGORY = "ideogram"

    def edit(self, images: RGBImageBatch, masks: MaskBatch, prompt: str, model: str,
             api_key: str = "", magic_prompt_option: str = "AUTO",
             num_images: int = 1, seed: int = 0, style_type: str = "AUTO") -> Tuple[torch.Tensor]:
        api_key = api_key_in_env_or_workflow(api_key)
        headers = {"Api-Key": api_key}
        image_responses = []
        for mask, image in zip(torch.unbind(masks), torch.unbind(images)):
            mask, = MaskToImage().mask_to_image(mask=mask)
            mask: RGBImageBatch

            image_pil = tensor2pil(image)
            mask_pil = tensor2pil(mask)

            image_bytes = BytesIO()
            mask_bytes = BytesIO()
            image_pil.save(image_bytes, format="PNG")
            mask_pil.save(mask_bytes, format="PNG")

            if model in V2_MODELS:
                files = {
                    "image_file": ("image.png", image_bytes.getvalue()),
                    "mask": ("mask.png", mask_bytes.getvalue()),
                }

                data = {
                    "prompt": prompt,
                    "model": model,
                    "magic_prompt_option": magic_prompt_option,
                    "num_images": num_images,
                    "style_type": style_type,
                }
                if seed:
                    data["seed"] = seed

                response = requests.post("https://api.ideogram.ai/edit", headers=headers, files=files, data=data)
            elif model == "V_3":
                files = {
                    "image": ("image.png", image_bytes.getvalue()),
                    "mask": ("mask.png", mask_bytes.getvalue()),
                }

                data = {
                    "prompt": prompt,
                    "magic_prompt": magic_prompt_option,
                    "num_images": num_images,
                }
                if seed:
                    data["seed"] = seed

                response = requests.post("https://api.ideogram.ai/v1/ideogram-v3/edit", headers=headers, files=files, data=data)
            else:
                raise ValueError(f"Invalid model={model}")

            response.raise_for_status()

            for item in response.json()["data"]:
                img_response = requests.get(item["url"])
                img_response.raise_for_status()

                pil_image = Image.open(BytesIO(img_response.content))
                image_responses.append(pil2tensor(pil_image))

        return (torch.cat(image_responses, dim=0),)


class IdeogramRemix(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "resolution": (RESOLUTION_ENUM, {"default": RESOLUTION_ENUM[0]}),
                "model": (MODELS_ENUM, {"default": MODELS_ENUM[0]}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "image_weight": ("INT", {"default": 50, "min": 1, "max": 100}),
                "magic_prompt_option": (AUTO_PROMPT_ENUM, {"default": AUTO_PROMPT_ENUM[0]}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0}),
                "style_type": (STYLES_ENUM, {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remix"
    CATEGORY = "ideogram"

    def remix(self, images: torch.Tensor, prompt: str, resolution: str, model: str,
              api_key: str = "", image_weight: int = 50, magic_prompt_option: str = "AUTO",
              negative_prompt: str = "", num_images: int = 1, seed: int = 0, style_type: str = "AUTO") -> Tuple[torch.Tensor]:
        api_key = api_key_in_env_or_workflow(api_key)
        headers = {"Api-Key": api_key}

        result_images = []
        for image in images:
            image_pil = tensor2pil(image)
            image_bytes = BytesIO()
            image_pil.save(image_bytes, format="PNG")

            if model in V2_MODELS:

                files = {
                    "image_file": ("image.png", image_bytes.getvalue()),
                }

                data = {
                    "prompt": prompt,
                    "resolution": resolution,
                    "model": model,
                    "image_weight": image_weight,
                    "magic_prompt_option": magic_prompt_option,
                    "num_images": num_images,
                    "style_type": style_type,
                }

                if negative_prompt:
                    data["negative_prompt"] = negative_prompt
                if seed:
                    data["seed"] = seed

                # data = {"image_request": data}

                response = requests.post("https://api.ideogram.ai/remix", headers=headers, files=files, data={
                    "image_request": json.dumps(data)
                })
            elif model == "V_3":
                files = {
                    "image": ("image.png", image_bytes.getvalue()),
                }

                data = {
                    "prompt": prompt,
                    "resolution": to_v3_resolution(resolution),
                    "image_weight": image_weight,
                    "magic_prompt": magic_prompt_option,
                    "num_images": num_images,
                    "style_type": style_type,
                }

                if negative_prompt:
                    data["negative_prompt"] = negative_prompt
                if seed:
                    data["seed"] = seed

                response = requests.post("https://api.ideogram.ai/v1/ideogram-v3/remix", headers=headers, files=files, data=data)
            else:
                raise ValueError(f"Invalid model={model}")

            response.raise_for_status()

            for item in response.json()["data"]:
                img_response = requests.get(item["url"])
                img_response.raise_for_status()

                pil_image = Image.open(BytesIO(img_response.content))
                result_images.append(pil2tensor(pil_image))

        return (torch.cat(result_images, dim=0),)


class IdeogramDescribe(CustomNode):
    """
    A ComfyUI node to get a description of an image using the Ideogram API.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "describe"
    CATEGORY = "ideogram"

    def describe(self, images: ImageBatch, api_key: str = "") -> tuple[list[str]]:
        """
        Sends an image to the Ideogram API and returns a generated description.

        Args:
            images: A batch of images as a tensor.
            api_key: The Ideogram API key.

        Returns:
            A tuple containing the description string for the first image.
        """
        api_key = api_key_in_env_or_workflow(api_key)
        headers = {"Api-Key": api_key}

        descriptions_batch = []
        for image in images:
            pil_image = tensor2pil(image)

            image_bytes = BytesIO()
            pil_image.save(image_bytes, format="PNG")
            image_bytes.seek(0)

            files = {
                "image_file": ("image.png", image_bytes.getvalue(), "image/png"),
            }

            response = requests.post("https://api.ideogram.ai/describe", headers=headers, files=files)
            response.raise_for_status()

            data = response.json()
            descriptions = data.get("descriptions", [])

            if not descriptions:
                descriptions_batch.append("")
            else:
                first_description = descriptions[0].get("text", "")
                descriptions_batch.append(first_description)

        return (descriptions_batch,)


NODE_CLASS_MAPPINGS = {
    "IdeogramGenerate": IdeogramGenerate,
    "IdeogramEdit": IdeogramEdit,
    "IdeogramRemix": IdeogramRemix,
    "IdeogramDescribe": IdeogramDescribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ideogram Generate": "Ideogram Generate",
    "Ideogram Edit": "Ideogram Edit",
    "Ideogram Remix": "Ideogram Remix",
    "Ideogram Describe": "Ideogram Describe",
}
