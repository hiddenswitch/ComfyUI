from typing import Literal, Tuple

import torch

from comfy import utils
from comfy.component_model.tensor_types import ImageBatch, RGBImageBatch
from comfy.nodes.base_nodes import ImageScale
from comfy.nodes.package_typing import CustomNode
from ..constants.resolutions import RESOLUTION_MAP, RESOLUTION_NAMES, SD_RESOLUTIONS


def levels_adjustment(image: ImageBatch, black_level: float = 0.0, mid_level: float = 0.5, white_level: float = 1.0, clip: bool = True) -> ImageBatch:
    """
    Apply a levels adjustment to an sRGB image.

    Args:
    image (torch.Tensor): Input image tensor of shape (B, H, W, C) with values in range [0, 1]
    black_level (float): Black point (default: 0.0)
    mid_level (float): Midtone point (default: 0.5)
    white_level (float): White point (default: 1.0)
    clip (bool): Whether to clip the output values to [0, 1] range (default: True)

    Returns:
    torch.Tensor: Adjusted image tensor of shape (B, H, W, C)
    """
    # Ensure input is in correct shape and range
    assert image.dim() == 4 and image.shape[-1] == 3, "Input should be of shape (B, H, W, 3)"
    assert 0 <= black_level < mid_level < white_level <= 1, "Levels should be in ascending order in range [0, 1]"

    def srgb_to_linear(x):
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(x):
        return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)

    linear = srgb_to_linear(image)
    adjusted = (linear - black_level) / (white_level - black_level)
    power_factor = torch.log2(torch.tensor(0.5, device=image.device)) / torch.log2(torch.tensor(mid_level, device=image.device))
    # apply power function to avoid nans
    adjusted = torch.where(adjusted > 0, torch.pow(adjusted.clamp(min=1e-8), power_factor), adjusted)
    result = linear_to_srgb(adjusted)

    if clip:
        result = torch.clamp(result, 0.0, 1.0)

    return result


class ImageResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize_mode": (["cover", "contain", "auto"], {"default": "cover"}),
                "resolutions": (RESOLUTION_NAMES, {"default": RESOLUTION_NAMES[0]}),
                "interpolation": (ImageScale.upscale_methods, {"default": "lanczos"}),
            },
            "optional": {
                "aspect_ratio_tolerance": ("FLOAT", {"min": 0, "max": 1.0, "default": 0.05, "step": 0.001})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "image/transform"

    def resize_image(self, image: ImageBatch, resize_mode: Literal["cover", "contain", "auto"], resolutions: str, interpolation: str, aspect_ratio_tolerance=0.05) -> tuple[RGBImageBatch]:
        supported_resolutions = RESOLUTION_MAP.get(resolutions, SD_RESOLUTIONS)
        return self.resize_image_with_supported_resolutions(image, resize_mode, supported_resolutions, interpolation, aspect_ratio_tolerance=aspect_ratio_tolerance)

    def resize_image_with_supported_resolutions(self, image: ImageBatch, resize_mode: Literal["cover", "contain", "auto"], supported_resolutions: list[tuple[int, int]], interpolation: str, aspect_ratio_tolerance=0.05) -> tuple[RGBImageBatch]:
        resized_images = []
        for img in image:
            h, w = img.shape[:2]
            current_aspect_ratio = w / h
            aspect_ratio_diffs = [(abs(res[0] / res[1] - current_aspect_ratio), res) for res in supported_resolutions]
            min_diff = min(aspect_ratio_diffs, key=lambda x: x[0])[0]
            close_enough_resolutions = [res for diff, res in aspect_ratio_diffs if diff <= min_diff + aspect_ratio_tolerance]
            target_resolution = max(close_enough_resolutions, key=lambda res: res[0] * res[1])

            if resize_mode == "cover":
                scale = max(target_resolution[0] / w, target_resolution[1] / h)
                new_w, new_h = int(w * scale), int(h * scale)
            elif resize_mode == "contain":
                scale = min(target_resolution[0] / w, target_resolution[1] / h)
                new_w, new_h = int(w * scale), int(h * scale)
            else:  # auto
                if current_aspect_ratio > target_resolution[0] / target_resolution[1]:
                    new_w, new_h = target_resolution[0], int(h * target_resolution[0] / w)
                else:
                    new_w, new_h = int(w * target_resolution[1] / h), target_resolution[1]

            img_tensor = img.permute(2, 0, 1).unsqueeze(0)
            resized = utils.common_upscale(img_tensor, new_w, new_h, interpolation, "disabled")

            if resize_mode == "contain":
                canvas = torch.zeros((1, 3, target_resolution[1], target_resolution[0]), device=resized.device, dtype=resized.dtype)
                y1, x1 = (target_resolution[1] - new_h) // 2, (target_resolution[0] - new_w) // 2
                canvas[:, :, y1:y1 + new_h, x1:x1 + new_w] = resized
                resized = canvas
            elif resize_mode == "cover":
                y1, x1 = (new_h - target_resolution[1]) // 2, (new_w - target_resolution[0]) // 2
                resized = resized[:, :, y1:y1 + target_resolution[1], x1:x1 + target_resolution[0]]
            else:  # auto
                if new_w != target_resolution[0] or new_h != target_resolution[1]:
                    canvas = torch.zeros((1, 3, target_resolution[1], target_resolution[0]), device=resized.device, dtype=resized.dtype)
                    y1, x1 = (target_resolution[1] - new_h) // 2, (target_resolution[0] - new_w) // 2
                    canvas[:, :, y1:y1 + new_h, x1:x1 + new_w] = resized
                    resized = canvas

            resized_images.append(resized.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0))
        return (torch.stack(resized_images),)


class ImageResize1(ImageResize):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize_mode": (["cover", "contain", "auto"], {"default": "cover"}),
                "width": ("INT", {"min": 1}),
                "height": ("INT", {"min": 1}),
                "interpolation": (ImageScale.upscale_methods, {"default": "lanczos"}),
            }
        }

    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)

    def execute(self, image: RGBImageBatch, resize_mode: Literal["cover", "contain", "auto"], width: int, height: int, interpolation: str) -> tuple[RGBImageBatch]:
        return self.resize_image_with_supported_resolutions(image, resize_mode, [(width, height)], interpolation)


class ImageLevels(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "black_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mid_level": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "white_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clip": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_levels"
    CATEGORY = "image/adjust"

    def apply_levels(self, image: ImageBatch, black_level: float, mid_level: float, white_level: float, clip: bool) -> Tuple[ImageBatch]:
        adjusted_image = levels_adjustment(image, black_level, mid_level, white_level, clip)
        return (adjusted_image,)


class ImageLuminance(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute_luminance"
    CATEGORY = "image/color"

    def compute_luminance(self, image: ImageBatch) -> Tuple[ImageBatch]:
        assert image.dim() == 4 and image.shape[-1] == 3, "Input should be of shape (B, H, W, 3)"
        coeffs = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device, dtype=image.dtype)
        luminance = torch.sum(image * coeffs, dim=-1, keepdim=True)
        luminance = luminance.expand(-1, -1, -1, 3)
        return (luminance,)


class ImageShape:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_shape"
    CATEGORY = "image/operations"

    def get_shape(self, image: ImageBatch):
        shape = image.shape
        return shape[2], shape[1]


NODE_CLASS_MAPPINGS = {
    "ImageResize": ImageResize,
    "ImageResize1": ImageResize1,
    "ImageLevels": ImageLevels,
    "ImageLuminance": ImageLuminance,
    "ImageShape": ImageShape,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResize": "Fit Image to Diffusion Size",
    "ImageResize1": "Fit Image to Width Height",
    "ImageShape": "Get Image Shape",
}
