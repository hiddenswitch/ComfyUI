from comfy.cmd import folder_paths
from comfy.nodes.base_nodes import LoadImage
from comfy.nodes.common import MAX_RESOLUTION


class WebcamCapture(LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("WEBCAM", {}),
                "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "capture_on_queue": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_capture"

    CATEGORY = "image"

    def load_capture(self, image, **kwargs):
        return super().load_image(folder_paths.get_annotated_filepath(image))


NODE_CLASS_MAPPINGS = {
    "WebcamCapture": WebcamCapture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WebcamCapture": "Webcam Capture",
}
