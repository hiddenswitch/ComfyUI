"""
Test to investigate why DownloadableFileList serializes to empty in node_info.

The issue: When the UI requests /object_info, the LTXAVTextEncoderLoader node's
options for text_encoder and ckpt_name inputs are empty, even though json.dumps
shows non-empty results.
"""
import json
from dataclasses import asdict, dataclass
from typing import Any

import pytest

from comfy.model_downloader_types import DownloadableFileList, HuggingFile


class TestDownloadableFileListSerialization:
    """Test DownloadableFileList serialization behavior."""

    def test_basic_list_behavior(self):
        """Verify DownloadableFileList behaves as a list."""
        existing = ["model1.safetensors", "model2.safetensors"]
        downloadable = [HuggingFile("org/repo", "model3.safetensors")]

        dfl = DownloadableFileList(existing, downloadable)

        # It should be a list
        assert isinstance(dfl, list)

        # It should contain existing files and downloadable files
        assert "model1.safetensors" in dfl
        assert "model2.safetensors" in dfl
        assert "model3.safetensors" in dfl

        # Length should be correct
        assert len(dfl) == 3

    def test_json_dumps_directly(self):
        """Verify json.dumps works on DownloadableFileList."""
        existing = ["model1.safetensors", "model2.safetensors"]
        downloadable = [HuggingFile("org/repo", "model3.safetensors")]

        dfl = DownloadableFileList(existing, downloadable)

        # json.dumps should work and return non-empty
        result = json.dumps(dfl)
        assert result != "[]"
        assert "model1.safetensors" in result

        # Parse it back
        parsed = json.loads(result)
        assert len(parsed) == 3

    def test_json_dumps_in_dict(self):
        """Verify json.dumps works when DownloadableFileList is in a dict."""
        existing = ["model1.safetensors", "model2.safetensors"]
        downloadable = [HuggingFile("org/repo", "model3.safetensors")]

        dfl = DownloadableFileList(existing, downloadable)

        # Put it in a dict like INPUT_TYPES does
        input_dict = {
            "required": {
                "model_name": (dfl,)
            }
        }

        result = json.dumps(input_dict)
        assert "model1.safetensors" in result

    def test_json_dumps_in_tuple(self):
        """Verify json.dumps works when DownloadableFileList is in a tuple."""
        existing = ["model1.safetensors"]
        dfl = DownloadableFileList(existing, [])

        # Put it in a tuple like V1 combo types
        combo_tuple = (dfl, {"tooltip": "Select a model"})

        result = json.dumps(combo_tuple)
        assert "model1.safetensors" in result

    def test_asdict_with_dataclass_containing_dfl(self):
        """Test asdict behavior with dataclass containing DownloadableFileList."""
        @dataclass
        class NodeInfo:
            input: dict[str, Any]

        existing = ["model1.safetensors"]
        dfl = DownloadableFileList(existing, [])

        info = NodeInfo(input={
            "required": {
                "model_name": (dfl, {"tooltip": "test"})
            }
        })

        # asdict should preserve the list contents
        result = asdict(info)

        # Check the structure is preserved
        assert "input" in result
        assert "required" in result["input"]
        assert "model_name" in result["input"]["required"]

        # The first element of the tuple should be the list
        model_name_tuple = result["input"]["required"]["model_name"]
        assert isinstance(model_name_tuple, tuple)
        options = model_name_tuple[0]

        # This is where the bug might be - check if options is non-empty
        print(f"options type: {type(options)}")
        print(f"options value: {options}")
        assert len(options) > 0, f"Options should not be empty, got: {options}"
        assert "model1.safetensors" in options

    def test_asdict_converts_list_subclass(self):
        """Test how asdict handles list subclasses."""
        @dataclass
        class Container:
            items: list[str]

        existing = ["item1", "item2"]
        dfl = DownloadableFileList(existing, [])

        container = Container(items=dfl)
        result = asdict(container)

        # asdict might convert to plain list
        print(f"result['items'] type: {type(result['items'])}")
        print(f"result['items'] value: {result['items']}")

        # The key question: does asdict preserve the list contents?
        assert len(result["items"]) == 2

    def test_nested_dict_with_tuple_containing_dfl(self):
        """Test the exact structure used by V1 INPUT_TYPES."""
        @dataclass
        class NodeInfoV1:
            input: dict[str, Any]
            input_order: dict[str, list[str]]
            output: list[str]
            name: str

        existing = ["model1.safetensors", "model2.safetensors"]
        dfl = DownloadableFileList(existing, [])

        # This mimics what add_to_dict_v1 does
        input_dict = {
            "required": {
                "model_name": ("COMBO", {"options": dfl})
            }
        }

        info = NodeInfoV1(
            input=input_dict,
            input_order={"required": ["model_name"]},
            output=["MODEL"],
            name="TestNode"
        )

        result = asdict(info)

        # Check the options survived
        options = result["input"]["required"]["model_name"][1]["options"]
        print(f"options after asdict: {options}")
        print(f"options type after asdict: {type(options)}")

        # Serialize to JSON (this is what web.json_response does)
        json_str = json.dumps(result)
        print(f"JSON result: {json_str}")

        # Parse and check
        parsed = json.loads(json_str)
        parsed_options = parsed["input"]["required"]["model_name"][1]["options"]
        print(f"parsed options: {parsed_options}")

        assert len(parsed_options) == 2, f"Expected 2 options, got {len(parsed_options)}: {parsed_options}"

    def test_v3_combo_input_as_dict(self):
        """Test the V3 Combo.Input.as_dict() method with DownloadableFileList."""
        from comfy_api.latest import io

        existing = ["model1.safetensors", "model2.safetensors"]
        dfl = DownloadableFileList(existing, [])

        combo_input = io.Combo.Input(
            id="model_name",
            options=dfl,
            tooltip="Select a model"
        )

        result = combo_input.as_dict()
        print(f"as_dict result: {result}")
        print(f"options in as_dict: {result.get('options')}")

        # The options should be present
        assert "options" in result
        assert len(result["options"]) == 2

    def test_simulate_node_info_flow(self):
        """Simulate the full node_info flow for a V3 node."""
        from comfy_api.latest import io
        from dataclasses import asdict

        existing = ["model1.safetensors", "model2.safetensors"]
        dfl = DownloadableFileList(existing, [])

        # Create a schema like LTXAVTextEncoderLoader does
        combo_input = io.Combo.Input(
            id="text_encoder",
            options=dfl,
        )

        # Get as_dict like add_to_dict_v1 does
        input_as_dict = combo_input.as_dict()
        io_type = combo_input.get_io_type()

        print(f"io_type: {io_type}")
        print(f"input_as_dict: {input_as_dict}")

        # Build the input dict structure
        input_dict = {
            "required": {
                "text_encoder": (io_type, input_as_dict)
            }
        }

        # Now simulate what GET_NODE_INFO_V1 does - it creates a NodeInfoV1 dataclass and calls asdict
        @dataclass
        class NodeInfoV1:
            input: dict
            name: str

        info = NodeInfoV1(input=input_dict, name="TestNode")
        result = asdict(info)

        print(f"asdict result: {result}")

        # Now JSON serialize like web.json_response does
        json_str = json.dumps(result)
        print(f"JSON string: {json_str}")

        # Parse and check
        parsed = json.loads(json_str)
        options = parsed["input"]["required"]["text_encoder"][1].get("options", [])
        print(f"Final options: {options}")

        assert len(options) == 2, f"Expected 2 options, got: {options}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
