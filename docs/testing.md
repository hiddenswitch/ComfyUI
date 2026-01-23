# Testing Workflows

This page covers automated testing of ComfyUI workflows using pytest. For general library usage, see [Embedded / Library Usage](embedded.md).

## ProcessPoolExecutor for Isolation

Use `ProcessPoolExecutor` to run each workflow in a subprocess. This ensures VRAM is fully released between runs and prevents state leakage between tests:

```python
from comfy.client.embedded_comfy_client import Comfy
from comfy.cli_args import default_configuration
from comfy.cli_args_types import PerformanceFeature
from comfy.distributed.process_pool_executor import ProcessPoolExecutor

config = default_configuration()
config.disable_all_custom_nodes = True
config.novram = True
config.use_sage_attention = True
config.fast = {PerformanceFeature.CublasOps}

with ProcessPoolExecutor(max_workers=1) as executor:
    async with Comfy(configuration=config, executor=executor) as client:
        outputs = await client.queue_prompt(prompt)
```

## Directory Structure

ComfyUI's test suite uses `importlib.resources` to discover workflow JSON files from a Python package:

```
tests/
└── inference/
    ├── __init__.py
    ├── test_workflows.py
    └── workflows/
        ├── __init__.py          # makes this a Python package
        ├── sd15-basic-0.json
        ├── flux-0.json
        └── my-custom-workflow-0.json
```

The `__init__.py` inside `workflows/` is required so that `importlib.resources` can discover the JSON files. Each JSON file is an API-format workflow exported from the ComfyUI web UI (Save -> API Format).

## Minimal pytest Example

```python
import importlib.resources
import json
import pytest
from comfy.client.embedded_comfy_client import Comfy
from comfy.api.components.schema.prompt import Prompt
from comfy.cli_args import default_configuration
from comfy.cli_args_types import PerformanceFeature
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from comfy.model_downloader import add_known_models
from comfy.model_downloader_types import HuggingFile
from . import workflows


def _discover_workflows():
    add_known_models("loras", HuggingFile(
        "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
        "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"
    ))
    return {
        f.name: f
        for f in importlib.resources.files(workflows).iterdir()
        if f.is_file() and f.name.endswith(".json")
    }


@pytest.fixture(scope="function")
async def client():
    config = default_configuration()
    config.disable_all_custom_nodes = True
    config.novram = True
    config.fast = {PerformanceFeature.CublasOps}
    with ProcessPoolExecutor(max_workers=1) as executor:
        async with Comfy(configuration=config, executor=executor) as c:
            yield c


@pytest.mark.asyncio
@pytest.mark.parametrize("name, workflow_file", _discover_workflows().items())
async def test_workflow(name: str, workflow_file, client: Comfy):
    workflow = json.loads(workflow_file.read_text(encoding="utf8"))
    prompt = Prompt.validate(workflow)
    outputs = await client.queue_prompt(prompt)
    assert len(outputs) > 0
```

## Saving Image Outputs

The `SaveImage` node writes images to disk and returns their paths. Extract the output path from the workflow results to load or verify the generated image:

```python
from pathlib import Path
from PIL import Image


@pytest.mark.asyncio
async def test_generates_valid_image(client: Comfy):
    workflow = json.loads(Path("workflows/sd15-basic-0.json").read_text())
    prompt = Prompt.validate(workflow)
    outputs = await client.queue_prompt(prompt)

    save_node_id = next(
        key for key, value in prompt.items() if value.get("class_type") == "SaveImage"
    )
    image_path = outputs[save_node_id]["images"][0]["abs_path"]

    img = Image.open(image_path)
    assert img.size == (512, 512)
    assert img.mode == "RGB"
```

## Snapshot Testing with pytest-image-diff

Use [pytest-image-diff](https://pypi.org/project/pytest-image-diff/) to compare generated images against reference snapshots. On first run, the reference image is saved. On subsequent runs, the test fails if the output differs beyond a threshold.

Install:

```shell
uv pip install pytest-image-diff
```

```python
from pathlib import Path
from PIL import Image


@pytest.mark.asyncio
async def test_image_matches_snapshot(client: Comfy, image_diff):
    workflow = json.loads(Path("workflows/sd15-basic-0.json").read_text())
    prompt = Prompt.validate(workflow)

    # Pin the seed for deterministic output
    sampler_node_id = next(
        key for key, value in prompt.items() if value.get("class_type") == "KSampler"
    )
    prompt[sampler_node_id]["inputs"]["seed"] = 42

    outputs = await client.queue_prompt(prompt)

    save_node_id = next(
        key for key, value in prompt.items() if value.get("class_type") == "SaveImage"
    )
    image_path = outputs[save_node_id]["images"][0]["abs_path"]
    result = Image.open(image_path)

    image_diff(result, threshold=0.001)
```

Reference images are stored in a `image_snapshots/` directory next to your test file. Run with `--image-diff-update` to regenerate snapshots:

```shell
pytest --image-diff-update tests/inference/test_workflows.py
```

Note: deterministic output requires pinning the seed and using `--deterministic` in your ComfyUI configuration. Even then, results may vary across GPU architectures.
