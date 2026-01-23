# Using ComfyUI as a Library

ComfyUI can be used as an embedded library inside your own Python application. No server process is started — it runs the workflow engine directly in your process.

## Installing

```shell
uv pip install --torch-backend=auto "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
```

## Running a Workflow

Save a workflow from the ComfyUI web UI in API format (a JSON file). This JSON is a valid Python `dict[str, Any]` literal — paste it directly into your code:

```python
from comfy.client.embedded_comfy_client import Comfy
import copy

WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
    },
    "2": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "bad hands", "clip": ["1", 1]},
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "masterpiece best quality girl", "clip": ["1", 1]},
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 8566257,
            "steps": 20,
            "cfg": 8,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1,
            "model": ["1", 0],
            "positive": ["4", 0],
            "negative": ["3", 0],
            "latent_image": ["2", 0],
        },
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI_API", "images": ["6", 0]},
    },
}


async def run_example():
    prompt = copy.deepcopy(WORKFLOW)
    prompt["4"]["inputs"]["text"] = "masterpiece best quality man"
    prompt["5"]["inputs"]["seed"] = 5

    # Runs ComfyUI as a library. No server, no API call.
    async with Comfy() as client:
        outputs = await client.queue_prompt(prompt)

        # Find the SaveImage node and get the output path
        save_image_node_id = next(
            key for key in prompt if prompt[key]["class_type"] == "SaveImage"
        )
        return outputs[save_image_node_id]["images"][0]["abs_path"]

    # All models are unloaded and VRAM is released when the block exits.
```

Run it with:

```python
import asyncio

path = asyncio.run(run_example())
print(path)
```

Models referenced in the workflow are downloaded automatically.

## Building Workflows Programmatically

Use `GraphBuilder` to construct workflows in code instead of pasting JSON:

```python
from comfy_execution.graph_utils import GraphBuilder


def build_graph(positive_prompt_text="masterpiece best quality girl"):
    builder = GraphBuilder()

    checkpoint_loader = builder.node(
        "CheckpointLoaderSimple",
        ckpt_name="v1-5-pruned-emaonly.safetensors",
    )

    empty_latent = builder.node(
        "EmptyLatentImage",
        width=512,
        height=512,
        batch_size=1,
    )

    negative_prompt = builder.node(
        "CLIPTextEncode",
        text="bad hands",
        clip=checkpoint_loader.out(1),
    )

    positive_prompt = builder.node(
        "CLIPTextEncode",
        text=positive_prompt_text,
        clip=checkpoint_loader.out(1),
    )

    k_sampler = builder.node(
        "KSampler",
        seed=8566257,
        steps=20,
        cfg=8,
        sampler_name="euler",
        scheduler="normal",
        denoise=1,
        model=checkpoint_loader.out(0),
        positive=positive_prompt.out(0),
        negative=negative_prompt.out(0),
        latent_image=empty_latent.out(0),
    )

    vae_decode = builder.node(
        "VAEDecode",
        samples=k_sampler.out(0),
        vae=checkpoint_loader.out(2),
    )

    builder.node(
        "SaveImage",
        filename_prefix="ComfyUI_API",
        images=vae_decode.out(0),
    )

    return builder


builder = build_graph()
prompt = builder.finalize()  # Returns the same dict format as the JSON workflow
```

The `finalize()` output is identical to the API format JSON — pass it to `client.queue_prompt(prompt)`.

## Streaming Progress and Previews

Use `queue_with_progress` to receive preview images during inference:

```python
import copy
from comfy.client.embedded_comfy_client import Comfy
from comfy.api.components.schema.prompt import Prompt
from comfy.component_model.queue_types import BinaryEventTypes


async def run_with_previews():
    prompt = Prompt.validate(copy.deepcopy(WORKFLOW))

    async with Comfy() as client:
        task = client.queue_with_progress(prompt)

        async for notification in task.progress():
            if notification.event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
                image_data, metadata = notification.data
                # image_data.pil_image is a PIL Image of the current denoising step
                print(f"Preview: {image_data.pil_image.size}")

        # Get the final outputs after all progress events
        result = await task.get()
        save_image_node_id = next(
            key for key, value in prompt.items() if value.get("class_type") == "SaveImage"
        )
        return result.outputs[save_image_node_id]["images"][0]["abs_path"]
```

## Running the Web UI Programmatically

You can also start the full ComfyUI web server from Python:

```python
from comfy.cmd.main import entrypoint

entrypoint()
```

Or in Google Colab with a tunnel:

```python
from comfy.app.colab import start_server_in_colab

start_server_in_colab()
```

## Configuring Performance Options

Use `default_configuration()` to create a `Configuration` object, then set attributes or call `.update()` to configure performance options before passing it to `Comfy()`:

```python
from comfy.client.embedded_comfy_client import Comfy
from comfy.cli_args import default_configuration
from comfy.cli_args_types import PerformanceFeature

config = default_configuration()

# Enable SageAttention (requires sageattention package installed)
config.use_sage_attention = True

# Enable cuBLAS ops for faster matrix multiplications (NVIDIA Ampere+ GPUs)
config.fast = {PerformanceFeature.CublasOps}

# Minimize VRAM usage by aggressively offloading models to CPU
config.novram = True

# Run VAE in full precision (recommended for AMD GPUs)
config.fp32_vae = True

# Disable custom nodes for faster startup and isolation
config.disable_all_custom_nodes = True

# Multiple options can be set at once with .update()
config.update({
    "use_sage_attention": True,
    "novram": True,
    "fast": {PerformanceFeature.CublasOps},
})

async with Comfy(configuration=config) as client:
    outputs = await client.queue_prompt(prompt)
```

Available `PerformanceFeature` values for `config.fast`:

- `PerformanceFeature.CublasOps` — Use cuBLAS for supported operations. Recommended for NVIDIA Ampere (RTX 30xx) and newer GPUs.
- `PerformanceFeature.Fp16Accumulation` — Use FP16 accumulation. May reduce quality.
- `PerformanceFeature.Fp8MatrixMultiplication` — Use FP8 matrix multiplication.
- `PerformanceFeature.AutoTune` — Enable PyTorch autotuning.

### Using ProcessPoolExecutor for Isolation

When running workflows in production or automated testing, use `ProcessPoolExecutor` to run each workflow in a subprocess. This ensures VRAM is fully released between runs and configuration options like `novram` are applied correctly:

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

## Running Multiple Workflows

To run a list of workflows programmatically, iterate over them with the same `Comfy` client:

```python
import json
from pathlib import Path
from comfy.client.embedded_comfy_client import Comfy
from comfy.api.components.schema.prompt import Prompt
from comfy.cli_args import default_configuration

config = default_configuration()
config.novram = True

workflow_dir = Path("./my_workflows")

async with Comfy(configuration=config) as client:
    for workflow_path in sorted(workflow_dir.glob("*.json")):
        workflow = json.loads(workflow_path.read_text())
        prompt = Prompt.validate(workflow)
        outputs = await client.queue_prompt(prompt)
        print(f"{workflow_path.name}: {outputs}")
```

## Headless Workflow Execution with `--workflows`

The `--workflows` CLI flag runs workflows without starting the web server. Outputs are printed as JSON to stdout, and application logging goes to stderr.

**Run a single workflow file:**

```bash
uv run --no-sync comfyui --workflows my_workflow.json
```

**Run multiple workflow files:**

```bash
uv run --no-sync comfyui --workflows workflow1.json workflow2.json
```

**Read workflows from stdin (use `-`):**

```bash
cat my_workflow.json | uv run --no-sync comfyui --workflows -
```

**Pipe a literal JSON workflow:**

```bash
echo '{"1":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"v1-5-pruned-emaonly.safetensors"}}}' | uv run --no-sync comfyui --workflows -
```

The input stream supports concatenated JSON objects — multiple `{}{}{}` objects in sequence will each be executed as a separate workflow. Each workflow's outputs are printed as a single JSON line to stdout.

Combine with other flags for performance tuning:

```bash
uv run --no-sync comfyui --novram --use-sage-attention --fast cublas_ops --workflows my_workflow.json
```

## Structuring Automated Test Directories

ComfyUI's test suite uses `importlib.resources` to discover workflow JSON files from a Python package. Structure your test directory like this:

```
tests/
└── inference/
    ├── __init__.py
    ├── test_workflows.py
    └── workflows/
        ├── __init__.py          # Makes this a Python package
        ├── sd15-basic-0.json
        ├── flux-0.json
        └── my-custom-workflow-0.json
```

The `__init__.py` inside `workflows/` is required so that `importlib.resources` can discover the JSON files. Each JSON file is an API-format workflow exported from the ComfyUI web UI (Save → API Format).

A minimal test file:

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
from . import workflows  # the workflows/ package


def _discover_workflows():
    """Register any models needed by test workflows, then discover JSON files."""
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

## Adding Known Models for Automatic Download

Use `add_known_models()` to register models that should be downloaded automatically from Hugging Face when a workflow references them. This makes models appear in the UI dropdown and triggers on-demand downloads.

```python
from comfy.model_downloader import add_known_models
from comfy.model_downloader_types import HuggingFile, CivitFile

# Register a model from Hugging Face
add_known_models("checkpoints", HuggingFile(
    "stabilityai/stable-diffusion-xl-base-1.0",
    "sd_xl_base_1.0.safetensors"
))

# Register a LoRA
add_known_models("loras", HuggingFile(
    "ByteDance/Hyper-SD",
    "Hyper-SDXL-12steps-CFG-lora.safetensors"
))

# Register a model from CivitAI (by model ID and version ID)
add_known_models("checkpoints", CivitFile(
    model_id=133005,
    model_version_id=357609,
    filename="juggernautXL_v9Rundiffusionphoto2.safetensors"
))

# Register with a custom save filename (when the repo filename is generic)
add_known_models("controlnet", HuggingFile(
    "jschoormans/controlnet-densepose-sdxl",
    "diffusion_pytorch_model.safetensors",
    save_with_filename="controlnet-densepose-sdxl.safetensors"
))

# Register multiple models at once
add_known_models("diffusion_models",
    HuggingFile("black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
)
```

The first argument is the folder name (matching ComfyUI's model directory structure: `checkpoints`, `loras`, `controlnet`, `vae`, `clip`, `diffusion_models`, `upscale_models`, etc.).

### How Model Downloads Work

When a workflow references a model filename, ComfyUI checks the known models registry. If the file isn't found locally, it downloads from Hugging Face using `hf_hub_download`. Files are stored in the **Hugging Face cache** (`~/.cache/huggingface/hub/` by default) and symlinked into the appropriate model directory.

To use traditional flat file downloads instead of the Hugging Face cache, pass `--force-hf-local-dir-mode` at startup. This saves files directly to `models/huggingface/<repo_id>/`.

### Authenticating for Gated Repositories

Some models (like `black-forest-labs/FLUX.1-dev` or `stabilityai/stable-diffusion-3-medium`) require accepting terms on Hugging Face before downloading. If you try to download a gated model without authentication, ComfyUI raises a `GatedRepoError` with instructions.

To authenticate:

1. Visit the model's Hugging Face page and accept the terms.
2. Set your token using one of:

```bash
# Option 1: Environment variable
export HF_TOKEN=hf_your_token_here

# Option 2: Login via CLI (stores token persistently)
huggingface-cli login
```

ComfyUI passes `token=True` to `hf_hub_download`, which automatically uses the `HF_TOKEN` environment variable or the token stored by `huggingface-cli login`.

### Disabling Automatic Downloads

To prevent automatic model downloads (e.g., in air-gapped environments):

```python
config = default_configuration()
config.disable_known_models = True
```

Or via CLI:

```bash
uv run --no-sync comfyui --disable-known-models
```

## When to Use `--novram`

The `--novram` flag (or `config.novram = True`) aggressively offloads all model weights to CPU RAM between operations, minimizing GPU VRAM usage at the cost of speed. Use it when:

- **Your GPU has 16 GB of VRAM or less** and you're running large models (FLUX, SD3.5, video models like Wan 2.1 or HunyuanVideo).
- **Running automated tests or CI** where reliability matters more than speed. The test suite defaults to `novram=True` to avoid OOM crashes.
- **Running multiple workflows in sequence** where different models need to load/unload cleanly.
- **Your system has limited swap or RAM** and you want to prevent the OS from thrashing.

```python
config = default_configuration()
config.novram = True
```

Or via CLI:

```bash
uv run --no-sync comfyui --novram
```

Without `--novram`, ComfyUI uses smart memory management to keep recently-used models in VRAM for faster subsequent runs. This is better for interactive use but can cause OOM errors with large models on limited hardware.
