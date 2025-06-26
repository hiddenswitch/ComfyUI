ComfyUI LTS
=======

A vanilla, up-to-date fork of [ComfyUI](https://github.com/comfyanonymous/comfyui) intended for long term support (LTS) from [AppMana](https://appmana.com) and [Hidden Switch](https://hiddenswitch.com).

### New Features

- To run, just type `comfyui` in your command line and press enter.
- [Installable](#installing) via `uv`: `UV_TORCH_BACKEND=auto uv pip install comfyui@git+https://github.com/hiddenswitch/ComfyUI.git`.
- [Large Language Models](#large-language-models) with multi-modal support included.
- [Automatic model downloading](#model-downloading) to save you disk space and time.
- [Distributed](#distributed-multi-process-and-multi-gpu-comfy) with support for multiple GPUs, multiple backends and frontends, including in containers, using RabbitMQ.
- [Installable custom nodes](#custom-nodes) via `pip`, including LTS support for forked nodes from the community.
- [New configuration options](#command-line-arguments) for directories, models and metrics.
- [API](#using-comfyui-as-an-api--programmatically) support, using the vanilla ComfyUI API and new API endpoints.
- [Embed](#embedded) ComfyUI as a library inside your Python application. No server or frontend needed.
- [Containers](#containers) for running on Linux, Windows and Kubernetes with CUDA acceleration.
- Automated tests for new features.

![ComfyUI Screenshot](https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe)

ComfyUI lets you design and execute advanced stable diffusion pipelines using a graph/nodes/flowchart based interface. Available on Windows, Linux, and macOS.

## Get Started

#### [Desktop Application](https://www.comfy.org/download)

- The easiest way to get started.
- Available on Windows & macOS.

#### [Windows Portable Package](#installing)

- Get the latest commits and completely portable.
- Available on Windows.

#### [Manual Install](#manual-install-windows-linux)

Supports all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend).

## [Examples](https://comfyanonymous.github.io/ComfyUI_examples/)

See what ComfyUI can do with the [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).

## Upstream Features
- Nodes/graph/flowchart interface to experiment and create complex Stable Diffusion workflows without needing to code anything.
- Image Models
    - SD1.x, SD2.x,
    - [SDXL](https://comfyanonymous.github.io/ComfyUI_examples/sdxl/), [SDXL Turbo](https://comfyanonymous.github.io/ComfyUI_examples/sdturbo/)
    - [Stable Cascade](https://comfyanonymous.github.io/ComfyUI_examples/stable_cascade/)
    - [SD3 and SD3.5](https://comfyanonymous.github.io/ComfyUI_examples/sd3/)
    - Pixart Alpha and Sigma
    - [AuraFlow](https://comfyanonymous.github.io/ComfyUI_examples/aura_flow/)
    - [HunyuanDiT](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_dit/)
    - [Flux](https://comfyanonymous.github.io/ComfyUI_examples/flux/)
    - [Lumina Image 2.0](https://comfyanonymous.github.io/ComfyUI_examples/lumina2/)
    - [HiDream](https://comfyanonymous.github.io/ComfyUI_examples/hidream/)
   - [Cosmos Predict2](https://comfyanonymous.github.io/ComfyUI_examples/cosmos_predict2/)
- Image Editing Models
   - [Omnigen 2](https://comfyanonymous.github.io/ComfyUI_examples/omnigen/)
   - [Flux Kontext](https://comfyanonymous.github.io/ComfyUI_examples/flux/#flux-kontext-image-editing-model)
- Video Models
    - [Stable Video Diffusion](https://comfyanonymous.github.io/ComfyUI_examples/video/)
    - [Mochi](https://comfyanonymous.github.io/ComfyUI_examples/mochi/)
    - [LTX-Video](https://comfyanonymous.github.io/ComfyUI_examples/ltxv/)
    - [Hunyuan Video](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/)
    - [Nvidia Cosmos](https://comfyanonymous.github.io/ComfyUI_examples/cosmos/) and [Cosmos Predict2](https://comfyanonymous.github.io/ComfyUI_examples/cosmos_predict2/)
    - [Wan 2.1](https://comfyanonymous.github.io/ComfyUI_examples/wan/)
- Audio Models
   - [Stable Audio](https://comfyanonymous.github.io/ComfyUI_examples/audio/)
   - [ACE Step](https://comfyanonymous.github.io/ComfyUI_examples/audio/)
- 3D Models
    - [Hunyuan3D 2.0](https://docs.comfy.org/tutorials/3d/hunyuan3D-2)
- Asynchronous Queue system
- Many optimizations: Only re-executes the parts of the workflow that changes between executions.
- Smart memory management: can automatically run models on GPUs with as low as 1GB vram.
- Works even if you don't have a GPU with: ```--cpu``` (slow)
- Can load ckpt, safetensors and diffusers models/checkpoints. Standalone VAEs and CLIP models.
- Embeddings/Textual inversion
- [Loras (regular, locon and loha)](https://comfyanonymous.github.io/ComfyUI_examples/lora/)
- [Hypernetworks](https://comfyanonymous.github.io/ComfyUI_examples/hypernetworks/)
- Loading full workflows (with seeds) from generated PNG, WebP and FLAC files.
- Saving/Loading workflows as Json files.
- Nodes interface can be used to create complex workflows like one for [Hires fix](https://comfyanonymous.github.io/ComfyUI_examples/2_pass_txt2img/) or much more advanced ones.
- [Area Composition](https://comfyanonymous.github.io/ComfyUI_examples/area_composition/)
- [Inpainting](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/) with both regular and inpainting models.
- [ControlNet and T2I-Adapter](https://comfyanonymous.github.io/ComfyUI_examples/controlnet/)
- [Upscale Models (ESRGAN, ESRGAN variants, SwinIR, Swin2SR, etc...)](https://comfyanonymous.github.io/ComfyUI_examples/upscale_models/)
- [unCLIP Models](https://comfyanonymous.github.io/ComfyUI_examples/unclip/)
- [GLIGEN](https://comfyanonymous.github.io/ComfyUI_examples/gligen/)
- [Model Merging](https://comfyanonymous.github.io/ComfyUI_examples/model_merging/)
- [LCM models and Loras](https://comfyanonymous.github.io/ComfyUI_examples/lcm/)
- Latent previews with [TAESD](#how-to-show-high-quality-previews)
- Starts up very fast.
- Works fully offline: core will never download anything unless you want to.
- Optional API nodes to use paid models from external providers through the online [Comfy API](https://docs.comfy.org/tutorials/api-nodes/overview).
- [Config file](extra_model_paths.yaml.example) to set the search paths for models.

### Table of Contents

- [Workflows](https://comfyanonymous.github.io/ComfyUI_examples/)
- [Installing](#installing)
- [Configuration](#command-line-arguments)
- [Installing Custom Nodes](#installing-custom-nodes)
- [Authoring Custom Nodes](#custom-nodes)
- [API](#using-comfyui-as-an-api--programmatically)
- [Distributed](#distributed-multi-process-and-multi-gpu-comfy)

# Getting Started

## Installing

These instructions will install an interactive ComfyUI using the command line.

### Windows

When using Windows, open the **Windows Powershell** app. Then observe you are at a command line, and it is printing "where" you are in your file system: your user directory (e.g., `C:\Users\doctorpangloss`). This is where a bunch of files will go. If you want files to go somewhere else, consult a chat bot for the basics of using command lines, because it is beyond the scope of this document. Then:

1. Install Python 3.12, 3.11 or 3.10. You can do this from the Python website; or, you can use `chocolatey`, a Windows package manager:

   ```shell
   Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   choco install -y python --version 3.12.6
   ```

2. Install `uv`, which makes subsequent installation of Python packages much faster:

    ```shell
    choco install -y uv
    ```

3. Switch into a directory that you want to store your outputs, custom nodes and models in. This is your ComfyUI workspace. For example, if you want to store your workspace in a directory called `ComfyUI_Workspace` in your Documents folder:

   ```powershell
   mkdir ~/Documents/ComfyUI_Workspace
   cd ~/Documents/ComfyUI_Workspace
   ```

4. Create a virtual environment:
   ```shell
   uv venv
   ```
5. Run the following command to install `comfyui` into your current environment. This will correctly select the version of `torch` that matches the GPU on your machine (NVIDIA or CPU on Windows, NVIDIA, Intel, AMD or CPU on Linux, CPU on macOS):
   ```powershell
   uv pip install --torch-backend=auto "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
   ```
6. To run the web server:
   ```shell
   uv run comfyui
   ```
   When you run workflows that use well-known models, this will download them automatically.

   To make it accessible over the network:
   ```shell
   uv run comfyui --listen
   ```

**Running**

On Windows, you should change into the directory where you ran `uv venv`, then run `comfyui`. For example, if you ran `uv venv` inside `~\Documents\ComfyUI_Workspace\`

```powershell
cd ~\Documents\ComfyUI_Workspace\
uv run comfyui
```

Upgrades are delivered frequently and automatically. To force one immediately, run pip upgrade like so:

```shell
uv pip install --torch-backend=auto --upgrade "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
```

### macOS

1. Install Python 3.10, 3.11 or 3.12. This should be achieved by installing `brew`, a macOS package manager, first:
   ```shell
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   Then, install `uv`:
   ```shell
   HOMEBREW_NO_AUTO_UPDATE=1 brew install uv
   ```
3. Switch into a directory that you want to store your outputs, custom nodes and models in. This is your ComfyUI workspace. For example, if you want to store your workspace in a directory called `ComfyUI_Workspace` in your Documents folder:

   ```shell
   mkdir -pv ~/Documents/ComfyUI_Workspace
   cd ~/Documents/ComfyUI_Workspace
   ```

4. Create a virtual environment:
   ```shell
   uv venv --python 3.12
   ```

5. Run the following command to install `comfyui` into your current environment.
   ```shell
   uv pip install "comfyui@git+https://github.com/hiddenswitch/ComfyUI.git"
   ```
6. To run the web server:
   ```shell
   uv run comfyui
   ```
   When you run workflows that use well-known models, this will download them automatically.

   To make it accessible over the network:
   ```shell
   uv run comfyui --listen
   ```

**Running**

On macOS, you will need to open the terminal and `cd` into the directory in which you ran `uv venv`. For example, if you ran `uv venv` in `~/Documents/ComfyUI_Workspace/`:

```shell
cd ~/Documents/ComfyUI_Workspace/
uv run comfyui
```

## Model Downloading

ComfyUI LTS supports downloading models on demand.

Known models will be downloaded from Hugging Face or CivitAI.

To support licensed models like Flux, you will need to login to Hugging Face from the command line.

1. Activate your Python environment by `cd` followed by your workspace directory. For example, if your workspace is located in `~/Documents/ComfyUI_Workspace`, do:

```shell
cd ~/Documents/ComfyUI_Workspace
```

Then, on Windows: `& .venv/scripts/activate.ps1`; on macOS: `source .venv/bin/activate`.

2. Login with Huggingface:

```shell
uv pip install huggingface-cli
huggingface-cli login
```

3. Agree to the terms for a repository. For example, visit https://huggingface.co/black-forest-labs/FLUX.1-dev, login with your HuggingFace account, then choose **Agree**.

To disable model downloading, start with the command line argument `--disable-known-models`: `comfyui --disable-known-models`. However, this will generally only increase your toil for no return.

### Saving Space on Windows

To save space, you will need to enable Developer Mode in the Windows Settings, then reboot your computer. This way, Hugging Face can download models into a common place for all your apps, and place small "link" files that ComfyUI and others can read instead of whole copies of models.

## LTS Custom Nodes

These packages have been adapted to be installable with `pip` and download models to the correct places:

- **ELLA T5 Text Conditioning for SD1.5**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-ella.git`
- **IP Adapter**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-ipadapter-plus`
- **ControlNet Auxiliary Preprocessors**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-controlnet-aux.git`.
- **LayerDiffuse Alpha Channel Diffusion**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-layerdiffuse.git`.
- **BRIA Background Removal**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-bria-bg-removal.git`
- **Video Frame Interpolation**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-video-frame-interpolation`
- **Video Helper Suite**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-video-helper-suite`
- **AnimateDiff Evolved**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-animatediff-evolved`
- **Impact Pack**: `uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-impact-pack`
- **TensorRT**: `uv pip install git+https://github.com/AppMAna/appmana-comfyui-nodes-tensorrt`

Custom nodes are generally supported by this fork. Use these for a bug-free experience.

Request first-class, LTS support for more nodes by [creating a new issue](https://github.com/hiddenswitch/ComfyUI/issues/new). Remember, ordinary custom nodes from the ComfyUI ecosystem work in this fork. Create an issue if you experience a bug or if you think something needs more attention.

##### Running with TLS

To serve with `https://` on Windows easily, use [Caddy](https://github.com/caddyserver/caddy/releases/download/v2.7.6/caddy_2.7.6_windows_amd64.zip). Extract `caddy.exe` to a directory, then run it:

```shell
caddy reverse-proxy --from localhost:443 --to localhost:8188 --tls self_signed
```

##### Notes for AMD Users

Installation for `ROCm` should be explicit:
```shell
uv pip install "comfyui[rocm]@git+https://github.com/hiddenswitch/ComfyUI.git"
```

Then, until a workaround is found, specify these variables:

RDNA 3 (RX 7600 and later)

```shell
export HSA_OVERRIDE_GFX_VERSION=11.0.0
uv run comfyui
```

RDNA 2 (RX 6600 and others)

```shell
export HSA_OVERRIDE_GFX_VERSION=10.3.0
uv run comfyui
```

You can enable experimental memory efficient attention on pytorch 2.5 in ComfyUI on RDNA3 and potentially other AMD GPUs using this command:

```TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention```

You can also try setting this env variable `PYTORCH_TUNABLEOP_ENABLED=1` which might speed things up at the cost of a very slow initial run.

###### Notes for Ascend NPU Users

These instructions from upstream have not yet been validated.

For models compatible with Ascend Extension for PyTorch (`torch_npu`). To get started, ensure your environment meets the prerequisites outlined on the [installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) page. Here's a step-by-step guide tailored to your platform and installation method:

1. Begin by installing the recommended or newer kernel version for Linux as specified in the Installation page of torch-npu, if necessary.
2. Proceed with the installation of Ascend Basekit, which includes the driver, firmware, and CANN, following the instructions provided for your specific platform.
3. Next, install the necessary packages for torch-npu by adhering to the platform-specific instructions on the [Installation](https://ascend.github.io/docs/sources/pytorch/install.html#pytorch) page.
4. Finally, adhere to the [ComfyUI manual installation](#manual-install-windows-linux) guide for Linux. Once all components are installed, you can run ComfyUI as described earlier.

###### Notes for Cambricon MLU Users

These instructions from upstream have not yet been validated.

For models compatible with Cambricon Extension for PyTorch (`torch_mlu`). Here's a step-by-step guide tailored to your platform and installation method:

1. Install the Cambricon CNToolkit by adhering to the platform-specific instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html)
2. Next, install the PyTorch (`torch_mlu`) extension following the instructions on the [Installation](https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html)
3. Launch ComfyUI by running `python main.py`

# Large Language Models

ComfyUI LTS supports text and multi-modal LLM models from the `transformers` ecosystem. This means all the LLaMA family models, LLAVA-NEXT, Phi-3, etc. are supported out-of-the-box with no configuration necessary.

![llava_example_01.gif](docs/assets/llava_example_01.gif)

In this example, LLAVA-NEXT (LLAVA 1.6) is prompted to describe an image.

You can try the [LLAVA-NEXT](tests/inference/workflows/llava-0.json), [Phi-3](tests/inference/workflows/phi-3-0.json), and two [translation](tests/inference/workflows/translation-0.json) [workflows](tests/inference/workflows/translation-1.json).

# SVG Conversion and String Saving

ComfyUI LTS supports powerful SVG conversion capabilities using vtracer and Skia, along with enhanced string saving functionality. This allows for seamless conversion between raster images and SVG format, as well as flexible string saving options.

![SVG Conversion Example](docs/assets/svg_01.png)

In this example, a raster image is converted to SVG, potentially modified, and then converted back to a raster image. The resulting image and SVG code can be saved.

You can try the [SVG Conversion Workflow](tests/inference/workflows/svg-0.json) to explore these features.

# Ideogram

First class support for Ideogram, currently the best still images model.

Visit [API key management](https://ideogram.ai/manage-api) and set the environment variable `IDEOGRAM_API_KEY` to it.

The `IdeogramEdit` node expects the white areas of the mask to be kept, and the black areas of the mask to be inpainted.

Use the **Fit Image to Diffusion Size** with the **Ideogram** resolution set to correctly fit images for inpainting.

# Video Workflows

ComfyUI LTS supports video workflows with AnimateDiff Evolved.

First, install this package using the [Installation Instructions](#installing).

Then, install the custom nodes packages that support video creation workflows:

```shell
uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-video-frame-interpolation
uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-video-helper-suite
uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-animatediff-evolved
uv pip install git+https://github.com/AppMana/appmana-comfyui-nodes-controlnet-aux.git
```

Start creating an AnimateDiff workflow. When using these packages, the appropriate models will download automatically.

## SageAttention

Improve the performance of your Mochi model video generation using **Sage Attention**:

| Device | PyTorch 2.5.1 | SageAttention | S.A. + TorchCompileModel |
|--------|---------------|---------------|--------------------------|
| A5000  | 7.52s/it      | 5.81s/it      | 5.00s/it (but corrupted) |

[Use the default Mochi Workflow.](https://github.com/comfyanonymous/ComfyUI_examples/raw/refs/heads/master/mochi/mochi_text_to_video_example.webp) This does not require any custom nodes or any change to your workflow.

**Installation**

On Windows, you will need the CUDA Toolkit and Visual Studio 2022. If you do not already have this, use `chocolatey`:

```powershell
# install chocolatey
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install -y visualstudio2022buildtools
# purposefully executed separately
choco install -y visualstudio2022-workload-vctools
choco install -y vcredist2010 vcredist2013 vcredist140
```

Then, visit [NVIDIA.com's CUDA Toolkit Download Page](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=Server2022&target_type=exe_network) and download and install the CUDA Toolkit. Verify it is correctly installed by running `nvcc --version`.

You are now ready to install Sage Attention 2 and Triton:

```shell
uv pip install --torch-backend=auto "comfyui[attention]@git+https://github.com/hiddenswitch/ComfyUI.git"
```

To start ComfyUI with it:

```shell
uv run comfyui --use-sage-attention
```

![with_sage_attention.webp](./docs/assets/with_sage_attention.webp)
**With SageAttention**

![with_pytorch_attention](./docs/assets/with_pytorch_attention.webp)
**With PyTorch Attention**

## Cosmos Prompt Upsampling

The Cosmos prompt "upsampler," a fine tune of Mistral-Nemo-12b, correctly rewrites Cosmos prompts in the narrative style that NVIDIA's captioner used for the training data of Cosmos, improving generation results significantly.

Here is a comparison between a simple and "upsampled" prompt.

![prompt_upsampling_01.webp](docs/assets/prompt_upsampling_01.webp)
**A dog is playing with a ball.**

![prompt_upsampling_02.webp](docs/assets/prompt_upsampling_02.webp)
**In a sun-drenched park, a playful golden retriever bounds joyfully across the lush green grass, its tail wagging with excitement. The dog, adorned with a vibrant red collar, is captivated by a bright yellow ball, which it cradles gently in its mouth. The camera captures the dog's animated expressions, from eager anticipation to sheer delight, as it trots and leaps, showcasing its agility and enthusiasm. The scene is bathed in warm, golden-hour light, enhancing the vibrant colors of the dog's fur and the ball. The background features a serene tree line, framing the playful interaction and creating a tranquil atmosphere. The static camera angle allows for an intimate focus on the dog's joyful antics, inviting viewers to share in this heartwarming moment of pure canine happiness.**

To use the Cosmos upsampler, install the prerequisites:

```shell
uv pip install loguru pynvml
uv pip install --no-deps git+https://github.com/NVIDIA/Cosmos.git
```

Then, use the workflow embedded in the upsampled prompt by dragging and dropping the upsampled animation into your workspace.

The Cosmos upsampler ought to improve any text-to-image video generation pipeline. Use the `Video2World` upsampler nodes to download Pixtral-12b and upsample for an image to video workflow using NVIDIA's default prompt. Since Pixtral is not fine tuned, the improvement may not be significant over using another LLM.

# Custom Nodes

Custom Nodes can be added to ComfyUI by copying and pasting Python files into your `./custom_nodes` directory.

## Installing Custom Nodes

There are two kinds of custom nodes: vanilla custom nodes, which generally expect to be dropped into the `custom_nodes` directory and managed by a tool called the ComfyUI Extension manager ("vanilla" custom nodes) and this repository's opinionated, installable custom nodes ("installable").

### Vanilla Custom Nodes

Clone the repository containing the custom nodes into `custom_nodes/` in your working directory. Currently, this is not known to be compatible with ComfyUI Node Manager.

Run `pip install git+https://github.com/owner/repository`, replacing the `git` repository with the installable custom nodes URL. This is just the GitHub URL.

## Authoring Custom Nodes

These instructions will allow you to quickly author installable custom nodes.

#### Using `pyproject.toml` for projects with existing `requirements.txt`

Suppose your custom nodes called `my_comfyui_nodes` has a folder layout that looks like this:

```
__init__.py
some_python_file.py
requirements.txt
LICENSE.txt
some_directory/some_code.py
```

First, add an `__init__.py` to `some_directory`, so that it is a Python package:

```
__init__.py
some_python_file.py
requirements.txt
LICENSE.txt
some_directory/__init__.py
some_directory/some_code.py
```

Then, if your `NODE_CLASS_MAPPINGS` are declared in `__init__.py`, use the following as a `pyproject.toml`, substituting your actual project name:

**pyproject.toml**

```toml
[project]
name = "my_comfyui_nodes"
description = "My nodes description."
version = "1.0.0"
license = { file = "LICENSE.txt" }
dynamic = ["dependencies"]

[project.urls]
Repository = "https://github.com/your-github-username/my-comfyui-nodes"
#  Used by Comfy Registry https://comfyregistry.org

[tool.comfy]
PublisherId = "your-github-username"
DisplayName = "my_comfyui_nodes"
Icon = ""

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["my_comfyui_nodes", "my_comfyui_nodes.some_directory"]
package-dir = { "my_comfyui_nodes" = ".", "my_comfyui_nodes.some_directory" = "some_directory" }

[tool.setuptools.dynamic]
dependencies = { file = ["requirements.txt"] }

[project.entry-points."comfyui.custom_nodes"]
my_comfyui_nodes = "my_comfyui_nodes"
```

Observe that the directory should now be listed as a package in the `packages` and `package-dir` statement.

#### Using `setup.py`

Create a `requirements.txt`:

```
comfyui
```

Observe `comfyui` is now a requirement for using your custom nodes. This will ensure you will be able to access `comfyui` as a library. For example, your code will now be able to import the folder paths using `from comfyui.cmd import folder_paths`. Because you will be using my fork, use this:

```
comfyui @ git+https://github.com/hiddenswitch/ComfyUI.git
```

Additionally, create a `pyproject.toml`:

```
[build-system]
requires = ["setuptools", "wheel", "pip"]
build-backend = "setuptools.build_meta"
```

This ensures you will be compatible with later versions of Python.

Finally, move your nodes to a directory with an empty `__init__.py`, i.e., a package. You should have a file structure like this:

```
# the root of your git repository
/.git
/pyproject.toml
/requirements.txt
/mypackage_custom_nodes/__init__.py
/mypackage_custom_nodes/some_nodes.py
```

Finally, create a `setup.py` at the root of your custom nodes package / repository. Here is an example:

**setup.py**

```python
from setuptools import setup, find_packages
import os.path

setup(
    name="mypackage",
    version="0.0.1",
    packages=find_packages(),
    install_requires=open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines(),
    author='',
    author_email='',
    description='',
    entry_points={
        'comfyui.custom_nodes': [
            'mypackage = mypackage_custom_nodes',
        ],
    },
)
```

All `.py` files located in the package specified by the entrypoint with your package's name will be scanned for node class mappings declared like this:

**some_nodes.py**:

```py
from comfy.nodes.package_typing import CustomNode


class Binary_Preprocessor(CustomNode):
    ...


NODE_CLASS_MAPPINGS = {
    "BinaryPreprocessor": Binary_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BinaryPreprocessor": "Binary Lines"
}
```

These packages will be scanned recursively.

Extending the `comfy.nodes.package_typing.CustomNode` provides type hints for authoring nodes.

## Adding Custom Configuration

Declare an entry point for configuration hooks in your **setup.py** that defines a function that takes and returns an
`configargparser.ArgParser` object:

**setup.py**

```python
setup(
    name="mypackage",
    ...
entry_points = {
    'comfyui.custom_nodes': [
        'mypackage = mypackage_custom_nodes',
    ],
    'comfyui.custom_config': [
        'mypackage = mypackage_custom_config:add_configuration',
    ]
},
)
```

**mypackage_custom_config.py**:

```python
import configargparse


def add_configuration(parser: configargparse.ArgParser) -> configargparse.ArgParser:
    parser.add_argument("--openai-api-key",
                        required=False,
                        type=str,
                        help="Configures the OpenAI API Key for the OpenAI nodes", env_var="OPENAI_API_KEY")
    return parser

```

You can now see your configuration option at the bottom of the `--help` command along with hints for how to use it:

```shell
$ comfyui --help
usage: comfyui.exe [-h] [-c CONFIG_FILE] [--write-out-config-file CONFIG_OUTPUT_PATH] [-w CWD] [-H [IP]] [--port PORT]
                   [--enable-cors-header [ORIGIN]] [--max-upload-size MAX_UPLOAD_SIZE] [--extra-model-paths-config PATH [PATH ...]]
...
                   [--openai-api-key OPENAI_API_KEY]

options:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config CONFIG_FILE
                        config file path
  --write-out-config-file CONFIG_OUTPUT_PATH
                        takes the current command line args and writes them out to a config file at the given path, then exits
  -w CWD, --cwd CWD     Specify the working directory. If not set, this is the current working directory. models/, input/, output/ and other
                        directories will be located here by default. [env var: COMFYUI_CWD]
  -H [IP], --listen [IP]
                        Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to
                        0.0.0.0. (listens on all) [env var: COMFYUI_LISTEN]
  --port PORT           Set the listen port. [env var: COMFYUI_PORT]
...
  --distributed-queue-name DISTRIBUTED_QUEUE_NAME
                        This name will be used by the frontends and workers to exchange prompt requests and replies. Progress updates will be
                        prefixed by the queue name, followed by a '.', then the user ID [env var: COMFYUI_DISTRIBUTED_QUEUE_NAME]
  --external-address EXTERNAL_ADDRESS
                        Specifies a base URL for external addresses reported by the API, such as for image paths. [env var:
                        COMFYUI_EXTERNAL_ADDRESS]
  --openai-api-key OPENAI_API_KEY
                        Configures the OpenAI API Key for the OpenAI nodes [env var: OPENAI_API_KEY]
```

You can now start `comfyui` with:

```shell
comfyui --openai-api-key=abcdefg12345
```

or set the environment variable you specified:

```shell
export OPENAI_API_KEY=abcdefg12345
comfyui
```

or add it to your config file:

**config.yaml**:

```txt
openapi-api-key: abcdefg12345
```

```shell
comfyui --config config.yaml
```

Since `comfyui` looks for a `config.yaml` in your current working directory by default, you can omit the argument if
`config.yaml` is located in your current working directory:

```shell
comfyui
```

Your entry point for adding configuration options should **not** import your nodes. This gives you the opportunity to
use the configuration you added in your nodes; otherwise, if you imported your nodes in your configuration entry point,
the nodes will potentially be initialized without any configuration.

Access your configuration from `cli_args`:

```python
from comfy.cli_args import args
from comfy.cli_args_types import Configuration
from typing import Optional


# Add type hints when accessing args
class CustomConfiguration(Configuration):
    def __init__(self):
        super().__init__()
        self.openai_api_key: Optional[str] = None


args: CustomConfiguration


class OpenAINode(CustomNode):
    ...

    def execute(self):
        openai_api_key = args.open_api_key
```

# Troubleshooting

> I see a message like `RuntimeError: '"upsample_bilinear2d_channels_last" not implemented for 'Half''`

You must use Python 3.11 on macOS devices, and update to at least Ventura.

> I see a message like `Error while deserializing header: HeaderTooLarge`

Download your model file again.

# Using the Editor

## Notes

Only parts of the graph that have an output with all the correct inputs will be executed.

Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.

Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.

You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).

You can use {day|night}, for wildcard/dynamic prompts. With this syntax "{wild|card|test}" will be randomly replaced by either "wild", "card" or "test" by the frontend every time you queue the prompt. To use {} characters in your actual prompt escape them like: \\{ or \\}.

Dynamic prompts also support C-style comments, like `// comment` or `/* comment */`.

To use a textual inversion concepts/embeddings in a text prompt put them in the models/embeddings directory and use them in the CLIPTextEncode node like this (you can omit the .pt extension):

```embedding:embedding_filename.pt```

##### How to increase generation speed?

Make sure you use the regular loaders/Load Checkpoint node to load checkpoints. It will auto pick the right settings depending on your GPU.

You can set this command line setting to disable the upcasting to fp32 in some cross attention operations which will increase your speed. Note that this will very likely give you black images on SD2.x models. If you use xformers or pytorch attention this option does not do anything.

```--dont-upcast-attention```

##### How to show high-quality previews?

Use ```--preview-method auto``` to enable previews.

The default installation includes a fast latent preview method that's low-resolution. To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download the [taesd_decoder.pth](https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth) (for SD1.x and SD2.x) and [taesdxl_decoder.pth](https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth) (for SDXL) models and place them in the `models/vae_approx` folder. Once they're installed, restart ComfyUI to enable high-quality previews.

## Keyboard Shortcuts

| Keybind                                | Explanation                                                                                                        |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                       | Queue up current graph for generation                                                                              |
| `Ctrl` + `Shift` + `Enter`             | Queue up current graph as first for generation                                                                     |
| `Ctrl` + `Alt` + `Enter`               | Cancel current generation                                                                                          |
| `Ctrl` + `Z`/`Ctrl` + `Y`              | Undo/Redo                                                                                                          |
| `Ctrl` + `S`                           | Save workflow                                                                                                      |
| `Ctrl` + `O`                           | Load workflow                                                                                                      |
| `Ctrl` + `A`                           | Select all nodes                                                                                                   |
| `Alt `+ `C`                            | Collapse/uncollapse selected nodes                                                                                 |
| `Ctrl` + `M`                           | Mute/unmute selected nodes                                                                                         |
| `Ctrl` + `B`                           | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)            |
| `Delete`/`Backspace`                   | Delete selected nodes                                                                                              |
| `Ctrl` + `Backspace`                   | Delete the current graph                                                                                           |
| `Space`                                | Move the canvas around when held and moving the cursor                                                             |
| `Ctrl`/`Shift` + `Click`               | Add clicked node to selection                                                                                      |
| `Ctrl` + `C`/`Ctrl` + `V`              | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                     |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`    | Copy and paste selected nodes (maintaining connections from outputs of unselected nodes to inputs of pasted nodes) |
| `Shift` + `Drag`                       | Move multiple selected nodes at the same time                                                                      |
| `Ctrl` + `D`                           | Load default graph                                                                                                 |
| `Alt` + `+`                            | Canvas Zoom in                                                                                                     |
| `Alt` + `-`                            | Canvas Zoom out                                                                                                    |
| `Ctrl` + `Shift` + LMB + Vertical drag | Canvas Zoom in/out                                                                                                 |
| `P`                                    | Pin/Unpin selected nodes                                                                                           |
| `Ctrl` + `G`                           | Group selected nodes                                                                                               |
| `Q`                                    | Toggle visibility of the queue                                                                                     |
| `H`                                    | Toggle visibility of history                                                                                       |
| `R`                                    | Refresh graph                                                                                                      |
| `F`                                    | Show/Hide menu                                                                                                     |
| `.`                                    | Fit view to selection (Whole graph when nothing is selected)                                                       |
| Double-Click LMB                       | Open node quick search palette                                                                                     |
| `Shift` + Drag                         | Move multiple wires at once                                                                                        |
| `Ctrl` + `Alt` + LMB                   | Disconnect all wires from clicked slot                                                                             |

`Ctrl` can also be replaced with `Cmd` instead for macOS users

# Configuration

This supports configuration with command line arguments, the environment and a configuration file.

## Configuration File

First, run `comfyui --help` for all supported configuration and arguments.

Args that start with `--` can also be set in a config file (`config.yaml` or `config.json` or specified via `-c`). Config file syntax allows: `key=value`, `flag=true`, `stuff=[a,b,c]` (for details, see syntax [here](https://goo.gl/R74nmi)). In general, command-line values override environment variables which override config file values which override defaults.

## Extra Model Paths

Copy [docs/examples/configuration/extra_model_paths.yaml](docs/examples/configuration/extra_model_paths.yaml) to your working directory, and modify the folder paths to match your folder structure.

You can pass additional extra model path configurations with one or more copies of `--extra-model-paths-config=some_configuration.yaml`.

### Command Line Arguments

```
usage: comfyui [-h] [-c CONFIG_FILE] [--write-out-config-file CONFIG_OUTPUT_PATH] [-w CWD] [--base-paths BASE_PATHS [BASE_PATHS ...]] [-H [IP]] [--port PORT]
               [--enable-cors-header [ORIGIN]] [--max-upload-size MAX_UPLOAD_SIZE] [--base-directory BASE_DIRECTORY] [--extra-model-paths-config PATH [PATH ...]]
               [--output-directory OUTPUT_DIRECTORY] [--temp-directory TEMP_DIRECTORY] [--input-directory INPUT_DIRECTORY] [--auto-launch] [--disable-auto-launch]
               [--cuda-device DEVICE_ID] [--cuda-malloc | --disable-cuda-malloc] [--force-fp32 | --force-fp16 | --force-bf16]
               [--fp32-unet | --fp64-unet | --bf16-unet | --fp16-unet | --fp8_e4m3fn-unet | --fp8_e5m2-unet] [--fp16-vae | --fp32-vae | --bf16-vae] [--cpu-vae]
               [--fp8_e4m3fn-text-enc | --fp8_e5m2-text-enc | --fp16-text-enc | --fp32-text-enc | --bf16-text-enc] [--directml [DIRECTML_DEVICE]]
               [--oneapi-device-selector SELECTOR_STRING] [--disable-ipex-optimize] [--preview-method [none,auto,latent2rgb,taesd]] [--preview-size PREVIEW_SIZE]
               [--cache-classic | --cache-lru CACHE_LRU | --cache-none]
               [--use-split-cross-attention | --use-quad-cross-attention | --use-pytorch-cross-attention | --use-sage-attention | --use-flash-attention] [--disable-xformers]
               [--force-upcast-attention | --dont-upcast-attention] [--gpu-only | --highvram | --normalvram | --lowvram | --novram | --cpu] [--reserve-vram RESERVE_VRAM]
               [--default-hashing-function {md5,sha1,sha256,sha512}] [--disable-smart-memory] [--deterministic] [--fast [FAST ...]] [--dont-print-server] [--quick-test-for-ci]
               [--windows-standalone-build] [--disable-metadata] [--disable-all-custom-nodes] [--multi-user] [--create-directories] [--log-stdout]
               [--plausible-analytics-base-url PLAUSIBLE_ANALYTICS_BASE_URL] [--plausible-analytics-domain PLAUSIBLE_ANALYTICS_DOMAIN] [--analytics-use-identity-provider]
               [--distributed-queue-connection-uri DISTRIBUTED_QUEUE_CONNECTION_URI] [--distributed-queue-worker] [--distributed-queue-frontend]
               [--distributed-queue-name DISTRIBUTED_QUEUE_NAME] [--external-address EXTERNAL_ADDRESS] [--logging-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               [--disable-known-models] [--max-queue-size MAX_QUEUE_SIZE] [--otel-service-name OTEL_SERVICE_NAME] [--otel-service-version OTEL_SERVICE_VERSION]
               [--otel-exporter-otlp-endpoint OTEL_EXPORTER_OTLP_ENDPOINT] [--force-channels-last] [--force-hf-local-dir-mode] [--front-end-version FRONT_END_VERSION]
               [--panic-when PANIC_WHEN] [--front-end-root FRONT_END_ROOT] [--executor-factory EXECUTOR_FACTORY] [--openai-api-key OPENAI_API_KEY]
               [--ideogram-api-key IDEOGRAM_API_KEY] [--anthropic-api-key ANTHROPIC_API_KEY] [--user-directory USER_DIRECTORY] [--enable-compress-response-body]
               [--workflows WORKFLOWS [WORKFLOWS ...]]

options:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config CONFIG_FILE
                        config file path
  --write-out-config-file CONFIG_OUTPUT_PATH
                        takes the current command line args and writes them out to a config file at the given path, then exits
  -w CWD, --cwd CWD     Specify the working directory. If not set, this is the current working directory. models/, input/, output/ and other directories will be located here by
                        default. [env var: COMFYUI_CWD]
  --base-paths BASE_PATHS [BASE_PATHS ...]
                        Additional base paths for custom nodes, models and inputs. [env var: COMFYUI_BASE_PATHS]
  -H [IP], --listen [IP]
                        Specify the IP address to listen on (default: 127.0.0.1). You can give a list of ip addresses by separating them with a comma like: 127.2.2.2,127.3.3.3
                        If --listen is provided without an argument, it defaults to 0.0.0.0,:: (listens on all ipv4 and ipv6) [env var: COMFYUI_LISTEN]
  --port PORT           Set the listen port. [env var: COMFYUI_PORT]
  --enable-cors-header [ORIGIN]
                        Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'. [env var: COMFYUI_ENABLE_CORS_HEADER]
  --max-upload-size MAX_UPLOAD_SIZE
                        Set the maximum upload size in MB. [env var: COMFYUI_MAX_UPLOAD_SIZE]
  --base-directory BASE_DIRECTORY
                        Set the ComfyUI base directory for models, custom_nodes, input, output, temp, and user directories. [env var: COMFYUI_BASE_DIRECTORY]
  --extra-model-paths-config PATH [PATH ...]
                        Load one or more extra_model_paths.yaml files. [env var: COMFYUI_EXTRA_MODEL_PATHS_CONFIG]
  --output-directory OUTPUT_DIRECTORY
                        Set the ComfyUI output directory. Overrides --base-directory. [env var: COMFYUI_OUTPUT_DIRECTORY]
  --temp-directory TEMP_DIRECTORY
                        Set the ComfyUI temp directory (default is in the ComfyUI directory). Overrides --base-directory. [env var: COMFYUI_TEMP_DIRECTORY]
  --input-directory INPUT_DIRECTORY
                        Set the ComfyUI input directory. Overrides --base-directory. [env var: COMFYUI_INPUT_DIRECTORY]
  --auto-launch         Automatically launch ComfyUI in the default browser. [env var: COMFYUI_AUTO_LAUNCH]
  --disable-auto-launch
                        Disable auto launching the browser. [env var: COMFYUI_DISABLE_AUTO_LAUNCH]
  --cuda-device DEVICE_ID
                        Set the id of the cuda device this instance will use. [env var: COMFYUI_CUDA_DEVICE]
  --cuda-malloc         Enable cudaMallocAsync (enabled by default for torch 2.0 and up). [env var: COMFYUI_CUDA_MALLOC]
  --disable-cuda-malloc
                        Disable cudaMallocAsync. [env var: COMFYUI_DISABLE_CUDA_MALLOC]
  --force-fp32          Force fp32 (If this makes your GPU work better please report it). [env var: COMFYUI_FORCE_FP32]
  --force-fp16          Force fp16. [env var: COMFYUI_FORCE_FP16]
  --force-bf16          Force bf16. [env var: COMFYUI_FORCE_BF16]
  --fp32-unet           Run the diffusion model in fp32. [env var: COMFYUI_FP32_UNET]
  --fp64-unet           Run the diffusion model in fp64. [env var: COMFYUI_FP64_UNET]
  --bf16-unet           Run the diffusion model in bf16. [env var: COMFYUI_BF16_UNET]
  --fp16-unet           Run the diffusion model in fp16 [env var: COMFYUI_FP16_UNET]
  --fp8_e4m3fn-unet     Store unet weights in fp8_e4m3fn. [env var: COMFYUI_FP8_E4M3FN_UNET]
  --fp8_e5m2-unet       Store unet weights in fp8_e5m2. [env var: COMFYUI_FP8_E5M2_UNET]
  --fp16-vae            Run the VAE in fp16, might cause black images. [env var: COMFYUI_FP16_VAE]
  --fp32-vae            Run the VAE in full precision fp32. [env var: COMFYUI_FP32_VAE]
  --bf16-vae            Run the VAE in bf16. [env var: COMFYUI_BF16_VAE]
  --cpu-vae             Run the VAE on the CPU. [env var: COMFYUI_CPU_VAE]
  --fp8_e4m3fn-text-enc
                        Store text encoder weights in fp8 (e4m3fn variant). [env var: COMFYUI_FP8_E4M3FN_TEXT_ENC]
  --fp8_e5m2-text-enc   Store text encoder weights in fp8 (e5m2 variant). [env var: COMFYUI_FP8_E5M2_TEXT_ENC]
  --fp16-text-enc       Store text encoder weights in fp16. [env var: COMFYUI_FP16_TEXT_ENC]
  --fp32-text-enc       Store text encoder weights in fp32. [env var: COMFYUI_FP32_TEXT_ENC]
  --bf16-text-enc       Store text encoder weights in bf16. [env var: COMFYUI_BF16_TEXT_ENC]
  --directml [DIRECTML_DEVICE]
                        Use torch-directml. [env var: COMFYUI_DIRECTML]
  --oneapi-device-selector SELECTOR_STRING
                        Sets the oneAPI device(s) this instance will use. [env var: COMFYUI_ONEAPI_DEVICE_SELECTOR]
  --disable-ipex-optimize
                        Disables ipex.optimize default when loading models with Intel's Extension for Pytorch. [env var: COMFYUI_DISABLE_IPEX_OPTIMIZE]
  --preview-method [none,auto,latent2rgb,taesd]
                        Default preview method for sampler nodes. [env var: COMFYUI_PREVIEW_METHOD]
  --preview-size PREVIEW_SIZE
                        Sets the maximum preview size for sampler nodes. [env var: COMFYUI_PREVIEW_SIZE]
  --cache-classic       WARNING: Unused. Use the old style (aggressive) caching. [env var: COMFYUI_CACHE_CLASSIC]
  --cache-lru CACHE_LRU
                        Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM. [env var: COMFYUI_CACHE_LRU]
  --cache-none          Reduced RAM/VRAM usage at the expense of executing every node for each run. [env var: COMFYUI_CACHE_NONE]
  --use-split-cross-attention
                        Use the split cross attention optimization. Ignored when xformers is used. [env var: COMFYUI_USE_SPLIT_CROSS_ATTENTION]
  --use-quad-cross-attention
                        Use the sub-quadratic cross attention optimization . Ignored when xformers is used. [env var: COMFYUI_USE_QUAD_CROSS_ATTENTION]
  --use-pytorch-cross-attention
                        Use the new pytorch 2.0 cross attention function (default). [env var: COMFYUI_USE_PYTORCH_CROSS_ATTENTION]
  --use-sage-attention  Use sage attention. [env var: COMFYUI_USE_SAGE_ATTENTION]
  --use-flash-attention
                        Use FlashAttention. [env var: COMFYUI_USE_FLASH_ATTENTION]
  --disable-xformers    Disable xformers. [env var: COMFYUI_DISABLE_XFORMERS]
  --force-upcast-attention
                        Force enable attention upcasting, please report if it fixes black images. [env var: COMFYUI_FORCE_UPCAST_ATTENTION]
  --dont-upcast-attention
                        Disable all upcasting of attention. Should be unnecessary except for debugging. [env var: COMFYUI_DONT_UPCAST_ATTENTION]
  --gpu-only            Store and run everything (text encoders/CLIP models, etc... on the GPU). [env var: COMFYUI_GPU_ONLY]
  --highvram            By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory. [env var: COMFYUI_HIGHVRAM]
  --normalvram          Used to force normal vram use if lowvram gets automatically enabled. [env var: COMFYUI_NORMALVRAM]
  --lowvram             Split the unet in parts to use less vram. [env var: COMFYUI_LOWVRAM]
  --novram              When lowvram isn't enough. [env var: COMFYUI_NOVRAM]
  --cpu                 To use the CPU for everything (slow). [env var: COMFYUI_CPU]
  --reserve-vram RESERVE_VRAM
                        Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reserved depending on your OS. [env var:
                        COMFYUI_RESERVE_VRAM]
  --default-hashing-function {md5,sha1,sha256,sha512}
                        Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256. [env var:
                        COMFYUI_DEFAULT_HASHING_FUNCTION]
  --disable-smart-memory
                        Force ComfyUI to agressively offload to regular ram instead of keeping models in vram when it can. [env var: COMFYUI_DISABLE_SMART_MEMORY]
  --deterministic       Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases. [env var:
                        COMFYUI_DETERMINISTIC]
  --fast [FAST ...]     Enable some untested and potentially quality deteriorating optimizations. Pass a list specific optimizations if you only want to enable specific ones.
                        Current valid optimizations: fp16_accumulation fp8_matrix_mult cublas_ops [env var: COMFYUI_FAST]
  --dont-print-server   Don't print server output. [env var: COMFYUI_DONT_PRINT_SERVER]
  --quick-test-for-ci   Quick test for CI. Raises an error if nodes cannot be imported, [env var: COMFYUI_QUICK_TEST_FOR_CI]
  --windows-standalone-build
                        Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page
                        on startup). [env var: COMFYUI_WINDOWS_STANDALONE_BUILD]
  --disable-metadata    Disable saving prompt metadata in files. [env var: COMFYUI_DISABLE_METADATA]
  --disable-all-custom-nodes
                        Disable loading all custom nodes. [env var: COMFYUI_DISABLE_ALL_CUSTOM_NODES]
  --multi-user          Enables per-user storage. [env var: COMFYUI_MULTI_USER]
  --create-directories  Creates the default models/, input/, output/ and temp/ directories, then exits. [env var: COMFYUI_CREATE_DIRECTORIES]
  --log-stdout          Send normal process output to stdout instead of stderr (default). [env var: COMFYUI_LOG_STDOUT]
  --plausible-analytics-base-url PLAUSIBLE_ANALYTICS_BASE_URL
                        Enables server-side analytics events sent to the provided URL. [env var: COMFYUI_PLAUSIBLE_ANALYTICS_BASE_URL]
  --plausible-analytics-domain PLAUSIBLE_ANALYTICS_DOMAIN
                        Specifies the domain name for analytics events. [env var: COMFYUI_PLAUSIBLE_ANALYTICS_DOMAIN]
  --analytics-use-identity-provider
                        Uses platform identifiers for unique visitor analytics. [env var: COMFYUI_ANALYTICS_USE_IDENTITY_PROVIDER]
  --distributed-queue-connection-uri DISTRIBUTED_QUEUE_CONNECTION_URI
                        EXAMPLE: "amqp://guest:guest@127.0.0.1" - Servers and clients will connect to this AMPQ URL to form a distributed queue and exchange prompt execution
                        requests and progress updates. [env var: COMFYUI_DISTRIBUTED_QUEUE_CONNECTION_URI]
  --distributed-queue-worker
                        Workers will pull requests off the AMQP URL. [env var: COMFYUI_DISTRIBUTED_QUEUE_WORKER]
  --distributed-queue-frontend
                        Frontends will start the web UI and connect to the provided AMQP URL to submit prompts. [env var: COMFYUI_DISTRIBUTED_QUEUE_FRONTEND]
  --distributed-queue-name DISTRIBUTED_QUEUE_NAME
                        This name will be used by the frontends and workers to exchange prompt requests and replies. Progress updates will be prefixed by the queue name,
                        followed by a '.', then the user ID [env var: COMFYUI_DISTRIBUTED_QUEUE_NAME]
  --external-address EXTERNAL_ADDRESS
                        Specifies a base URL for external addresses reported by the API, such as for image paths. [env var: COMFYUI_EXTERNAL_ADDRESS]
  --logging-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level [env var: COMFYUI_LOGGING_LEVEL]
  --disable-known-models
                        Disables automatic downloads of known models and prevents them from appearing in the UI. [env var: COMFYUI_DISABLE_KNOWN_MODELS]
  --max-queue-size MAX_QUEUE_SIZE
                        The API will reject prompt requests if the queue's size exceeds this value. [env var: COMFYUI_MAX_QUEUE_SIZE]
  --otel-service-name OTEL_SERVICE_NAME
                        The name of the service or application that is generating telemetry data. [env var: OTEL_SERVICE_NAME]
  --otel-service-version OTEL_SERVICE_VERSION
                        The version of the service or application that is generating telemetry data. [env var: OTEL_SERVICE_VERSION]
  --otel-exporter-otlp-endpoint OTEL_EXPORTER_OTLP_ENDPOINT
                        A base endpoint URL for any signal type, with an optionally-specified port number. Helpful for when you're sending more than one signal to the same
                        endpoint and want one environment variable to control the endpoint. [env var: OTEL_EXPORTER_OTLP_ENDPOINT]
  --force-channels-last
                        Force channels last format when inferencing the models. [env var: COMFYUI_FORCE_CHANNELS_LAST]
  --force-hf-local-dir-mode
                        Download repos from huggingface.co to the models/huggingface directory with the "local_dir" argument instead of models/huggingface_cache with the
                        "cache_dir" argument, recreating the traditional file structure. [env var: COMFYUI_FORCE_HF_LOCAL_DIR_MODE]
  --front-end-version FRONT_END_VERSION
                        Specifies the version of the frontend to be used. This command needs internet connectivity to query and download available frontend implementations from
                        GitHub releases. The version string should be in the format of: [repoOwner]/[repoName]@[version] where version is one of: "latest" or a valid version
                        number (e.g. "1.0.0") [env var: COMFYUI_FRONT_END_VERSION]
  --panic-when PANIC_WHEN
                        List of fully qualified exception class names to panic (sys.exit(1)) when a workflow raises it. Example: --panic-when=torch.cuda.OutOfMemoryError. Can
                        be specified multiple times or as a comma-separated list. [env var: COMFYUI_PANIC_WHEN]
  --front-end-root FRONT_END_ROOT
                        The local filesystem path to the directory where the frontend is located. Overrides --front-end-version. [env var: COMFYUI_FRONT_END_ROOT]
  --executor-factory EXECUTOR_FACTORY
                        When running ComfyUI as a distributed worker, this specifies the kind of executor that should be used to run the actual ComfyUI workflow worker. A
                        ThreadPoolExecutor is the default. A ProcessPoolExecutor results in better memory management, since the process will be closed and large, contiguous
                        blocks of CUDA memory can be freed. [env var: COMFYUI_EXECUTOR_FACTORY]
  --openai-api-key OPENAI_API_KEY
                        Configures the OpenAI API Key for the OpenAI nodes. Visit https://platform.openai.com/api-keys to create this key. [env var: OPENAI_API_KEY]
  --ideogram-api-key IDEOGRAM_API_KEY
                        Configures the Ideogram API Key for the Ideogram nodes. Visit https://ideogram.ai/manage-api to create this key. [env var: IDEOGRAM_API_KEY]
  --anthropic-api-key ANTHROPIC_API_KEY
                        Configures the Anthropic API key for its nodes related to Claude functionality. Visit https://console.anthropic.com/settings/keys to create this key.
                        [env var: ANTHROPIC_API_KEY]
  --user-directory USER_DIRECTORY
                        Set the ComfyUI user directory with an absolute path. Overrides --base-directory. [env var: COMFYUI_USER_DIRECTORY]
  --enable-compress-response-body
                        Enable compressing response body. [env var: COMFYUI_ENABLE_COMPRESS_RESPONSE_BODY]
  --workflows WORKFLOWS [WORKFLOWS ...]
                        Execute the API workflow(s) specified in the provided files. For each workflow, its outputs will be printed to a line to standard out. Application
                        logging will be redirected to standard error. Use `-` to signify standard in. [env var: COMFYUI_WORKFLOWS]
                        
Args that start with '--' can also be set in a config file (config.yaml or config.json or specified via -c). Config file syntax allows: key=value, flag=true, stuff=[a,b,c] (for details, see syntax at
https://goo.gl/R74nmi). In general, command-line values override environment variables which override config file values which override defaults.
```

# Using ComfyUI as an API / Programmatically

There are multiple ways to use this ComfyUI package to run workflows programmatically:

### Embedded

Start ComfyUI by creating an ordinary Python object. This does not create a web server. It runs ComfyUI as a library, like any other package you are familiar with:

```python
from comfy.client.embedded_comfy_client import Comfy

async with Comfy() as client:
    # This will run your prompt
    # To get the prompt JSON, visit the ComfyUI interface, design your workflow and click **Save (API Format)**. This JSON is what you will use as your workflow.
    outputs = await client.queue_prompt(prompt)
    # At this point, your prompt is finished and all the outputs, like saving images, have been completed.
    # Now the outputs will contain the same thing that the Web UI expresses: a file path for each output.
    # Let's find the node ID of the first SaveImage node. This will work when you change your workflow JSON from
    # the example above.
    save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
    # Now let's print the absolute path to the image.
    print(outputs[save_image_node_id]["images"][0]["abs_path"])
# At this point, all the models have been unloaded from VRAM, and everything has been cleaned up.
```

See [script_examples/basic_api_example.py](docs/examples/script_examples/basic_api_example.py) for a complete example.

### Remote

Visit the ComfyUI interface, design your workflow and click **Save (API Format)**. This JSON is what you will use as your workflow.

You can use the built-in Python client library by installing this package without its dependencies.

```shell
pip install aiohttp
pip install --no-deps git+https://github.com/hiddenswitch/ComfyUI.git
```

Then the following idiomatic pattern is available:

```python
from comfy.client.aio_client import AsyncRemoteComfyClient

client = AsyncRemoteComfyClient(server_address="http://localhost:8188")
# Now let's get the bytes of the PNG image saved by the SaveImage node:
png_image_bytes = await client.queue_prompt(prompt)
# You can save these bytes wherever you need!
with open("image.png", "rb") as f:
    f.write(png_image_bytes)
```

See [script_examples/remote_api_example.py](docs/examples/script_examples/remote_api_example.py) for a complete example.

##### REST API

First, install this package using the [Installation Instructions](#installing). Then, run `comfyui`.

Visit the ComfyUI interface, design your workflow and click **Save (API Format)**. This JSON is what you will use as your workflow.

Then, send a request to `api/v1/prompts`. Here are some examples:

**`curl`**:

```shell
curl -X POST "http://localhost:8188/api/v1/prompts" \
     -H "Content-Type: application/json" \
     -H "Accept: image/png" \
     -o output.png \
     -d '{
       "prompt": {
         # ... (include the rest of the workflow)
       }
     }'
```

**Python**:

```python
import requests

url = "http://localhost:8188/api/v1/prompts"
headers = {
    "Content-Type": "application/json",
    "Accept": "image/png"
}
workflow = {
    "4": {
        "inputs": {
            "ckpt_name": "sd_xl_base_1.0.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    # ... (include the rest of the workflow)
}

payload = {"prompt": workflow}

response = requests.post(url, json=payload, headers=headers)
```

**Javascript (Browser)**:

```javascript
async function generateImage() {
    const prompt = "a man walking on the beach";
    const workflow = {
        "4": {
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        // ... (include the rest of the workflow)
    };

    const response = await fetch('http://localhost:8188/api/v1/prompts', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'image/png'
        },
        body: JSON.stringify({prompt: workflow})
    });

    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);
    const img = document.createElement('img');
    // load image into the DOM
    img.src = imageUrl;
    document.body.appendChild(img);
}

generateImage().catch(console.error);
```

You can use the OpenAPI specification file to learn more about all the supported API methods.

### OpenAPI Spec for Vanilla API, Typed Clients

Use a typed, generated API client for your programming language and access ComfyUI server remotely as an API.

You can generate the client from [comfy/api/openapi.yaml](comfy/api/openapi.yaml).

### RabbitMQ / AMQP Support

Submit jobs directly to a distributed work queue. This package supports AMQP message queues like RabbitMQ. You can submit workflows to the queue, including from the web using RabbitMQ's STOMP support, and receive realtime progress updates from multiple workers. Continue to the next section for more details.

# Distributed, Multi-Process and Multi-GPU Comfy

This package supports multi-processing across machines using RabbitMQ. This means you can launch multiple ComfyUI backend workers and queue prompts against them from multiple frontends.

## Getting Started

ComfyUI has two roles: `worker` and `frontend`. An unlimited number of workers can consume and execute workflows (prompts) in parallel; and an unlimited number of frontends can submit jobs. All of the frontends' API calls will operate transparently against your collection of workers, including progress notifications from the websocket.

To share work among multiple workers and frontends, ComfyUI uses RabbitMQ or any AMQP-compatible message queue like SQS or Kafka.

### Example with RabbitMQ and File Share

On a machine in your local network, install **Docker** and run RabbitMQ:

```shell
docker run -it --rm --name rabbitmq -p 5672:5672 rabbitmq:latest
```

Find the machine's main LAN IP address:

**Windows (PowerShell)**:

```pwsh
Get-NetIPConfiguration | Where-Object { $_.InterfaceAlias -like '*Ethernet*' -and $_.IPv4DefaultGateway -ne $null } | ForEach-Object { $_.IPv4Address.IPAddress }
```

**Linux**

```shell
ip -4 addr show $(ip route show default | awk '/default/ {print $5}') | grep -oP 'inet \K[\d.]+'
```

**macOS**

```shell
ifconfig $(route get default | grep interface | awk '{print $2}') | awk '/inet / {print $2; exit}'
```

On my machine, this prints `10.1.0.100`, which is a local LAN IP that other hosts on my network can reach.

On this machine, you can also set up a file share for models, outputs and inputs.

Once you have installed this Python package following the installation steps, you can start a worker using:

**Starting a Worker:**

```shell
# you must replace the IP address with the one you printed above
comfyui-worker --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100"
```

All the normal command line arguments are supported. This means you can use `--cwd` to point to a file share containing the `models/` directory:

```shell
comfyui-worker --cwd //10.1.0.100/shared/workspace --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100"
```

**Starting a Frontend:**

```shell
comfyui --listen --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100" --distributed-queue-frontend
```

However, the frontend will **not** be able to find the output images or models to show the client by default. You must specify a place where the frontend can find the **same** outputs and models that are available to the backends:

```shell
comfyui --cwd //10.1.0.100/shared/workspace --listen --distributed-queue-connection-uri="amqp://guest:guest@10.1.0.100" --distributed-queue-frontend
```

You can carefully mount network directories into `outputs/` and `inputs/` such that they are shared among workers and frontends; you can store the `models/` on each machine, or serve them over a file share too.

### Operating

The frontend expects to find the referenced output images in its `--output-directory` or in the default `outputs/` under `--cwd` (aka the "workspace").

This means that workers and frontends do **not** have to have the same argument to `--cwd`. The paths that are passed to the **frontend**, such as the `inputs/` and `outputs/` directories, must have the **same contents** as the paths passed as those directories to the workers.

Since reading models like large checkpoints over the network can be slow, you can use `--extra-model-paths-config` to specify additional model paths. Or, you can use `--cwd some/path`, where `some/path` is a local directory, and, and mount `some/path/outputs` to a network directory.

Known models listed in [**model_downloader.py**](./comfy/model_downloader.py) are downloaded using `huggingface_hub` with the default `cache_dir`. This means you can mount a read-write-many volume, like an SMB share, into the default cache directory. Read more about this [here](https://huggingface.co/docs/huggingface_hub/en/guides/download).

# Containers

Build the `Dockerfile`:

```shell
docker build . -t hiddenswitch/comfyui
```

To run:

```shell
docker run -it -v ./output:/workspace/output -v ./models:/workspace/models --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm hiddenswitch/comfyui
```

## Frontend Development

As of August 15, 2024, we have transitioned to a new frontend, which is now hosted in a separate repository: [ComfyUI Frontend](https://github.com/Comfy-Org/ComfyUI_frontend). This repository now hosts the compiled JS (from TS/Vue) under the `web/` directory.

### Reporting Issues and Requesting Features

For any bugs, issues, or feature requests related to the frontend, please use the [ComfyUI Frontend repository](https://github.com/Comfy-Org/ComfyUI_frontend). This will help us manage and address frontend-specific concerns more efficiently.

### Using the Latest Frontend

The new frontend is now the default for ComfyUI. However, please note:

1. The frontend in the main ComfyUI repository is updated fortnightly.
2. Daily releases are available in the separate frontend repository.

To use the most up-to-date frontend version:

1. For the latest daily release, launch ComfyUI with this command line argument:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@latest
   ```

2. For a specific version, replace `latest` with the desired version number:

   ```
   --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
   ```

This approach allows you to easily switch between the stable fortnightly release and the cutting-edge daily updates, or even specific versions for testing purposes.

### Accessing the Legacy Frontend

If you need to use the legacy frontend for any reason, you can access it using the following command line argument:

```
--front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest
```

This will use a snapshot of the legacy frontend preserved in the [ComfyUI Legacy Frontend repository](https://github.com/Comfy-Org/ComfyUI_legacy_frontend).

## Community

[Discord](https://comfy.org/discord): Try the #help or #feedback channels.

## Known Issues

Please visit the Issues tab for documented known issues.
