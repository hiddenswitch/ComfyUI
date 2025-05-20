FROM nvcr.io/nvidia/pytorch:24.12-py3

ARG TZ="America/Los_Angeles"

ENV PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND="noninteractive"
# mitigates
# RuntimeError: Failed to import transformers.generation.utils because of the following error (look up to see its traceback):
# numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
RUN echo "numpy<2" >> numpy-override.txt

# mitigates https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
# mitigates AttributeError: module 'cv2.dnn' has no attribute 'DictValue' \
# see https://github.com/facebookresearch/nougat/issues/40
RUN pip install uv && uv --version && \
    apt-get update && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 -y && \
    uv pip uninstall --system $(pip list --format=freeze | grep opencv) && \
    rm -rf /usr/local/lib/python3.12/dist-packages/cv2/ && \
    uv pip install wheel && \
    uv pip install --no-build-isolation opencv-python-headless && \
    uv pip install --overrides=numpy-override.txt "comfyui[attention]@git+https://github.com/hiddenswitch/ComfyUI.git" && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
# addresses https://github.com/pytorch/pytorch/issues/104801
# and issues reported by importing nodes_canny
RUN comfyui --quick-test-for-ci --cpu --cwd /workspace

EXPOSE 8188
CMD ["python", "-m", "comfy.cmd.main", "--listen"]
