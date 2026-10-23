# ComfyUI at a pinned release, on PyTorch's CUDA image (torch, torchvision and torchaudio come
# with it, so pip leaves them alone). Built on your computer by `local-ai up`. The image keeps
# torch in the system Python, which Ubuntu marks as externally managed: in a container with
# nothing else on that Python, --break-system-packages is how it's meant to be extended.
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime@sha256:3c0b174a1339193a5bcc4b37d737c9f5ca3c4782a2293d3888e4271367eddf44
ARG COMFYUI_COMMIT=63be2f9b82ad0dac88a7ceaa28a9cf3154125a18
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
 && git clone -q https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
 && git -C /opt/ComfyUI -c advice.detachedHead=false checkout -q "$COMFYUI_COMMIT" \
 && pip install --no-cache-dir --break-system-packages -r /opt/ComfyUI/requirements.txt
WORKDIR /opt/ComfyUI
ENV HOME=/tmp
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
