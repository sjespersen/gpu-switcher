# Local AI on your own computer

For AI on the machine you work on: no server and no network setup. Everything listens on
127.0.0.1 and starts with one command. The chat model starts when a request names it and stops
after `IDLE_MINUTES` without requests, so it isn't holding your GPU or RAM while you do
something else. If ComfyUI is set up, the page and the Studio switch the GPU over to it.

```sh
git clone https://github.com/sjespersen/gpu-switcher && cd gpu-switcher
local/local-ai up      # the first time: writes ~/.config/gpu-switcher/local.env, check it
local/local-ai up      # starts it
local/local-ai key     # the API key: the page's login, the API's Bearer token
```

Then:

- `http://127.0.0.1:8005`: the page (which model has the GPU; switch, stop)
- `http://127.0.0.1:8005/v1`: the OpenAI-compatible API, for your editor, agent or chat app.
  Use model `qwen3.8-27b` (or your `CHAT_NAME`) and the key as the API key ([Hermes](#hermes)).
- `http://127.0.0.1:8008`: the [Studio](https://github.com/sjespersen/studio), if it's on

`local-ai status` shows what runs. `local-ai down` stops everything. After changing
`local.env`, run `local-ai up` again.

## Linux with an NVIDIA GPU

Runs in Docker ([`compose.yaml`](compose.yaml)). You need:

- Docker, with your user in the `docker` group;
- the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  with its [CDI spec generated](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
  (`nvidia-ctk cdi list` shows `nvidia.com/gpu=all`).

Installing those needs sudo once; `local-ai` itself doesn't. What it runs:

- the switcher;
- llama.cpp's CUDA server for the chat model;
- ComfyUI, built on your computer from a pinned release (`COMFYUI=yes`, the default; the
  first build downloads PyTorch, ~5 GB);
- the Studio (`STUDIO=yes`). For the Studio, `local-ai up` also puts its two ComfyUI node
  packs in place and fetches MiniMax's H3 prompt guide.

The models go in `MODELS_DIR`: `llama.cpp/` (downloaded by llama.cpp on the first start) and
`comfyui/` (ComfyUI's `models` folder: `checkpoints/`, `loras/`, `diffusion_models/` …).
ComfyUI's own page is at `http://127.0.0.1:8188` while it has the GPU.

## macOS (Apple Silicon)

Runs natively: Docker on a Mac can't use the GPU. You need Homebrew's `llama.cpp` (Metal) and
Python 3.11 or newer:

```sh
brew install llama.cpp python
```

`local-ai up` writes the services file and runs the switcher as a launchd user agent: no
root, it starts again after a restart, and `local-ai down` removes it. The chat model runs on
`127.0.0.1:18080` (`CHAT_PORT`). Its log is `switcher.log`, the models' logs `<id>.log`, all in
`STATE_DIR`.

**ComfyUI** (optional): set `COMFYUI_DIR` to your ComfyUI checkout. The switcher then starts and
stops it with the Python in its `.venv` (on `COMFYUI_PORT`, 8188).
[ComfyUI's install guide](https://github.com/comfyanonymous/ComfyUI#apple-mac-silicon) covers
PyTorch with MPS.

**Studio** (optional): set `STUDIO_DIR` to a checkout of
[studio](https://github.com/sjespersen/studio) (needs `COMFYUI_DIR`). Expect the stills to
work, slowly; MiniMax H3 video at the Studio's sizes needs an NVIDIA card with 16 GB.

**How big a model?** The default chat model (Qwen3.8-27B at 3 bits) needs about 16 GB for
itself plus its context. On a 16 GB Mac, pick a smaller GGUF (`CHAT_REPO`, `CHAT_FILE`; and
set `CHAT_NAME` to what you want to call it).

## Hermes

[Hermes Agent](https://github.com/NousResearch/hermes-agent) on the same computer, as a named
custom provider in `~/.hermes/config.yaml`. `key_cmd` reads the key from `local-ai`, so it
isn't copied anywhere (use the path of your checkout):

```yaml
model:
  default: qwen3.8-27b
  provider: local-ai
providers:
  local-ai:
    name: local-ai
    base_url: http://127.0.0.1:8005/v1
    key_cmd: ~/gpu-switcher/local/local-ai key
    models:
      qwen3.8-27b: {}
    # Hermes sends no thinking level to custom endpoints, and Qwen 3.8 then thinks at the
    # highest; llama.cpp takes it as a chat template argument.
    extra_body:
      chat_template_kwargs:
        reasoning_effort: medium
```

The first request starts the chat model, so it takes as long as loading it. Everything listens
on 127.0.0.1 only: a Hermes on another machine (or in a container without host networking)
can't reach it; for that, run the switcher on a server.

## Settings (`~/.config/gpu-switcher/local.env`)

| Setting | Default | What |
|---|---|---|
| `API_KEY` | random | The page's login and the API's key |
| `PORT`, `STUDIO_PORT` | 8005, 8008 | The page and API; the Studio |
| `IDLE_MINUTES` | 30 | Stop the chat model after this long without requests; 0: never |
| `CHAT_REPO`, `CHAT_FILE` | Qwen3.8-27B, UD-Q3_K_XL | The chat model, a GGUF file on Hugging Face |
| `CHAT_NAME`, `CHAT_TITLE`, `CHAT_CONTEXT` | `qwen3.8-27b`, …, 32768 | The model name API requests use, the page's title, the context |
| `STATE_DIR` | `~/.local/share/gpu-switcher` | The switcher's state, the Studio's projects, logs |
| `MODELS_DIR`, `MEDIA_DIR` | `~/ai-models`, `~/ai-media` | Models; renders |
| `COMFYUI`, `STUDIO` | yes, no | Linux: ComfyUI and the Studio in Docker |
| `COMFYUI_DIR`, `STUDIO_DIR` | | macOS: your ComfyUI and Studio checkouts |
| `CHAT_PORT`, `COMFYUI_PORT` | 18080, 8188 | macOS: where the models listen |

The file holds `KEY=value` lines, like a compose `.env` file. It isn't run as shell code.
