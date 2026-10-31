# gpu-switcher

GPUs, and more models than fit on them: gpu-switcher gives each card to one service at a time.
It has an OpenAI-compatible API that starts whichever model a request names (stopping what
runs on that card), modes that put several services on the cards at once (a chat model on one,
images on the other), stops a chat model after a while without requests, and a page to see what
runs and switch by hand: GPU memory, load, temperatures, power, disks.

It came out of [homelab-agent](https://github.com/sjespersen/homelab-agent), where it runs a
GPU box with llama.cpp, Strata, SwarmUI and Ollama. It works on its own: with Docker
containers, or without Docker by starting the model servers itself (e.g. on a Mac).

Only the Python standard library, one process, no database.

**On your own computer** (Linux with an NVIDIA GPU, or a Mac): `local/local-ai up` sets it up
with a chat model, ComfyUI and optionally the [Studio](https://github.com/sjespersen/studio),
all on 127.0.0.1. See [`local/README.md`](local/README.md).

## How it works

- **Services** are the things that take turns on the GPU: chat models (llama.cpp's
  `llama-server`, vLLM, Strata …), image tools (ComfyUI, SwarmUI) and at most one Ollama.
- **Cards**: each service says which cards it uses (`homelab.ai.gpus`; without it, all of
  them) and how much RAM it takes (`homelab.ai.ram`). Switching to a service stops only what
  runs on one of its cards (or in its `homelab.ai.group`), then chat models, biggest first,
  while the RAM wouldn't do (`RAM_GB`). The page's buttons switch; so does an API request that
  names another chat model.
- **Modes**: named sets of services that run together, one per card (`SCENARIOS_FILE`). Picking
  a mode starts its services and stops everything else. The page has an editor for them: it
  refuses two services on one card or in one group, and more RAM than `RAM_GB`.
- **The API** (`/v1/...`) passes each request on to the chat model it names, after starting
  it if it has to. `/v1/models` lists them all, running or not. A request waits while the
  model loads (up to an hour, for a first download), and the answer streams as it comes.
  One model can have several services (placements: both cards, or one card beside images) with
  the same `homelab.ai.model`: a request goes to the one that runs; else to one that doesn't
  need the card of a running image tool or speech engine; else to the first (by `homelab.ai.order`). Requests to
  what has to stop for it are answered first.
- **Change**: each running service's card has a Change button listing what can take its place
  on its cards without touching anything else (e.g. in a mode, another model on the 3060 while
  SwarmUI keeps the 5060 Ti).
- **Model stats**: for each running chat model its card shows what it does (loading, prefill
  until the first token, then decode), how fast it writes now and wrote the last answer, the
  last prompt's prefill speed and how much of its context window (`homelab.ai.context`) the
  last request filled. The switcher measures this on the requests it passes on, and uses the
  engine's own numbers when the answer has them (llama.cpp's `timings`, Ollama's durations).
- **Idle stop**: after `IDLE_MINUTES` without requests or GPU load, the chat model stops (only
  it: an image tool on the other card stays); the next request starts it again.
- **Image tools don't get pushed aside by the API**: a chat request that would need a card
  ComfyUI runs on gets an error until you switch on the page; one that fits beside it runs. Another app can also *hold* the GPU
  (`POST /_ai/api/hold`), and the page then won't switch.
- **Ollama** (optional) runs all the time and only takes the GPU while it has models loaded:
  switching unloads them, and a request that runs an Ollama model stops the others first.
  Clients use it through `/ollama/...` on the switcher.

## With Docker

The switcher finds the services by their container labels, and starts and stops them through
a Docker socket proxy that allows nothing else. A minimal `compose.yaml`:

```yaml
services:
  qwen:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    profiles: [qwen]          # compose up doesn't start it; the switcher does
    command: ["-hf", "unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL", "--host", "0.0.0.0", "--port", "8080", "-ngl", "99"]
    labels:
      homelab.ai.title: Qwen3.8-27B
      homelab.ai.about: "Chat on llama.cpp"
      homelab.ai.kind: chat
      homelab.ai.model: qwen3.8-27b   # the name API requests use
      homelab.ai.order: "1"
    healthcheck:
      test: ["CMD", "curl", "-fs", "http://localhost:8080/health"]
    deploy:
      resources:
        reservations:
          devices: [{driver: cdi, device_ids: [nvidia.com/gpu=all], capabilities: [gpu]}]

  switcher:
    image: ghcr.io/sjespersen/gpu-switcher:latest
    environment:
      API_KEY: ${API_KEY}       # the page's login and the API's Bearer token
      DOCKER_PROXY: socket-proxy:2375
      IDLE_MINUTES: "30"
    volumes:
      - ./data:/state
    ports:
      - "127.0.0.1:8005:8079"
    deploy:
      resources:
        reservations:   # only for nvidia-smi, the page's GPU gauge
          devices: [{driver: cdi, device_ids: [nvidia.com/gpu=all], capabilities: [gpu]}]

  socket-proxy:
    image: lscr.io/linuxserver/socket-proxy:latest
    environment: {CONTAINERS: "1", ALLOW_START: "1", ALLOW_STOP: "1", POST: "0"}
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    read_only: true
    tmpfs: [/run]
```

The labels:

| Label | What |
|---|---|
| `homelab.ai.title` | Name on the page; the switcher manages only containers that have it |
| `homelab.ai.about` | One line under the name |
| `homelab.ai.kind` | `chat` (default), `image`, or `audio`: a speech engine (TTS) with the same API routing as a chat model, but left out of `/v1/models` and started only by a request that names it |
| `homelab.ai.model` | The model names API requests use, comma-separated (the first is listed in `/v1/models`) |
| `homelab.ai.port` | Its API port in the compose network (default 8080) |
| `homelab.ai.order` | Position on the page |
| `homelab.ai.site` | A link for its Open button (a key in `LINKS_FILE`) |
| `homelab.ai.ui` | The port of its own web UI in the compose network: its Open button goes to `/_ai/ui/<id>/`, which the switcher passes through (WebSockets too) behind its login; the UI sees the prefix in `X-Forwarded-Prefix` and should use relative links |
| `homelab.ai.files` | Its model folder under `MODELS_DIR`, dropped from the page cache when another service starts |
| `homelab.ai.always-on` | `true` for Ollama (see above) |
| `homelab.ai.variant-of` | Another service's id: another model on the same engine (e.g. a fine-tune). No card of its own on the page; API requests naming its model start it |
| `homelab.ai.gpus` | The cards it uses, by nvidia-smi's number: `0`, `1`, `0,1`. Empty: all of them. Give the container the same cards (CDI `nvidia.com/gpu=1`) |
| `homelab.ai.ram` | The RAM it takes, in GB, for `RAM_GB` |
| `homelab.ai.group` | Services with the same group never run together (e.g. two placements of one app sharing a data folder) |
| `homelab.ai.context` | Its context window in tokens: the page shows how much of it the last request filled |

(The `homelab.` prefix is from where it came from; it stays so existing stacks keep working.)

## Without Docker

Point `SERVICES_FILE` at a JSON file with the services as commands, and the switcher runs
them itself. Each service gets its own process group, so stopping it also stops whatever it
started, and its output goes to `<id>.log` next to `STATE_FILE`. They stop when the switcher
stops. A service whose port already answers before it starts is refused, so another program on
that port can't pass for it. See [`services.example.json`](services.example.json):

```sh
API_KEY=$(openssl rand -hex 24) SERVICES_FILE=services.json STATE_FILE=~/.local/state/gpu-switcher/active \
  LISTEN=127.0.0.1 PORT=8005 python3 app.py
```

Each entry has the label settings as keys (`title`, `about`, `kind`, `models` as a list,
`order`, `site`, `ui`, `files`, `always_on`, `variant_of`, `gpus`, `ram`, `group`), and also:

| Key | What |
|---|---|
| `id` | Its name in `STATE_FILE`, the page and the API |
| `command` | What to run, as a list; `~` is expanded |
| `cwd`, `env` | Optional: where to run it, settings on top of the switcher's own |
| `port`, `host` | Where its API answers (default `127.0.0.1:8080`) |
| `health` | A path that answers 200 once it's ready (e.g. `/health`); without one, an open port counts as ready |

## Settings

| Setting | Default | What |
|---|---|---|
| `API_KEY` | (required) | Login for the page, Bearer token (or `x-api-key`) for the API |
| `STATE_FILE` | `/state/active` | Which service was picked last; its folder also holds the other state files |
| `LISTEN`, `PORT` | `0.0.0.0`, `8079` | Where the page and the API listen. Use `127.0.0.1` outside a container |
| `IDLE_MINUTES` | `0` (never) | Stop the chat model after this long without requests |
| `SCENARIOS_FILE` | `scenarios.json` next to `STATE_FILE` | The modes, `{"scenarios": [{"id", "title", "about", "services": [ids]}]}`; the page's editor writes it. Without it, the `scenarios` in `SERVICES_FILE` |
| `RAM_GB` | `0` (not checked) | The RAM the services may take together (their `homelab.ai.ram`) |
| `DOCKER_PROXY` | `socket-proxy:2375` | The Docker socket proxy (with Docker) |
| `PROJECT` | | With Docker: only this compose project's containers (when other stacks on the host have the labels too) |
| `SERVICES_FILE` | | The services as commands (without Docker) |
| `OLLAMA_URL` | `http://ollama:11434` | Where Ollama answers, if there is one |
| `MODELS_DIR`, `MEDIA_DIR` | `/models`, `/media` | Disks shown on the page; models dropped from the page cache |
| `API_LOG_DIR` | | Log every request that runs a model and its answer to `api.jsonl` here (one JSON object per line, rotated at 10 MB, 5 old files kept). Streamed answers are put back together; pictures and embeddings are left out |
| `NAME`, `LINKS_FILE` | | The page's title, and a JSON list of other links to show |
| `POWER` | | `yes`: suspend/restart/shut down buttons. They only write `power` next to `STATE_FILE`; something on the host has to act on it (homelab-agent's `homelab-power.sh`) |
| `LEDS`, `LEDS_EFFECT`, `LEDS_SPEED` | | `yes`: a case-lighting card, for an OpenRGB service on the host that reads the `leds` file (homelab-agent's `homelab-leds.sh`) |
| `CLASSIFIER_SHIM` | | `allow`: answer Claude Code's auto-mode safety-classifier calls here instead of asking the model (see below) |

## Claude Code auto-mode classifier

Claude Code's "auto mode" sends a safety-classifier request to the chat model before each tool
call — a system prompt beginning "You are a security monitor for autonomous AI coding agents",
with a terse verdict (`<block>no</block>` allows, `<block>yes</block>…` blocks). A local chat
model is slow and seldom hits that exact format, so the verdict times out or parse-fails, and
Claude Code falls back to a model the account's policy forbids — a dead end that stops the
session.

With `CLASSIFIER_SHIM=allow` the switcher recognises these requests (by that system prompt) and
answers them itself — instantly, without the GPU, always allowing. Ordinary `/v1` chat requests
are untouched. This turns auto mode's tool review into a rubber stamp, so use it only where you
would otherwise run the agent without that review (your own box, your own commands).

## API

- `GET /v1/models`: the chat models.
- `POST /v1/chat/completions` (and everything else under `/v1`): passed on to the model the
  body's `model` names (also matches longer names starting with it), else the one running,
  else the last used.
- `/ollama/...`: Ollama's own API, for Open WebUI and the like. Needs no login, so don't
  expose it beyond the machine or the compose network.
- `GET /_ai/api/status`, `POST /_ai/api/switch {"service": "<id>" | "none"}` (or
  `{"scenario": "<id>"}` for a mode, `{"stop": "<id>"}` for just that one),
  `POST /_ai/api/scenarios {"scenarios": [...]}` (the mode editor; JSON only),
  `POST /_ai/api/hold {"by": "<app>", "seconds": 90}`: what the page uses. The status has
  `gpus` (each card) and `gpu` (all cards as one: memory and power added up, the busiest load).
- `GET /_ai/auth`: 204 with a valid login, for a reverse proxy's forward auth.

## Security

- Everything but `/ollama` and the login page needs `API_KEY`. The page's login is a cookie
  derived from the key; the API takes only the key itself, so other sites can't use the
  cookie to call it.
- A service's web UI (`homelab.ai.ui`) gets the page's login in front of it; the login cookie
  and any Authorization header stay with the switcher, and it's `SameSite=Lax`, so other sites
  can't post to the UI with it.
- With Docker it only gets a socket proxy that can list, start and stop containers: no
  creating containers, running commands in them or reading their files.
- Listen on `127.0.0.1` unless something in front of it (Caddy, Tailscale) needs more.
- The power buttons are off unless `POWER=yes`, and even then only write a word to a file.
- The mode editor only takes JSON (`Content-Type: application/json`): a form on another site
  can't send that without the browser asking first, so the login cookie alone can't change the
  modes. It checks every mode (known services, no shared card, the RAM) and writes the file
  whole (a temporary file, then a rename).
- `API_LOG_DIR` keeps the prompts and answers of every API request (also Open WebUI's chats
  through `/ollama`) in plain text: keep that folder private, and off by default.

## License

MIT
