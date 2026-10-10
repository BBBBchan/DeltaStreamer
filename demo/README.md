# DeltaStreamer Interactive Demo

Upload a video, let DeltaStreamer observe changes and organize its memory, then ask a question. The interface displays the generated observations, memory, plan, tool activity and answer. This deployment makes new model requests for your uploaded video. It does not use the promotional demo's recorded trajectories or predetermined answers.

**[中文部署说明](README.zh-CN.md) · [Main repository](../README.md)**

There are two deployment options. **Local inference** runs Qwen3.6-27B or a smaller Qwen3.5 vision model on your GPU server and keeps model requests on that server. **Remote inference** connects the same application to an OpenAI-compatible vision API; it needs no local model weights or GPU. Choose one model backend, then install and start the web application.

## Local model deployment

Choose one model. For a smaller download and single-GPU setup, go directly to [the 2B / 4B / 9B examples](#3-optional-use-a-smaller-vision-model); you do not need to download 27B first.

### 1. Download Qwen3.6-27B

Run these commands from the repository root. Use Linux or a compatible WSL installation for the local vLLM server, Python 3.10–3.13, and a compatible NVIDIA driver. Keep the model server in a separate environment from the web application and the benchmark runtime.

```bash
python3 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install --upgrade pip
python -m pip install "huggingface_hub>=0.34" uv

# Optional: inspect the download size before transferring weights.
hf download Qwen/Qwen3.6-27B --dry-run

hf download Qwen/Qwen3.6-27B --local-dir models/Qwen3.6-27B
```

The official checkpoint includes the vision encoder and tokenizer. Download the complete repository; a text-only checkpoint cannot compare video frames. Use `hf auth login` if your Hugging Face account needs authentication. For a repeatable installation, add `--revision <model-commit-hash>` to the download command and record that revision. See the [official model card](https://huggingface.co/Qwen/Qwen3.6-27B) and [Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download).

### 2. Start the local vision API

Install vLLM in the **model** environment. The pinned version below matches the inference dependency version in this repository. Let vLLM resolve its compatible PyTorch installation rather than combining it with a separate benchmark environment.

```bash
uv pip install "vllm==0.19.1" --torch-backend=auto

CUDA_VISIBLE_DEVICES=0,1 vllm serve models/Qwen3.6-27B \
  --served-model-name DeltaStreamer-VLM \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":8}' \
  --reasoning-parser qwen3
```

This command illustrates two GPUs; set `CUDA_VISIBLE_DEVICES` and `--tensor-parallel-size` for your machine. A 27B model needs substantial GPU memory for weights, visual processing and the KV cache. The command is not a guarantee that a particular GPU will fit. A 32K context and one concurrent sequence are deployment choices for this demo, not the checkpoint's maximum context or the paper's evaluation configuration. Do not add `--language-model-only`, which disables visual input. See [vLLM installation](https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/) and [multimodal serving](https://docs.vllm.ai/en/v0.19.1/features/multimodal_inputs/) for platform-specific requirements.

Keep this terminal open. In a second terminal, check that the model name is available:

```bash
curl --fail http://127.0.0.1:8000/v1/models
```

The output should contain `DeltaStreamer-VLM`. When the web application runs on another machine, point it to the model server's reachable address; `127.0.0.1` always means the machine running that process. Use a private network or an authenticated tunnel for a separate server.

### 3. Optional: use a smaller vision model

For a lighter local deployment, choose one of these official post-trained checkpoints. All three include a vision encoder and can accept the demo's frame inputs. **4B is a useful starting configuration**; 2B reduces model size further, while 9B provides a larger alternative to try. These are configuration examples; the smaller checkpoints have not yet been validated end-to-end with DeltaStreamer, and their accuracy and latency are not the paper's reported results.

| Model | Size | Local directory |
| --- | --- | --- |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2B | `models/Qwen3.5-2B` |
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B | `models/Qwen3.5-4B` |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9B | `models/Qwen3.5-9B` |

From the repository root, create the model environment if you have not already done so, and download **one** checkpoint. For 4B:

```bash
python3 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install --upgrade pip
python -m pip install "huggingface_hub>=0.34" uv
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
uv pip install "vllm==0.19.1" --torch-backend=auto

CUDA_VISIBLE_DEVICES=0 vllm serve models/Qwen3.5-4B \
  --served-model-name DeltaStreamer-VLM \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":8}' \
  --reasoning-parser qwen3
```

For 2B or 9B, use the corresponding download command below, then replace `models/Qwen3.5-4B` in the serving command with that checkpoint's directory:

```bash
# Alternative 1: 2B
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B

# Alternative 2: 9B
hf download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
```

Stop the previous model server before replacing it on port 8000. Keep `--served-model-name DeltaStreamer-VLM`, so the web application's model name remains the same. After installing the web application below, use these settings in `demo/.env` for any of the three checkpoints:

```dotenv
DEMO_BACKEND=local
DEMO_API_BASE_URL=http://127.0.0.1:8000/v1
DEMO_MODEL=DeltaStreamer-VLM
DEMO_API_KEY=
DEMO_EXTRA_BODY_JSON={"chat_template_kwargs":{"enable_thinking":false}}
```

The explicit `enable_thinking:false` requests direct structured responses. Qwen3.5-4B and 9B otherwise enable thinking by default; 2B defaults to non-thinking. These single-GPU commands use a 32K context, one concurrent sequence and up to eight images per request. Actual memory requirements also include visual processing and the context cache, so no particular GPU capacity is guaranteed. If memory is insufficient, try 2B, reduce the context limit, or distribute the model across compatible GPUs. Then run the UI's connection test and a short uploaded video to check your chosen configuration.

## Remote vision API

Skip the model download and vLLM installation when using a remote provider. Obtain an API endpoint, model ID and API key from that provider. The endpoint must support OpenAI-compatible `POST /chat/completions` requests with multiple `image_url` items containing JPEG data URLs. Text-only models and APIs accepting only public image URLs are insufficient for this application.

The application sends sampled frames and relevant textual context to the configured endpoint. Original video files are stored by the demo server. Use footage you are permitted to share with your chosen provider. API credentials stay in the backend environment and are not requested in the browser.

## Install the web application

Open a new terminal at the repository root. Use **Python 3.10–3.12** for this environment to match its pinned NumPy dependency. The demo uses the repository's agent and video processor through an HTTP model adapter. It does not load model weights in the web process, but its shared processor imports PyTorch, so install the **CPU** build in this environment.

```bash
python3 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r demo/requirements.txt
```

Install FFmpeg and ffprobe on the web server; both must be on `PATH`.

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y ffmpeg

ffmpeg -version
ffprobe -version
```

For macOS, use `brew install ffmpeg` and the platform's PyTorch wheel (`python -m pip install "torch==2.10.0"`), then connect to a remote vision API or a model server running on a supported GPU host. FFmpeg must include the `libx264` and AAC encoders used to generate browser-compatible previews. It also extracts native-resolution frames at whole-second timestamps for inference. The demo does not need the root `requirements.txt`, vLLM, ms-swift, benchmark datasets, SAM2 or Whisper checkpoints in its web environment.

### Configure a backend

Copy the example configuration, then edit `demo/.env`. The server loads this file automatically; process environment variables take precedence.

```bash
cp demo/.env.example demo/.env
```

For the local vLLM server above:

```dotenv
DEMO_BACKEND=local
DEMO_API_BASE_URL=http://127.0.0.1:8000/v1
DEMO_MODEL=DeltaStreamer-VLM
DEMO_API_KEY=
DEMO_EXTRA_BODY_JSON={"chat_template_kwargs":{"enable_thinking":false}}
```

If you started vLLM with an API key, put the same key in `DEMO_API_KEY`. The extra body requests Qwen's direct response mode; it avoids spending the generation budget on thinking before the structured observation or answer. This is the [Qwen API setting](https://huggingface.co/Qwen/Qwen3.6-27B#instruct-or-non-thinking-mode), not a change to the model weights.

For a remote endpoint, replace the placeholders with your provider's values:

```bash
cp demo/.env.remote.example demo/.env
```

```dotenv
DEMO_BACKEND=remote
DEMO_API_BASE_URL=https://your-provider.example/v1
DEMO_MODEL=your-vision-model-id
DEMO_API_KEY=your-api-key
DEMO_EXTRA_BODY_JSON={}
```

The base URL includes the API prefix, such as `/v1`; do not append `/chat/completions`. Provider-specific options belong in `DEMO_EXTRA_BODY_JSON`. Only use options supported by that endpoint. For example, some Qwen services accept `{"enable_thinking":false}` instead of vLLM's nested `chat_template_kwargs`. Do not commit `.env` or share it with exported logs.

### Start the interface

```bash
python -m demo.server --host 127.0.0.1 --port 7861
```

Open **http://127.0.0.1:7861**. The model API uses port **8000** and the demo uses **7861**; they are separate services. The equivalent ASGI command is:

```bash
uvicorn demo.server:app --host 127.0.0.1 --port 7861
```

Use one web worker so active jobs remain with the process tracking them. This is a local research interface, with no built-in user authentication. For access from another machine, an SSH tunnel keeps the default loopback binding:

```bash
ssh -L 7861:127.0.0.1:7861 your-user@your-demo-server
```

Then open `http://127.0.0.1:7861` on your own machine. To expose the application on a network, put authentication, HTTPS and upload limits at a reverse proxy rather than directly publishing the development server.

## Upload, observe and ask

1. Check the backend status. Use the connection test to send two small generated images to the selected model and check multimodal compatibility. This explicit test is a real model request and may be billed by a remote provider.
2. Drag a video into the upload area or browse for a file. Supported extensions are **MP4, MOV, MKV, WebM, AVI and M4V**, with content that FFmpeg can decode. Start with a short, clear clip of around 10–30 seconds. The default upload limit is **512 MiB** and the maximum duration is **300 seconds**.
3. Review the video and choose the query time. The initial query time is the end of the clip; choose an earlier player position to ask while later frames are still available. The processor aligns that time to the preceding sampled second and clamps it to the last available frame.
4. Enter a question and start inference. The agent first observes the available prefix without seeing the question. At the selected query time, it receives the question and plans its investigation; it may continue observing subsequent frames until it answers.
5. Inspect the generated Delta, Working memory, Episodes, Pins, plan, tool activity and answer. You can ask another question about the same upload; each question starts an independent run. One inference job runs at a time; finish or cancel it before starting another.

The model's processing progress can lag behind normal-speed video playback. The interface shows actual model results as they become available; it does not use a fixed response delay. Answer quality and latency depend on the model, video, chosen time and endpoint.

Cancel stops the local worker and further API requests. Whether a request already executing on the provider's server stops immediately depends on that API.

The demo invokes the existing agent and processor, including explicit visual changes, structured memory, planning, replay and crop. It samples observations at 1 fps. A demo-side replay adapter sends the requested historical image in the next model request, so successful replay provides visual evidence rather than only a tool acknowledgment. This adapter leaves the original `core/` source unchanged. SAM2 tracking and Whisper audio are disabled in this deployment and are not presented as successful tool results. Use the [benchmark instructions](../README.md#evaluation) for the complete evaluation setup. Changing the backend, tool availability or runtime configuration does not reproduce the paper's scores automatically.

## Configuration and saved runs

Optional settings in `demo/.env`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DEMO_API_TIMEOUT` | `180` | Timeout in seconds for one model request. |
| `DEMO_MAX_TOKENS` | `4096` | Maximum generated tokens per model call. |
| `DEMO_MAX_PIXELS` | `100352` | Per-image pixel budget for model requests; original stored video, frames and crop sources are retained at their original resolution. |
| `DEMO_MAX_UPLOAD_MB` | `512` | Upload size limit in MiB. |
| `DEMO_MAX_DURATION_SECONDS` | `300` | Maximum accepted video duration. |
| `DEMO_MAX_MODEL_CALLS` | `600` | Upper bound on model calls per run. |
| `DEMO_DATA_DIR` | `demo/.data` | Storage location for uploads and run artifacts. |

Restart the server after editing configuration. Longer videos can make many model calls, particularly during observation; increasing the limits also increases processing and storage requirements.

The default storage layout is:

```text
demo/.data/
├── videos/<video-id>/        # Uploaded media and processed video
└── runs/<run-id>/
    ├── status.json          # Progress and final status
    ├── trajectory.json      # Recorded agent trajectory
    ├── steps.jsonl          # Step-by-step records
    └── ...                 # Run logs and intermediate artifacts
```

These files are local and excluded from version control. Keep the source clip, model revision, backend settings and run artifacts together when comparing runs; remove API keys before sharing a configuration. Stop the server and remove only the unwanted run/upload folders when cleaning disk space. Restarting the process is not an automatic resume mechanism for in-flight inference.

## Verify your installation

Check `http://127.0.0.1:7861/api/health`, then use the UI's explicit connection test. A provider may omit `GET /models`; an unavailable model-list endpoint alone does not establish that vision requests will fail. Conversely, a model appearing in that list does not prove it accepts multiple images.

For an end-to-end check, upload a short clip with an unmistakable visual change, ask about that change, and inspect the run artifacts. Confirm that observations and deltas have source timestamps, that the plan is generated after the query, and that any claimed replay has corresponding historical image input. The question and answer should come from this run. This check validates your selected endpoint and video path; it is not a benchmark accuracy evaluation.

Developers can also run the isolated transport and agent smoke tests without downloading a model. These tests use a deterministic local HTTP fixture and verify that replay actually delivers a historical image; they do not measure model quality.

```bash
python -m pip install pytest
python -m pytest demo/tests -q
```

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ffmpeg` or `ffprobe` not found | Install both executables on the web server and restart it with the correct `PATH`. |
| Upload rejected | Check file size, duration and whether FFmpeg can decode the video. Increase the relevant limit only if you have capacity. |
| HTTP 401 or 403 from the model | Check `DEMO_API_KEY` and the provider account. Restart the web server after changing `.env`. |
| HTTP 404 or model not found | Match `DEMO_MODEL` to the served model ID and check the API base prefix. For the local command above, use `DeltaStreamer-VLM`. |
| API rejects `image_url` or multiple images | Use a compatible vision model. For vLLM, retain its vision encoder and set an adequate image limit. The example allows eight images. |
| Empty output or output ends before JSON is complete | For Qwen, configure the supported non-thinking option; also check token limits and the provider's response format. |
| CUDA out of memory | Change the local model deployment: reduce context/concurrency or distribute the model over more compatible GPUs. The CPU web environment does not solve model-server memory pressure. |
| API timeout or rate limit | Check model-server logs or provider limits. Increase `DEMO_API_TIMEOUT` for slow requests and begin with a shorter clip. |
| Connection refused | Start the model server, verify its host/port and remember that `localhost` is local to the web server, not the browser. |
| Video plays ahead of observations | Playback and inference have independent clocks. Wait for the model progress; playback speed is not a measurement of inference throughput. |

For inference-engine configuration, consult the [vLLM OpenAI-compatible server documentation](https://docs.vllm.ai/en/v0.19.1/serving/openai_compatible_server/).

## License

This demo follows the repository's [CC BY-NC 4.0 license](../LICENSE). Model weights and third-party dependencies retain their respective licenses.
