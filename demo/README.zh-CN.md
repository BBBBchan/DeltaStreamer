# DeltaStreamer 交互 Demo 部署

**[English](README.md) · [项目主页](../README.md)**

上传视频后，系统会调用视觉模型观察画面变化、组织记忆，再根据问题生成计划、检索证据并回答。界面展示此次运行产生的观察、记忆、计划和证据，**不是宣发视频中的预录轨迹或预设答案**。

可以选择本地模型或远程视觉 API。使用远程 API 时，不需要下载 Qwen3.6-27B，也不需要在网页服务器上安装 GPU 推理环境。

## 方案一：下载并部署本地模型

### 1. 下载完整模型

以下命令均从仓库根目录执行。本地 vLLM 推理服务需要 Linux（或兼容的 WSL 环境）、Python 3.10–3.13 和兼容的 NVIDIA 驱动。模型服务、网页服务与原有 benchmark 环境分别安装，避免依赖冲突。

```bash
python3 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install --upgrade pip
python -m pip install "huggingface_hub>=0.34" uv

# 可先查看待下载文件和体积。
hf download Qwen/Qwen3.6-27B --dry-run
hf download Qwen/Qwen3.6-27B --local-dir models/Qwen3.6-27B
```

需要下载完整的视觉模型，不能换成去掉视觉编码器的纯文本版本。如果下载需要登录，运行 `hf auth login`。为了固定模型版本，可以给下载命令增加 `--revision <模型提交哈希>` 并记录该值。模型与下载方式分别见 [Qwen 官方模型页](https://huggingface.co/Qwen/Qwen3.6-27B) 和 [Hugging Face 官方说明](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download)。

### 2. 启动模型 API

在上面的模型环境中安装并启动 vLLM。这里固定为仓库使用的 0.19.1，让安装程序选择与之兼容的 PyTorch，不要混入已有 benchmark 的依赖。

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

这是两张 GPU 的示例，根据机器调整可见设备与 `--tensor-parallel-size`。27B 模型的权重、视觉计算和 KV cache 都占用显存，不能只按参数量判断能否运行。32K 上下文和单并发是本 Demo 的部署设置，不是模型最大能力或论文的实验配置，也不保证任意显卡都能装下。不要加 `--language-model-only`，否则模型不能接收视频帧。平台安装要求见 [vLLM 官方文档](https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/)。

保持模型服务运行，在另一终端检查模型名称：

```bash
curl --fail http://127.0.0.1:8000/v1/models
```

输出应包含 `DeltaStreamer-VLM`。如果网页后端与模型服务不在同一机器，后端应配置可访问的模型服务器地址；`127.0.0.1` 始终指运行该进程的机器。跨机器连接建议使用内网或带认证的隧道。

## 方案二：使用远程视觉 API

跳过模型下载与 vLLM 安装，准备服务商提供的 API 地址、模型 ID 和 API key。接口需兼容 OpenAI 的 `POST /chat/completions`，支持在同一请求中接收多个 `image_url`，并接受 JPEG 的 base64 data URL。纯文本模型或只接受公网图片 URL 的接口不能直接用于此 Demo。

应用会把抽取的视频帧和相关文本发送到该接口，原视频保存在网页后端机器上。请选择允许向该服务商发送的视频。API key 只保存在后端环境，不在浏览器页面中填写或返回。

## 安装并启动网页服务

### 1. 安装依赖

另开一个终端，在仓库根目录用 **Python 3.10–3.12** 创建独立的网页环境，以兼容固定版本的 NumPy。网页进程通过 HTTP 请求模型，不加载模型权重；但复用的 agent/video processor 会导入 PyTorch，因此这里安装 **CPU 版 PyTorch**。

```bash
python3 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r demo/requirements.txt

# Ubuntu / Debian，安装后端使用的视频处理程序。
sudo apt-get update
sudo apt-get install -y ffmpeg

ffmpeg -version
ffprobe -version
```

macOS 可以用 `brew install ffmpeg`，PyTorch 使用平台对应的 wheel（`python -m pip install "torch==2.10.0"`），模型则连接远程 API 或另一台支持 GPU 推理的服务器。FFmpeg 需包含生成网页预览所用的 `libx264` 和 AAC 编码器；推理帧也由 FFmpeg 按整秒时间点抽取，保留原始分辨率。网页环境不需要安装仓库根目录的全部 `requirements.txt`、vLLM、ms-swift，也不需要 benchmark 数据、SAM2 或 Whisper 权重。

### 2. 配置模型连接

```bash
cp demo/.env.example demo/.env
```

编辑 `demo/.env`。后端自动读取该文件，已经在进程环境中设置的同名变量优先。

连接上面启动的本地 vLLM：

```dotenv
DEMO_BACKEND=local
DEMO_API_BASE_URL=http://127.0.0.1:8000/v1
DEMO_MODEL=DeltaStreamer-VLM
DEMO_API_KEY=
DEMO_EXTRA_BODY_JSON={"chat_template_kwargs":{"enable_thinking":false}}
```

如果 vLLM 配置了 API key，`DEMO_API_KEY` 填入同一个值。`DEMO_EXTRA_BODY_JSON` 使用 Qwen 的直接回答模式，避免输出预算在生成结构化观察或答案前就耗在 thinking 上。该参数用法见 [Qwen 官方说明](https://huggingface.co/Qwen/Qwen3.6-27B#instruct-or-non-thinking-mode)。

使用远程视觉 API：

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

将占位值换成服务商的实际信息。Base URL 包含 `/v1` 等接口前缀，末尾不要再写 `/chat/completions`。服务商的额外参数写入 `DEMO_EXTRA_BODY_JSON`，例如某些 Qwen 接口使用 `{"enable_thinking":false}` 而不是 vLLM 的嵌套字段；以对方接口文档为准。不要提交 `.env`，也不要把带 key 的配置放入分享的日志。

### 3. 启动

```bash
python -m demo.server --host 127.0.0.1 --port 7861
```

打开 **http://127.0.0.1:7861**。模型 API 的端口是 **8000**，网页端口是 **7861**。也可以用：

```bash
uvicorn demo.server:app --host 127.0.0.1 --port 7861
```

使用单个网页 worker，让当前任务与维护任务状态的进程保持一致。此界面面向本地研究使用，没有内置登录认证。如果网页部署在远程服务器，可在自己电脑上建立 SSH 隧道：

```bash
ssh -L 7861:127.0.0.1:7861 your-user@your-demo-server
```

然后在自己电脑打开 `http://127.0.0.1:7861`。需要正式提供网络访问时，在反向代理层配置认证、HTTPS 和上传限制。

## 上传视频与提问

1. 查看后端状态，点击连接检查。该操作会向模型发送两张程序生成的小图片，检查多图输入是否可用；使用远程 API 时，这次真实模型请求可能计费。
2. 将视频拖入上传区域，或点击浏览文件。支持 **MP4、MOV、MKV、WebM、AVI、M4V** 扩展名，文件内容需能被 FFmpeg 解码。首次建议用 10–30 秒、变化清楚的短片。默认限制为 **512 MiB**、最长 **300 秒**。
3. 查看视频并选择提问时刻。默认在视频末尾提问，也可以选择播放器的较早位置，让提问后仍有后续帧可观察。处理器把该时间向下对齐到采样的整秒，最晚不超过最后一个可用帧。
4. 输入问题并运行。Agent 先在不知道问题的情况下观察选定时间前的内容，到达提问时刻后接收问题、生成计划，并可继续观察直到给出答案。
5. 查看实际生成的 Delta、Working memory、Episodes、Pins、计划、工具调用和答案。同一视频可以多次提问，每次创建独立运行。同一时间只执行一个推理任务，结束或取消后才能开始下一个。

视频播放与模型处理是异步的，播放进度可能快于观察进度。界面在模型返回后展示结果，没有设置固定的“几秒后回答”。延迟和答案质量取决于视频、提问时刻、模型与服务端。

取消会停止本地 worker 和后续 API 请求。已经在服务商端执行的请求能否立即终止，取决于该 API。

此 Demo 复用现有 agent 和 processor，以 1 fps 观察，包含显式视觉变化、结构化记忆、规划、回看与裁剪。Demo 侧的回看适配器会把请求的历史图像送入下一次模型请求，让回看真正提供视觉证据；原有 `core/` 源文件保持不变。该部署默认关闭 SAM2 跟踪和 Whisper 音频工具，不会把未执行的工具展示为成功。完整评测配置见[主 README](../README.md#evaluation)；换用模型或改变工具及运行设置，不代表能直接复现论文分数。

## 配置与保存的结果

以下可选参数写入 `demo/.env`，修改后重启网页服务：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `DEMO_API_TIMEOUT` | `180` | 单次模型请求超时，秒。 |
| `DEMO_MAX_TOKENS` | `4096` | 单次生成的最大 token 数。 |
| `DEMO_MAX_PIXELS` | `100352` | 发送给模型的单张图片像素预算；保存的原视频、帧与裁剪源图仍保留原始分辨率。 |
| `DEMO_MAX_UPLOAD_MB` | `512` | 视频上传大小上限，MiB。 |
| `DEMO_MAX_DURATION_SECONDS` | `300` | 视频时长上限，秒。 |
| `DEMO_MAX_MODEL_CALLS` | `600` | 一次运行的模型调用上限。 |
| `DEMO_DATA_DIR` | `demo/.data` | 上传文件和运行结果目录。 |

视频越长，观察阶段的模型调用通常越多。提高上限也会增加计算、远程 API 用量和磁盘需求。

```text
demo/.data/
├── videos/<video-id>/        # 上传视频和处理后的视频
└── runs/<run-id>/
    ├── status.json          # 进度和最终状态
    ├── trajectory.json      # Agent 轨迹
    ├── steps.jsonl          # 逐步记录
    └── ...                 # 日志及其他中间文件
```

这些本地文件不纳入 Git。比较结果时保留原视频、模型版本、后端设置和运行记录；分享配置前移除 key。清理磁盘时先停止服务，再删除不需要的视频或运行目录。重启进程不会自动恢复尚未完成的推理。

## 检查与排错

打开 `http://127.0.0.1:7861/api/health` 检查服务，再用网页上的连接测试验证多图请求。部分供应商不实现 `GET /models`，不能仅凭模型列表失败判断视觉 API 不可用；反过来，能列出模型也不能证明支持多图输入。

端到端验证时，上传一个含明显变化的短片，询问变化内容，并查看运行记录。观察与 Delta 应对应源视频时间，计划应在问题到达后生成；如果出现回看，应有实际历史图像输入记录。该检查验证连接和处理链路，不替代 benchmark 的准确率评测。

开发者也可在不下载模型的情况下运行通信与 agent 冒烟测试。测试使用本地确定性的 HTTP fixture，并检查回看是否实际发送历史图像，不衡量模型回答质量。

```bash
python -m pip install pytest
python -m pytest demo/tests -q
```

| 现象 | 检查方法 |
| --- | --- |
| 找不到 FFmpeg/ffprobe | 确认两个程序均安装在网页服务器上，并在服务进程的 `PATH` 中。 |
| 视频被拒绝 | 检查大小、时长及 FFmpeg 是否能解码；确有资源余量时再提高对应上限。 |
| 模型返回 401/403 | 检查后端 `.env` 的 API key 和账户权限，修改后重启。 |
| 404 或找不到模型 | 核对 API 前缀和模型 ID；本地示例的名称为 `DeltaStreamer-VLM`。 |
| 不支持图片或多图 | 使用视觉模型，确保未关闭视觉编码器；vLLM 示例允许每次八张图。 |
| 输出为空或 JSON 被截断 | 检查模型的非 thinking 配置、生成预算和服务商返回格式。 |
| GPU 显存不足 | 在模型服务侧降低上下文/并发，或使用更多兼容 GPU；CPU 网页环境不会降低模型本身的显存需求。 |
| 超时或限流 | 查看模型日志或供应商限额，必要时调整超时，先用更短视频验证。 |
| 无法连接模型 | 确认模型服务启动、地址及端口可达；后端的 `localhost` 并不指用户浏览器所在机器。 |
| 视频已播放但观察仍在更新 | 两者进度独立，等待模型处理即可；不能以视频播放速度代表推理吞吐。 |

模型 API 的更多配置见 [vLLM OpenAI-compatible server 文档](https://docs.vllm.ai/en/v0.19.1/serving/openai_compatible_server/)。

## 许可

本 Demo 遵循仓库的 [CC BY-NC 4.0 协议](../LICENSE)。模型权重和第三方依赖各自遵循其原有协议。
