# LLM Engineering Master - AI & Large Language Models

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue) ![PyTorch CUDA](https://img.shields.io/badge/PyTorch-2.12%2Bcu130-green) ![uv](https://img.shields.io/badge/managed%20by-uv-purple) ![Status Active](https://img.shields.io/badge/Status-Active-brightgreen)

## 📚 About This Project

This repository documents my learning journey through the **[Udemy LLM Engineering: Master AI and Large Language Models](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models)** course. It contains practical exercises, projects, and implementations covering the complete spectrum of LLM application development.

## 🎯 Learning Objectives

The course teaches:

- **LLM Fundamentals**: Transformer architectures, tokenization, and how large language models work
- **Prompt Engineering**: Techniques for crafting effective prompts and prompt templates
- **API Integration**: Working with OpenAI, Anthropic Claude, Google, and other LLM providers
- **LangChain & LlamaIndex**: Building production applications with orchestration frameworks
- **Tool Use & Function Calling**: Enabling LLMs to interact with external APIs and systems
- **RAG (Retrieval Augmented Generation)**: Combining LLMs with external knowledge bases
- **Local Models**: Running open-source models locally (Ollama, LLaMA, etc.)
- **Production Patterns**: Monitoring, evaluation, cost optimization, and deployment
- **Multi-modal AI**: Working with text, images, and other modalities

## 📂 Project Structure

```
LLM-Engineering-Exercise/
├── Exercise_1.1.py                  # Week 1: Basic LLM API calls
├── Exercise_1.2.py                  # Week 1: Prompt engineering
├── Exercise_1.5.py                  # Week 1: Token counting
├── Exercise_1.5v2.py                # Week 1: Cost estimation
├── Exercise_1.6.py                  # Week 1: Building applications
├── Exercise_2.1.py                  # Week 2: LangChain basics
├── Exercise_2.2.py                  # Week 2: Tool use & function calling
├── Exercise_2.3.py                  # Week 2: Multi-provider orchestration
├── Exercise_2.4.py                  # Week 2: Stateful chatbots
├── Exercise_2.5.py                  # Week 2: Error handling
├── Exercise_2.6.py                  # Week 2: KeyCraft chatbot (LangChain)
├── Exercise_3_1.py                  # 🚀 Week 3: SDXL on Local GPU
├── Exercise_3.2.py                  # Week 3: HuggingFace Pipelines (9 tasks + Gradio)
├── Exercise_3.3.py                  # Week 3: Tokenizers Explorer (Gradio)
├── Exercise_3.4.py                  # Week 3: Models Explorer — 4-bit quant (Gradio)
├── Exercise_3.5.py                  # Week 3: Meeting Minutes from Audio (Whisper + LLM, Gradio)
├── Week_3_Day_5_Meeting_Minutes_product.ipynb  # Source notebook for Exercise 3.5
├── outputs/                         # Generated images, audio, meeting minutes (.md)
├── samples/                         # Downloaded sample audio (denver_extract.mp3)
├── check_torch.py                   # Quick CUDA / torch sanity check
├── pyproject.toml                   # Project + dependency definition (uv)
├── uv.lock                          # Pinned dependency lockfile (uv)
├── requirements.txt                 # Legacy pip deps (Week 1-2) — superseded by pyproject
├── requirements-cuda.txt            # Legacy pip GPU deps — superseded by pyproject
├── README.md                        # This file
├── CLAUDE.md                        # Guidance for AI coding agents
├── .env.example                     # Environment variables template
└── .gitignore                       # Git ignore rules
```

> **Dependency management:** This project uses **[uv](https://docs.astral.sh/uv/)**. The `requirements*.txt` files are kept for reference only — `pyproject.toml` + `uv.lock` are the source of truth.

## 🚀 Quick Start

This project is managed with **[uv](https://docs.astral.sh/uv/)**. Install it first (`pip install uv` or see the uv docs), then:

```bash
# 1. Clone & enter repo
git clone https://github.com/khthana/LLM-Engineering-Exercise.git
cd LLM-Engineering-Exercise

# 2. Create the venv and install everything from pyproject.toml + uv.lock
#    (CUDA torch is pulled automatically via the pinned cu130 index)
uv sync

# 3. Set API keys
copy .env.example .env   # then edit .env with your actual keys

# 4. Run any exercise inside the uv-managed environment
uv run Exercise_3.4.py
```

> **CUDA note:** `pyproject.toml` pins the `torch` / `torchvision` / `torchaudio` stack to the
> **cu130** PyTorch index (see `[tool.uv.sources]`), so `uv sync` installs the GPU build instead
> of the default CPU wheels. `uv run check_torch.py` verifies CUDA is available.

## 📚 Exercises

> Run any exercise with **`uv run <file>`** (e.g. `uv run Exercise_1.1.py`). The `python <file>` commands below also work once the uv venv is activated.

### Week 1: LLM Fundamentals

---

#### Exercise 1.1 — Multi-Provider Website Summarizer

Scrapes a webpage with BeautifulSoup and summarizes the content using a chosen LLM provider. Supports OpenAI, Claude, Gemini, OpenRouter, and Ollama (local) from a single config.

**Key concepts:** API clients, provider abstraction, web scraping, prompt design

```bash
python Exercise_1.1.py
```

---

#### Exercise 1.2 — Multi-Provider LLM Manager (OOP)

Refactors Exercise 1.1 into a proper OOP `LLMManager` class. Sends the same prompt to multiple providers in parallel using `concurrent.futures` and compares responses side-by-side.

**Key concepts:** OOP design, parallel requests, dataclasses, prompt library

```bash
python Exercise_1.2.py
```

---

#### Exercise 1.5 — Company Brochure Generator

3-stage pipeline: (1) scrape all links from a company homepage, (2) LLM selects relevant pages (About, Careers, etc.) as JSON, (3) LLM generates a polished Markdown brochure with streaming output.

**Key concepts:** Chained LLM calls, structured output (JSON), streaming, multi-step pipelines

```bash
python Exercise_1.5.py
```

---

#### Exercise 1.5v2 — Company Brochure Generator (Multi-Provider)

Same pipeline as Exercise 1.5 but uses `LLMManager` from Exercise 1.2 instead of a hardcoded OpenAI client. Defaults to Ollama (local) — no API key required.

**Key concepts:** Reusable components, local inference with Ollama, provider swapping

```bash
python Exercise_1.5v2.py
```

---

#### Exercise 1.6 — Thai News Summarizer

Scrapes [thaipost.net](https://thaipost.net) across 3 levels (categories → article list → article content), filters to today's articles, groups by category, then uses an LLM to summarize each article with streaming output.

**Key concepts:** Deep scraping, date filtering, concurrent fetching, category grouping, LLM summarization

```bash
python Exercise_1.6.py
```

---

### Week 2: Production Techniques

---

#### Exercise 2.1 — Streaming Chat with Multiple Providers

Builds a streaming chat interface that works across OpenAI, Claude, Gemini, OpenRouter, and Ollama. Abstracts away provider differences so the same conversation loop works everywhere.

**Key concepts:** Streaming responses, provider abstraction, chat history, token/cost tracking

```bash
python Exercise_2.1.py
```

---

#### Exercise 2.2 — Unified LLM Interface with LiteLLM

Replaces manual provider wrappers with [LiteLLM](https://docs.litellm.ai/) — a single library that routes to 100+ models with one API. Demonstrates how production apps abstract away provider-specific SDKs.

**Key concepts:** LiteLLM, provider-agnostic code, drop-in replacement for OpenAI SDK

```bash
python Exercise_2.2.py
```

---

#### Exercise 2.3 — Multi-Provider Chat with LangChain

Replaces LiteLLM with LangChain's chat model wrappers (`ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`, `ChatOllama`). Sends the same messages to all providers in parallel and displays results.

**Key concepts:** LangChain chat models, `HumanMessage`/`SystemMessage`, parallel execution

```bash
python Exercise_2.3.py
```

---

#### Exercise 2.4 — Gradio Chat UI

Wraps the multi-provider LLM client in a Gradio web interface. Adds a model selector dropdown, reasoning effort control, dark mode, and streaming output directly in the browser.

**Key concepts:** Gradio `gr.Blocks`, streaming UI, model selector, dark mode CSS injection

```bash
python Exercise_2.4.py
```

---

#### Exercise 2.5 — KeyCraft Chatbot with Tool Use

Adds LangChain **tool calling** to the Gradio chatbot. The `KeyBot` AI assistant can call `list_keyboards()` and `list_mice()` tools to fetch live product data before responding — the LLM decides when and what to call.

**Key concepts:** `@tool` decorator, LangChain tool calling, `ToolMessage`, agentic loops, chatbot persona

```bash
python Exercise_2.5.py
```

---

#### Exercise 2.6 — KeyCraft Chatbot with SQLite Backend

Upgrades Exercise 2.5 by replacing in-memory product dicts with a **SQLite database**. Expands catalog to keyboards, mice, headsets, and mousepads. Tools now query the DB at runtime.

**Key concepts:** SQLite with LangChain tools, persistent data, expanded tool set, production-style architecture

```bash
python Exercise_2.6.py
```

---

### Week 3: GPU & Local Models

---

#### Exercise 3.1 — Local Diffusion Models on GPU ✅

Runs Stability AI's SDXL model family locally on an RTX 3060 (12GB VRAM). Three generation modes of increasing quality, plus Microsoft SpeechT5 text-to-speech. Uses float16 precision and attention slicing to fit within VRAM limits.

**Key concepts:** Stable Diffusion XL, two-stage base+refiner pipeline, float16, attention slicing, VRAM management

| Mode | Steps | Speed | VRAM |
|---|---|---|---|
| SDXL Turbo | 4 | ~5s | ~2GB |
| SDXL Base | 30 | ~20s | ~8GB |
| Base + Refiner | 32+8 | ~60s | ~10GB |

```bash
python Exercise_3_1.py
# Outputs saved to outputs/
```

> **Requires:** NVIDIA GPU with 12GB VRAM and a recent driver (CUDA 13.0 build via `uv sync`)

---

#### Exercise 3.2 — HuggingFace Pipelines Explorer ✅

A 9-tab Gradio app exposing HuggingFace's high-level `pipeline()` API for common NLP and multi-modal tasks: sentiment analysis (incl. multilingual), named entity recognition, question answering, summarization, translation, zero-shot classification, text generation, image generation, and text-to-speech. Models are lazy-loaded on first use and cached.

**Key concepts:** `transformers.pipeline`, task abstraction, lazy model loading, multi-task UI, SpeechT5 TTS

```bash
python Exercise_3.2.py
```

---

#### Exercise 3.3 — Tokenizers Explorer ✅

A 5-tab Gradio app for understanding tokenization across model families (Llama 3.1, Phi-4, DeepSeek V3.1, QwenCoder 2.5). Encodes text to token IDs and back, inspects vocab size and special tokens, applies chat templates, compares token counts side-by-side, and tokenizes source code. Runs entirely on CPU — no GPU required.

**Key concepts:** `AutoTokenizer`, token IDs ↔ fragments, special/added vocab, chat templates, cross-model comparison

```bash
python Exercise_3.3.py
```

---

#### Exercise 3.4 — Models Explorer (Quantization) ✅

A 2-tab Gradio app that loads instruct models locally (Llama 3.2 1B, Phi-4 Mini, Gemma 3 270M, Qwen3 4B, DeepSeek-R1 Distill) with optional **4-bit quantization** via `bitsandbytes`. Tab 1 streams generated text token-by-token; Tab 2 inspects layer architecture and memory footprint. Only one model lives in VRAM at a time — switching auto-unloads the previous one.

**Key concepts:** `BitsAndBytesConfig` (NF4, double quant), `TextIteratorStreamer`, VRAM swapping, `get_memory_footprint()`

```bash
python Exercise_3.4.py
```

> **Requires:** `bitsandbytes` + CUDA torch (installed by `uv sync`) for 4-bit models. Llama and Gemma are gated — set `HF_TOKEN` in `.env`.

---

#### Exercise 3.5 — Meeting Minutes from Audio ✅

A 2-step Gradio pipeline that turns a meeting recording into formatted minutes, **fully local on GPU**:
**(1)** transcribe the audio with **Whisper** (`medium.en` by default), then **(2)** feed the transcript
to a 4-bit LLM (Phi-4 Mini by default; Llama 3.2 3B / Qwen3 4B selectable) to produce a summary,
discussion points, takeaways, and action items with owners in Markdown. Audio is decoded with
`soundfile` (no ffmpeg needed); a **"Load Denver sample"** button auto-downloads the course audio, or
you can upload/record your own. Finished minutes stream live and are saved to `outputs/`.

**Key concepts:** Whisper ASR pipeline, ffmpeg-free audio decoding + resampling, two-stage audio→text→summary pipeline, streaming generation, VRAM swapping

```bash
uv run Exercise_3.5.py
```

> **Requires:** CUDA torch + `bitsandbytes`. No OpenAI key needed (transcription is local). The default
> LLM (Phi-4 Mini) is open; picking Llama 3.2 3B needs approved HF access + `HF_TOKEN`.

## 🔑 API Keys

| Service | Get your key |
|---------|-------------|
| **OpenAI** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Anthropic Claude** | [console.anthropic.com](https://console.anthropic.com/) |
| **Google Gemini** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access) |

## 🛠️ Technology Stack

Versions below reflect the current `uv.lock`.

| Component | Version | Use |
|-----------|---------|-----|
| **Python** | 3.12 | Runtime environment |
| **uv** | 0.11+ | Dependency & environment management |
| **PyTorch** | 2.12.0+cu130 | Deep learning with CUDA 13.0 acceleration |
| **Transformers** | 5.10+ | HuggingFace model loading and inference |
| **Diffusers** | 0.37+ | Stable Diffusion pipelines and utilities |
| **bitsandbytes** | 0.49+ | 4-bit / 8-bit quantization (Week 3 Day 4) |
| **Accelerate** | 1.13+ | `device_map` + quantized model loading |
| **Gradio** | 6.16+ | Web UIs for exercises (Week 2 Day 4+) |
| **Anthropic SDK** | 0.105+ | Claude API integration |
| **OpenAI SDK** | 2.41+ | GPT model integration |
| **LangChain** | 1.3+ | LLM orchestration and chaining |
| **Ollama** | Latest | Local LLM inference (Week 3 Day 2+) |

## 💡 Key Learnings

### Memory Management Strategies

1. **Float16 Precision**: Reduces model size by ~50% with minimal quality loss
2. **Attention Slicing**: Trades compute for memory (slower but fits in 12GB)
3. **Sequential CPU Offload**: Moves tensors to CPU between layers (very slow but very cheap)
4. **Pipeline Cleanup**: Always `empty_cache()` and `del` models after use
5. **Model Sharing**: Share components between base and refiner to reduce duplication

### Model Selection Strategy

- **Exploration**: Use SDXL Turbo for quick iteration (4 steps)
- **Production**: Use SDXL Base for balanced quality (30 steps)
- **Polishing**: Add Refiner for final outputs (additional 8 steps, ~30% quality boost)

### Cost-Benefit Analysis

Running SDXL locally on RTX 3060:
- **Cost**: $300 GPU (amortized) + electricity (~$0.10 per generation)
- **Benefit**: Unlimited generations, no API costs, privacy, control
- **Compare**: OpenAI DALL-E costs ~$0.02 per image (100x cheaper at scale)

## 📖 External Resources

- **[Udemy Course](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models)** — Main learning material
- **[HuggingFace Hub](https://huggingface.co)** — Model zoo, datasets, community
- **[PyTorch](https://pytorch.org)** — Deep learning framework
- **[Diffusers Docs](https://huggingface.co/docs/diffusers)** — Image generation pipelines
- **[LangChain Docs](https://python.langchain.com)** — LLM orchestration
- **[Anthropic Claude](https://claude.ai)** — Advanced LLM API
- **[OpenAI API](https://openai.com/api)** — GPT models

## 🔧 Troubleshooting

**`torch.cuda.is_available()` is `False` / torch installed as `x.y.z+cpu`**
The default PyPI `torch` is CPU-only. `pyproject.toml` already pins the CUDA (cu130) index, so a clean
`uv sync` installs the GPU build. If a stray CPU torch slipped in, force the CUDA trio back:
```bash
uv sync --reinstall
# or, ad-hoc, let uv pick the right CUDA wheels:
uv pip install --reinstall --torch-backend=auto torch torchvision torchaudio
uv run check_torch.py   # should print cuda_available True
```

**`Could not import module 'Phi3ForCausalLM'` (Exercise 3.4)**
Transformers 5.x supports these models natively — do **not** pass `trust_remote_code=True`, or it pulls
in 4.x-era remote code that fails to import. Exercise 3.4 already loads without it.

**Out of Memory (OOM) when running Exercise 3.1**
```python
pipe.enable_sequential_cpu_offload()  # slower but uses less VRAM
# or reduce steps: num_inference_steps=16
```

**`PackageNotFoundError: No package metadata was found for bitsandbytes` (Exercise 3.4)**
```bash
# bitsandbytes is in pyproject.toml — a sync installs it:
uv sync
```

**`ModuleNotFoundError`**
```bash
uv sync   # reinstall everything from pyproject.toml + uv.lock
```

**`.env` / API key not found**
```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
# then fill in your keys
```

## 📝 Important Notes

- **First Run (Ex 3.1)**: Downloads ~50GB of models — may take 10–30 min. Cached in `~/.cache/huggingface/` for reuse.
- **GPU Requirement**: CUDA compute capability 7.0+ (RTX 20-series or newer), 12GB VRAM.
- **Privacy (Ex 3.1)**: All computation is local — no data sent to external services.

## 🤝 Learning Notes

This repository documents my progression through the LLM Engineering course:
- Exercises are completed in order (Week 1 → Week 2 → Week 3)
- Each week builds on previous concepts
- Code is organized by day/topic for easy reference
- Comments explain "why" rather than "what" (code is self-documenting)

**Current Progress**: Week 3 Day 5 ✅ (Meeting Minutes from Audio — Whisper + local LLM)

---

**Last Updated**: July 2026  
**Status**: 🟢 Active Learning  
**Next Focus**: Week 4 (Fine-tuning / LoRA & further projects)
