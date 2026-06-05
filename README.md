# LLM Engineering Master - AI & Large Language Models

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue) ![PyTorch CUDA](https://img.shields.io/badge/PyTorch-2.5.1%2BCuda12.1-green) ![Status Active](https://img.shields.io/badge/Status-Active-brightgreen)

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
├── outputs/                         # Generated images and audio
├── requirements.txt                 # Base dependencies (Week 1-2)
├── requirements-cuda.txt            # GPU dependencies (Week 3 - extends requirements.txt)
├── README.md                        # This file
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
└── pyproject.toml                   # Project configuration
```

## 🚀 Quick Start

```bash
# 1. Clone & enter repo
git clone https://github.com/khthana/LLM-Engineering-Exercise.git
cd LLM-Engineering-Exercise

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 3a. Week 1-2: API-based exercises
pip install -r requirements.txt

# 3b. Week 3: GPU exercises (includes everything above + PyTorch CUDA)
pip install -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu121

# 4. Set API keys
copy .env.example .env   # then edit .env with your actual keys
```

## 📚 Exercises

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

> **Requires:** NVIDIA GPU with 12GB VRAM, CUDA 12.1, `requirements-cuda.txt`

## 🔑 API Keys

| Service | Get your key |
|---------|-------------|
| **OpenAI** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Anthropic Claude** | [console.anthropic.com](https://console.anthropic.com/) |
| **Google Gemini** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access) |

## 🛠️ Technology Stack

| Component | Version | Use |
|-----------|---------|-----|
| **Python** | 3.12 | Runtime environment |
| **PyTorch** | 2.5.1+cu121 | Deep learning with CUDA acceleration |
| **Transformers** | 4.46+ | HuggingFace model loading and inference |
| **Diffusers** | 0.31+ | Stable Diffusion pipelines and utilities |
| **Anthropic SDK** | Latest | Claude API integration |
| **OpenAI SDK** | Latest | GPT-4, GPT-3.5 integration |
| **LangChain** | Latest | LLM orchestration and chaining |
| **LlamaIndex** | Latest | RAG framework (as needed) |
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

**"Torch not compiled with CUDA" / CUDA not available**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python check_torch.py
```

**Out of Memory (OOM) when running Exercise 3.1**
```python
pipe.enable_sequential_cpu_offload()  # slower but uses less VRAM
# or reduce steps: num_inference_steps=16
```

**`ModuleNotFoundError`**
```bash
# Make sure venv is activated, then:
pip install -r requirements.txt          # Week 1-2
pip install -r requirements-cuda.txt ... # Week 3
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

**Current Progress**: Week 3 Day 1 ✅ (Local GPU models)

---

**Last Updated**: June 2026  
**Status**: 🟢 Active Learning  
**Next Focus**: Week 3 Day 2 (Local Ollama models & RAG)
