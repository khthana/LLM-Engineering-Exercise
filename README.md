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
├── requirements.txt                 # Standard dependencies (Week 1-2)
├── requirements_cuda.txt            # GPU-accelerated dependencies (Week 3)
├── REQUIREMENTS.md                  # Detailed setup & installation guide
├── README.md                        # This file
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
└── pyproject.toml                   # Project configuration
```

## 🚀 Exercise 3.1: Local GPU Diffusion & Text-to-Speech

### Overview

**Exercise_3_1.py** demonstrates state-of-the-art generative AI on consumer hardware. It runs Stability AI's SDXL model family locally on an RTX 3060 (12GB VRAM), with memory optimization techniques and multiple generation strategies.

This represents Week 3 Day 1 content: moving beyond cloud APIs to local, cost-effective inference with fine-grained control.

### Features

| Model | Steps | Speed | Quality | VRAM |
|-------|-------|-------|---------|------|
| **SDXL Turbo** | 4 | ~5s | Good | ~2GB |
| **SDXL Base 1.0** | 30 | ~20s | Excellent | ~8-10GB |
| **Base + Refiner** | 32+8 | ~60s | Best | ~10GB |
| **SpeechT5 TTS** | N/A | ~2s | Good | ~1GB |

### Hardware Requirements

- **GPU**: NVIDIA GPU (RTX 20-series or newer recommended)
- **VRAM**: Minimum 12GB (8-10GB for models + 2-4GB system overhead)
- **CUDA**: 12.1 or compatible
- **System RAM**: 16GB+ recommended
- **Disk**: ~50GB initial (HuggingFace model cache)
- **Internet**: Required for first download, subsequent runs use cache

### Quick Start

**For detailed setup instructions, see [REQUIREMENTS.md](REQUIREMENTS.md)**

Quick installation for Week 1-2:

```bash
# 1. Clone repo
git clone https://github.com/khthana/LLM-Engineering-Exercise.git
cd LLM-Engineering-Exercise

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API keys
copy .env.example .env
# Edit .env and add your API keys
```

**For Week 3 (GPU-based exercises):**

```bash
# Install CUDA-enabled PyTorch + diffusers
pip install -r requirements_cuda.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Verify GPU setup
python check_torch.py
```

**System Requirements:**
- Python 3.12+
- 16GB+ RAM (32GB recommended)
- GPU with 12GB VRAM (optional, for Week 3)
- API keys from OpenAI, Anthropic, HuggingFace, etc.

👉 **See [REQUIREMENTS.md](REQUIREMENTS.md) for complete setup guide, API key instructions, and troubleshooting.**

### Running

```bash
# Activate environment first
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1

# Run all exercises
python Exercise_3_1.py
```

### Output Files

The script generates outputs in `outputs/` subfolder:
- **outputs/output_sdxl_turbo.png** — Fast generation (4 steps, ~5 sec)
- **outputs/output_sdxl_base.png** — Quality generation (30 steps, ~20 sec)
- **outputs/output_sdxl_base_refiner.png** — Refined quality (80/20 split, ~60 sec)
- **outputs/output_speech.wav** — Text-to-speech audio (optional)

Example prompt: *"A class of data scientists learning AI engineering in a vibrant pop-art style"*

### Understanding the Code

**GPU Memory Optimization** (lines 164-165)
```python
# Trade computation time for memory - critical for RTX 3060
base.enable_attention_slicing()
```

**Float16 Precision** (throughout)
```python
# Reduces model size by ~50% with minimal quality loss
torch_dtype=torch.float16,
variant="fp16"
```

**Two-Stage Generation** (lines 179-199)
```python
# Base model (80%): Denoises from high noise → medium noise
# Refiner (20%): Denoises from medium → low noise (final image)
# Higher quality than single-stage at same total steps
```

**Pipeline Cleanup** (lines 100, 137, 206)
```python
# Essential - prevents memory accumulation between models
torch.cuda.empty_cache()
del pipe
```

### Troubleshooting

**"Torch not compiled with CUDA enabled"**
```bash
# Wrong version installed - reinstall with CUDA
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

**Out of Memory (OOM) Error**
```python
# Option 1: Enable CPU offloading (slower but fits in 12GB)
pipe.enable_sequential_cpu_offload()

# Option 2: Reduce inference steps
num_inference_steps=16  # Instead of 30

# Option 3: Use SDXL Turbo (only 4 steps)
# Run SDXL Turbo instead of Base
```

**Models not downloading**
- Verify token: `echo $HF_TOKEN` or `$env:HF_TOKEN`
- Check disk space: `df -h` (need ~50GB)
- Check internet connection
- Try manual download: `huggingface-cli download stabilityai/sdxl-turbo`

## 📚 Course Structure

### Week 1: Foundations
- Day 1 (Exercise_1.1): LLM introduction, OpenAI API setup
- Day 2 (Exercise_1.2): Prompt engineering fundamentals
- Day 3-5 (Exercise_1.5, 1.5v2, 1.6): Token counting, cost estimation, and applications

### Week 2: Production Techniques
- Day 1 (Exercise_2.1): LangChain basics and chains
- Day 2 (Exercise_2.2): Tool use and function calling
- Day 3 (Exercise_2.3): Multi-provider orchestration (OpenAI + Anthropic)
- Day 4 (Exercise_2.4): Building stateful chatbots
- Day 5 (Exercise_2.5): Error handling and retry strategies
- Day 6 (Exercise_2.6): KeyCraft chatbot with LangChain tool use

### Week 3: Advanced & Multimodal
- **Day 1**: Local GPU diffusion models (SDXL) ✅
- Day 2: Local LLMs with Ollama
- Day 3: RAG (Retrieval Augmented Generation)
- Day 4: Image understanding with vision models
- Day 5: Production deployment patterns

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

## 📝 Important Notes

- **First Run**: Downloads ~50GB of models. May take 10-30 minutes depending on internet speed.
- **Caching**: Models are cached locally in `~/.cache/huggingface/` and reused between runs.
- **GPU Requirement**: Requires CUDA compute capability 7.0+ (RTX 20-series or newer).
- **VRAM Allocation**: Models allocate and deallocate VRAM dynamically; monitor with `nvidia-smi`.
- **Privacy**: All computation happens locally — no data sent to external services.

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
