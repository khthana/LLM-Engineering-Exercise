# Requirements & Setup Guide

## System Requirements

### Hardware
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 16GB minimum (32GB recommended for comfortable development)
- **GPU** (optional but recommended for Week 3):
  - NVIDIA GPU with CUDA compute capability 7.0+ (RTX 20-series or newer)
  - 12GB+ VRAM for running local diffusion models
  - CUDA Toolkit 12.1 compatible

### Software
- **Python**: 3.12+ (installed and in PATH)
- **Git**: For version control
- **Internet**: For downloading models and API calls

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/khthana/LLM-Engineering-Exercise.git
cd LLM-Engineering-Exercise
```

### 2. Create Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Choose based on your use case:

**Option A: Week 1-2 (API-based, no GPU)**

```bash
pip install -r requirements.txt
```

**Option B: Week 3 (GPU-accelerated, all features)**

```bash
# Install PyTorch with CUDA 12.1 support
pip install -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
python check_torch.py
```

**Note:** `requirements-cuda.txt` includes everything from `requirements.txt` + GPU/torch support. You only need one.

### 5. Set Up Environment Variables

Copy the template and fill in your API keys:

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Then edit `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-proj-your-actual-key
ANTHROPIC_API_KEY=sk-ant-your-actual-key
HF_TOKEN=hf_your-actual-token
# ... etc
```

**How to get API keys:**

| Service | Instructions |
|---------|--------------|
| **OpenAI** | 1. Create account at https://platform.openai.com<br/>2. Go to API keys<br/>3. Create new secret key |
| **Anthropic Claude** | 1. Create account at https://console.anthropic.com<br/>2. Navigate to API keys<br/>3. Create new key |
| **Google** | 1. Go to Google Cloud Console<br/>2. Create new project<br/>3. Enable APIs (Vertex AI, etc.)<br/>4. Create service account key |
| **OpenRouter** | 1. Sign up at https://openrouter.ai<br/>2. Go to Keys section<br/>3. Create new API key |
| **HuggingFace** | 1. Create account at https://huggingface.co<br/>2. Settings → Access Tokens<br/>3. Create new token (read access sufficient) |

## Dependencies Overview

### `requirements.txt` (Week 1-2 - Base)

Core API clients and LLM orchestration:
- `openai` — OpenAI API client (GPT-4, GPT-3.5)
- `anthropic` — Anthropic Claude API
- `google-generativeai` — Google Gemini API
- `langchain` — LLM orchestration framework
- `langchain-community` — Extended LangChain tools
- `langchain-openai`, `langchain-anthropic` — Provider integrations
- `python-dotenv` — Environment variable management
- `requests` — HTTP library
- `pydantic` — Data validation

### `requirements-cuda.txt` (Week 3 - Complete with GPU)

Includes everything from `requirements.txt` PLUS GPU/ML libraries:
- `torch==2.5.1` — Deep learning framework (CUDA 12.1)
- `torchvision==0.20.1` — Computer vision utilities
- `torchaudio==2.5.1` — Audio processing
- `transformers==4.46.0` — HuggingFace model inference
- `diffusers==0.31.0` — Stable Diffusion image generation
- `accelerate` — GPU optimization utilities
- `safetensors` — Safe tensor serialization (faster loading)
- `soundfile` — Audio file I/O
- `datasets` — HuggingFace dataset utilities
- `peft` — Parameter-efficient fine-tuning

## Verification

### Check Python Installation
```bash
python --version  # Should be 3.12+
```

### Check Virtual Environment
```bash
which python  # or 'where python' on Windows
# Should point to .venv/bin/python or .venv\Scripts\python.exe
```

### Check Dependencies
```bash
# Show all installed packages
pip list

# Check specific package
pip show torch
pip show openai
```

### Check GPU (if you have one)
```bash
python check_torch.py
# Should show RTX 3060 and CUDA version
```

## Troubleshooting

### "Python 3.12 not found"
- Install Python 3.12 from https://www.python.org
- Or use package manager:
  - Windows: `winget install Python.Python.3.12`
  - macOS: `brew install python@3.12`
  - Linux: `sudo apt-get install python3.12`

### "Torch not compiled with CUDA" or "CUDA not available"
```bash
# Uninstall wrong version
pip uninstall torch torchvision torchaudio -y

# Reinstall correct version with CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# Verify
python check_torch.py
```

### "ModuleNotFoundError: No module named 'openai'"
```bash
pip install openai
# Or reinstall all requirements
pip install -r requirements.txt
```

### "HF_TOKEN environment variable not found"
- Make sure you've created `.env` file (copy from `.env.example`)
- Set your HuggingFace token in `.env`
- For quick testing:
  - Windows: `$env:HF_TOKEN='hf_your_token'`
  - Linux/Mac: `export HF_TOKEN='hf_your_token'`

### Import errors in Exercise files
- Activate virtual environment first
- Run `pip install -r requirements.txt` (and `requirements_cuda.txt` for Week 3)

## Next Steps

1. ✅ Complete system setup above
2. 📚 Read the main [README.md](README.md)
3. 🚀 Start with Exercise_1.1.py (Week 1 Day 1)
4. 📝 Set up .env with your API keys
5. 🔄 Work through exercises sequentially

---

**Need help?** Check the [README.md](README.md) for detailed exercise documentation.
