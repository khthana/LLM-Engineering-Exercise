# CLAUDE.md

Guidance for AI coding agents working in this repository.

## What this is

A personal learning repo for the Udemy *"LLM Engineering: Master AI and Large Language Models"*
course. Each lesson is a **standalone script** named `Exercise_<week>.<day>.py` (e.g.
`Exercise_2.5.py`), grouped into `week1/`, `week2/`, `week3/` folders. They are adapted from the
course's Jupyter notebooks into runnable `.py` files. There is **no shared package or import graph**
between exercises — each copies what it needs (e.g. `LLMManager`) inline, so treat every file as
self-contained and moveable.

- **`week1/`** (`Exercise_1.*`) — LLM fundamentals: API calls, web scraping, brochure/news generators.
- **`week2/`** (`Exercise_2.*`) — production techniques: streaming, LiteLLM, LangChain, Gradio UIs, tool calling, SQLite.
- **`week3/`** (`Exercise_3.*`) — local GPU models on an **RTX 3060 (12GB)**: SDXL, HF pipelines, tokenizers, quantized models, audio→minutes.

Scripts use **cwd-relative** paths (`outputs/`, `samples/`, `.env`), so always run them from the
repo root: `uv run week3/Exercise_3.5.py`.

## Environment & commands

This project is managed with **uv** (`pyproject.toml` + `uv.lock` are the source of truth).
The `requirements*.txt` files are legacy references — do not edit them to change dependencies;
edit `pyproject.toml` and run `uv lock`.

```bash
uv sync                       # create/refresh the venv from the lockfile
uv run week3/Exercise_3.4.py  # run an exercise (from the repo root)
uv run check_torch.py         # verify CUDA is available
uv add <package>           # add a dependency (updates pyproject + lock)
```

On Windows the interpreter is `.\.venv\Scripts\python.exe`.

## Critical environment facts (read before touching Week 3)

The installed stack is **newer than the notebooks/course assume**:

- `torch 2.12.0+cu130` (CUDA 13.0), `transformers 5.10.2`, `bitsandbytes 0.49.2`, `accelerate 1.13.0`, `gradio 6.16`.
- **CUDA torch is pinned via a custom index.** `pyproject.toml` has `[tool.uv.sources]` +
  `[[tool.uv.index]]` pointing `torch`/`torchvision`/`torchaudio` at
  `https://download.pytorch.org/whl/cu130`. Without this, uv resolves the CPU wheel
  (`torch==x.y.z+cpu`) and all GPU/quantization code silently breaks. If you see a `+cpu` torch,
  the fix is `uv sync --reinstall` (or `uv pip install --reinstall --torch-backend=auto torch torchvision torchaudio`).
- Keep the three torch packages **version-matched** — a mismatched `torchvision` throws
  `operator torchvision::nms does not exist` at import.

## Transformers 5.x gotchas (bit us already; keep code compatible)

When editing model-loading code, follow these — the course notebooks predate transformers 5.x:

1. **Do NOT pass `trust_remote_code=True`** for Llama/Phi/Gemma/Qwen/DeepSeek. They are natively
   supported in 5.x; forcing remote code pulls in 4.x-era files that fail with
   `Could not import module 'Phi3ForCausalLM'`.
2. **`apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding`, not a bare tensor.**
   Use `return_dict=True` and pass it straight to `generate(**inputs)`. Do not call
   `torch.ones_like()` on the result.
3. **Use `dtype=` not `torch_dtype=`** in `from_pretrained` (the latter is deprecated in 5.x).
4. 4-bit loading (`BitsAndBytesConfig` + `device_map="auto"`) requires both `bitsandbytes` **and**
   `accelerate` installed, plus a CUDA torch build.

## Audio (Exercise 3.5)

- The Whisper ASR `pipeline` decodes audio **paths** via a system `ffmpeg`, which is **not installed**
  on this Windows box. Instead, decode audio yourself with `soundfile` (libsndfile 1.2 handles
  wav/flac/ogg/mp3), downmix to mono, resample to 16 kHz with `torchaudio.functional.resample`, and
  pass the raw `np.float32` array to the pipeline (it assumes 16 kHz for a bare array). See
  `_load_audio` in `week3/Exercise_3.5.py`. Exotic formats (m4a/aac) fall back to the path loader.
- `week3/Week_3_Day_5_Meeting_Minutes_product.ipynb` is the source notebook for `week3/Exercise_3.5.py` — the
  port drops Colab bits (Drive mount, `userdata`) and does transcription **locally** (no OpenAI key).

## Conventions

- Week 2 Day 4+ and all Week 3 explorer exercises wrap their logic in a **Gradio `gr.Blocks` UI**
  with a `build_ui()` function and a `if __name__ == "__main__":` launcher.
- Week 3 model scripts keep **one model in VRAM at a time** and unload on switch — preserve this
  pattern when editing (see `_load`/`_unload` in `week3/Exercise_3.4.py`).
- Multi-provider exercises abstract over OpenAI / Claude / Gemini / OpenRouter / Ollama from a
  single config; secrets come from `.env` (`HF_TOKEN` is needed for gated models: Llama, Gemma).
- Comments explain **why**, not what. Match the surrounding style when adding code.

## Housekeeping

- Generated media goes in `outputs/`.
- Don't commit `.env`, model caches, or `.venv`.
