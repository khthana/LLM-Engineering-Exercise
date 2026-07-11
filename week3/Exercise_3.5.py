#!/usr/bin/env python3
"""
Exercise_3.5: Meeting Minutes from Audio — Local RTX 3060 + Gradio UI
Adapted from Week_3_Day_5_Meeting_Minutes_product.ipynb

Pipeline:
  1. Transcribe an audio file locally with OpenAI Whisper (on GPU).
  2. Feed the transcript to Llama 3.2 3B (4-bit quant) to produce meeting
     minutes — summary, discussion points, takeaways, action items.

Everything runs locally on the GPU; no OpenAI API key is required.
"""

import os
import re
import gc
import time
import subprocess
from pathlib import Path
from threading import Thread

import requests
import numpy as np
import soundfile as sf
import torch
import gradio as gr
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    pipeline,
)

load_dotenv()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLE_DIR = Path("samples")
SAMPLE_DIR.mkdir(exist_ok=True)

# ============================================================================
# MODEL REGISTRY
# ============================================================================
# LLM choices for writing the minutes (all loaded 4-bit). Phi-4 Mini is the
# default because it's open; Llama 3.2 is the notebook's model but is gated and
# needs approved HF access. Qwen3 is another strong open option.
PHI   = "microsoft/Phi-4-mini-instruct"
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
QWEN  = "Qwen/Qwen3-4B-Instruct-2507"

LLM_REGISTRY = {
    "Phi-4 Mini (open, default)":               PHI,
    "Llama 3.2 3B (gated — needs HF access)":   LLAMA,
    "Qwen3 4B (open)":                          QWEN,
}

# Whisper variants — medium.en is the notebook default (English-only, ~1.5 GB
# in fp16). Larger = better quality but slower / more VRAM.
WHISPER_MODELS = {
    "Whisper medium.en (default)": "openai/whisper-medium.en",
    "Whisper small.en (faster)":   "openai/whisper-small.en",
    "Whisper large-v3 (best, multilingual)": "openai/whisper-large-v3",
}

# Denver city council extract from the course (small mp3 on Google Drive).
SAMPLE_GDRIVE_ID = "1N_kpSojRR5RYzupz6nqM8hMSoEF_R7pU"
SAMPLE_PATH = SAMPLE_DIR / "denver_extract.mp3"

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

# Lazily-loaded, cached components. Whisper (fp16) + one LLM (4-bit) together
# fit comfortably in 12 GB, so we keep both cached and offer a manual free.
# The LLM swaps in place when a different one is selected.
_asr: dict = {"model_id": None, "pipe": None}
_llm: dict = {"model_id": None, "model": None, "tokenizer": None}


# ============================================================================
# GPU DETECTION  (shared pattern with Exercise 3.4)
# ============================================================================
def check_gpu():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print(result.stdout)
        if "RTX 3060" in result.stdout:
            print("[OK] Connected to RTX 3060")
            return True
        elif torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("[ERROR] No GPU detected — transcription & generation will be very slow on CPU")
            return False
    except FileNotFoundError:
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        return False


# ============================================================================
# HUGGINGFACE LOGIN
# ============================================================================
def login_huggingface():
    from huggingface_hub import login

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("\n[WARN] HF_TOKEN not set — gated model (Llama 3.2) will fail to load\n")
        return False
    try:
        login(hf_token, add_to_git_credential=True)
        print("[OK] Logged in to HuggingFace")
        return True
    except Exception as e:
        print(f"[ERROR] HuggingFace login failed: {e}")
        return False


# ============================================================================
# SAMPLE AUDIO DOWNLOAD (Google Drive)
# ============================================================================
def download_sample() -> str:
    """Download the Denver extract once and return its local path."""
    if SAMPLE_PATH.exists() and SAMPLE_PATH.stat().st_size > 0:
        return str(SAMPLE_PATH)

    print(f"  Downloading sample audio to {SAMPLE_PATH}...")
    base = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    resp = session.get(base, params={"id": SAMPLE_GDRIVE_ID}, stream=True, timeout=60)

    # Small files download directly; large ones return an HTML "virus scan"
    # interstitial with a confirm token — handle both.
    if "text/html" in resp.headers.get("Content-Type", ""):
        html = resp.text
        confirm = re.search(r'name="confirm"\s+value="([^"]+)"', html)
        uuid = re.search(r'name="uuid"\s+value="([^"]+)"', html)
        if confirm:
            params = {"id": SAMPLE_GDRIVE_ID, "export": "download", "confirm": confirm.group(1)}
            if uuid:
                params["uuid"] = uuid.group(1)
            resp = session.get(
                "https://drive.usercontent.google.com/download",
                params=params, stream=True, timeout=60,
            )
    resp.raise_for_status()

    with open(SAMPLE_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 15):
            if chunk:
                f.write(chunk)
    print(f"[OK] Sample saved — {SAMPLE_PATH.stat().st_size / 1e6:.1f} MB")
    return str(SAMPLE_PATH)


def fn_load_sample():
    try:
        return download_sample()
    except Exception as e:
        raise gr.Error(f"Could not download sample audio: {e}")


# ============================================================================
# STEP 1: TRANSCRIPTION (Whisper on GPU)
# ============================================================================
def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Decode audio to a mono float32 waveform at 16 kHz using soundfile —
    avoids the ffmpeg system dependency that the pipeline's path-loader needs.
    Handles wav/flac/ogg/mp3 natively; exotic formats fall back to ffmpeg."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:                       # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio), sr, target_sr).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def _load_asr(model_id: str):
    if _asr["model_id"] == model_id and _asr["pipe"] is not None:
        return _asr["pipe"]

    # free a previously-loaded (different) Whisper first
    if _asr["pipe"] is not None:
        del _asr["pipe"]
        _asr.update({"model_id": None, "pipe": None})
        gc.collect()
        torch.cuda.empty_cache()

    print(f"  Loading ASR model {model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        return_timestamps=True,   # enables long-form transcription
    )
    _asr.update({"model_id": model_id, "pipe": pipe})
    print(f"[OK] ASR model ready ({device})")
    return pipe


def fn_transcribe(audio_path: str, whisper_choice: str):
    if not audio_path:
        return "", "No audio provided — upload a file or load the sample first."

    model_id = WHISPER_MODELS[whisper_choice]
    try:
        pipe = _load_asr(model_id)
    except Exception as e:
        return "", f"[Error loading Whisper] {e}"

    try:
        # Decode ourselves (no ffmpeg); fall back to the path loader if the
        # format isn't supported by libsndfile (e.g. m4a) and ffmpeg exists.
        try:
            audio_input = _load_audio(audio_path)
        except Exception:
            audio_input = audio_path
        t0 = time.time()
        result = pipe(audio_input)
        transcription = result["text"].strip()
        elapsed = time.time() - t0
    except Exception as e:
        return "", f"[Error during transcription] {e}"

    stats = (
        f"Model: {model_id}  |  "
        f"Characters: {len(transcription):,}  |  "
        f"Words: {len(transcription.split()):,}  |  "
        f"Time: {elapsed:.1f}s"
    )
    return transcription, stats


# ============================================================================
# STEP 2: MEETING MINUTES (Llama 3.2 3B, 4-bit, streaming)
# ============================================================================
SYSTEM_MESSAGE = (
    "You produce minutes of meetings from transcripts, with summary, key "
    "discussion points, takeaways and action items with owners, in markdown "
    "format without code blocks."
)


def _build_user_prompt(transcription: str) -> str:
    return (
        "Below is an extract transcript of a council meeting.\n"
        "Please write minutes in markdown without code blocks, including:\n"
        "- a summary with attendees, location and date\n"
        "- discussion points\n"
        "- takeaways\n"
        "- action items with owners\n\n"
        f"Transcription:\n{transcription}\n"
    )


def _load_llm(model_id: str):
    if _llm["model_id"] == model_id and _llm["model"] is not None:
        return _llm["model"], _llm["tokenizer"]

    # swap out a previously-loaded (different) LLM
    if _llm["model"] is not None:
        del _llm["model"], _llm["tokenizer"]
        _llm.update({"model_id": None, "model": None, "tokenizer": None})
        gc.collect()
        torch.cuda.empty_cache()

    print(f"  Loading {model_id} (4-bit quant)...")
    # transformers 5.x supports these natively — no trust_remote_code (see CLAUDE.md).
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", quantization_config=QUANT_CONFIG
    )
    _llm.update({"model_id": model_id, "model": model, "tokenizer": tokenizer})
    mb = model.get_memory_footprint() / 1e6
    print(f"[OK] {model_id} loaded — {mb:,.1f} MB")
    return model, tokenizer


def fn_minutes(transcription: str, model_name: str, max_new_tokens: int):
    if not transcription or not transcription.strip():
        yield "", "Nothing to summarize — transcribe an audio file first."
        return

    model_id = LLM_REGISTRY[model_name]
    try:
        model, tokenizer = _load_llm(model_id)
    except Exception as e:
        yield "", f"[Error loading LLM] {e}"
        return

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": _build_user_prompt(transcription)},
    ]

    # transformers 5.x returns a BatchEncoding here — pass it straight to generate().
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    ).to("cuda")

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    minutes = ""
    for chunk in streamer:
        minutes += chunk
        yield minutes, "Generating..."
    thread.join()

    # Persist the finished minutes.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"minutes_{stamp}.md"
    out_path.write_text(minutes, encoding="utf-8")
    yield minutes, f"Done — saved to {out_path}"


# ============================================================================
# VRAM MANAGEMENT
# ============================================================================
def fn_free_vram():
    freed = []
    if _asr["pipe"] is not None:
        del _asr["pipe"]
        _asr.update({"model_id": None, "pipe": None})
        freed.append("Whisper")
    if _llm["model"] is not None:
        del _llm["model"], _llm["tokenizer"]
        _llm.update({"model_id": None, "model": None, "tokenizer": None})
        freed.append("LLM")
    gc.collect()
    torch.cuda.empty_cache()
    return f"Freed: {', '.join(freed) if freed else 'nothing was loaded'}. VRAM released."


# ============================================================================
# GRADIO UI
# ============================================================================
def build_ui():
    whisper_names = list(WHISPER_MODELS.keys())
    llm_names = list(LLM_REGISTRY.keys())

    with gr.Blocks(title="Meeting Minutes from Audio — RTX 3060") as demo:
        gr.Markdown(
            "# Meeting Minutes from Audio\n"
            "Adapted from `Week_3_Day_5_Meeting_Minutes_product.ipynb`  •  "
            "Runs fully local on GPU: **Whisper** transcribes → a **4-bit LLM** writes the minutes."
        )

        # ----- STEP 1: Transcription -------------------------------------
        gr.Markdown("## Step 1 — Transcribe audio")
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    label="Audio file (upload or record)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                sample_btn = gr.Button("Load Denver sample (auto-download)")
                whisper_dd = gr.Dropdown(
                    whisper_names, value=whisper_names[0], label="Whisper model"
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary")
            with gr.Column(scale=2):
                transcript_box = gr.Textbox(
                    label="Transcript (editable before generating minutes)",
                    lines=14,
                    max_lines=30,
                )
                transcribe_status = gr.Textbox(label="Transcription status", lines=1)

        sample_btn.click(fn_load_sample, outputs=[audio_in])
        transcribe_btn.click(
            fn_transcribe,
            inputs=[audio_in, whisper_dd],
            outputs=[transcript_box, transcribe_status],
        )

        # ----- STEP 2: Minutes -------------------------------------------
        gr.Markdown("## Step 2 — Generate meeting minutes")
        with gr.Row():
            with gr.Column(scale=1):
                llm_dd = gr.Dropdown(
                    llm_names, value=llm_names[0], label="LLM (all 4-bit)"
                )
                max_tokens = gr.Slider(
                    minimum=500, maximum=3000, value=2000, step=100,
                    label="max_new_tokens",
                )
                minutes_btn = gr.Button("Generate Minutes", variant="primary")
                free_btn = gr.Button("Free VRAM", variant="stop")
                minutes_status = gr.Textbox(label="Generation status", lines=2)
            with gr.Column(scale=2):
                minutes_out = gr.Markdown(label="Meeting Minutes")

        minutes_btn.click(
            fn_minutes,
            inputs=[transcript_box, llm_dd, max_tokens],
            outputs=[minutes_out, minutes_status],
        )
        free_btn.click(fn_free_vram, outputs=[minutes_status])

    return demo


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Exercise 3.5: Meeting Minutes from Audio — Local RTX 3060 + Gradio")
    print("=" * 70)
    print("Pipeline: Whisper (GPU transcription) -> 4-bit LLM (minutes)")
    print()
    print("Notes:")
    print("  - Default LLM is Phi-4 Mini (open). Llama 3.2 3B is gated — pick it")
    print("    only if your HF account has access (set HF_TOKEN in your .env).")
    print("  - Click 'Load Denver sample' to auto-download the course audio,")
    print("    or upload/record your own.")
    print()

    check_gpu()
    login_huggingface()

    demo = build_ui()
    demo.launch()
