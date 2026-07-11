#!/usr/bin/env python3
"""
Exercise_3.4: Models Explorer — Local RTX 3060 + Gradio UI
Adapted from Week_3_Day_4_models.ipynb
"""

import os
import gc
import subprocess
from threading import Thread
from dotenv import load_dotenv
import torch
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

load_dotenv()

# ============================================================================
# MODEL REGISTRY
# ============================================================================
LLAMA    = "meta-llama/Llama-3.2-1B-Instruct"
PHI      = "microsoft/Phi-4-mini-instruct"
GEMMA    = "google/gemma-3-270m-it"
QWEN     = "Qwen/Qwen3-4B-Instruct-2507"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# (model_id, use_4bit_quant)
MODEL_REGISTRY: dict[str, tuple[str, bool]] = {
    "Llama 3.2 1B — 4-bit":        (LLAMA,    True),
    "Phi-4 Mini — 4-bit":          (PHI,      True),
    "Gemma 3 270M — fp16":         (GEMMA,    False),
    "Qwen3 4B — 4-bit":            (QWEN,     True),
    "DeepSeek-R1 Distill 1.5B — fp16": (DEEPSEEK, False),
}

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

# Only one model lives in VRAM at a time — swap on model change
_active: dict = {"model_id": None, "model": None, "tokenizer": None}


def _unload():
    if _active["model"] is not None:
        old = _active["model_id"]
        del _active["model"], _active["tokenizer"]
        _active.update({"model_id": None, "model": None, "tokenizer": None})
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[INFO] Unloaded {old}")


def _load(model_id: str, quant: bool):
    if _active["model_id"] == model_id:
        return _active["model"], _active["tokenizer"]

    _unload()
    print(f"  Loading {model_id} ({'4-bit quant' if quant else 'fp16'})...")

    # transformers 5.x supports all five models natively — do NOT use
    # trust_remote_code, or Phi-4 pulls in 4.x-era remote code that fails to
    # import under 5.x ("Could not import module 'Phi3ForCausalLM'").
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto"}
    if quant:
        kwargs["quantization_config"] = QUANT_CONFIG
    else:
        kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    _active.update({"model_id": model_id, "model": model, "tokenizer": tokenizer})

    mb = model.get_memory_footprint() / 1e6
    print(f"[OK] {model_id} loaded — {mb:,.1f} MB")
    return model, tokenizer


# ============================================================================
# GPU DETECTION
# ============================================================================
def check_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
        if 'RTX 3060' in result.stdout:
            print("[OK] Connected to RTX 3060")
            return True
        elif torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("[ERROR] No GPU detected — generation will be very slow on CPU")
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

    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("\n[WARN] HF_TOKEN not set — gated models (Llama, Gemma) will fail\n")
        return False
    try:
        login(hf_token, add_to_git_credential=True)
        print("[OK] Logged in to HuggingFace")
        return True
    except Exception as e:
        print(f"[ERROR] HuggingFace login failed: {e}")
        return False


# ============================================================================
# TAB 1: TEXT GENERATION  (streaming)
# ============================================================================
def fn_generate(model_name: str, system_msg: str, user_msg: str, max_new_tokens: int):
    model_id, quant = MODEL_REGISTRY[model_name]
    try:
        model, tokenizer = _load(model_id, quant)
    except Exception as e:
        yield f"[Error loading model] {e}"
        return

    messages = []
    if system_msg.strip():
        messages.append({"role": "system", "content": system_msg.strip()})
    messages.append({"role": "user", "content": user_msg})

    try:
        # transformers 5.x returns a BatchEncoding (input_ids + attention_mask)
        # here, not a bare tensor — pass it straight to generate().
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to("cuda")
    except Exception as e:
        yield f"[Error applying chat template] {e}"
        return

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

    output = ""
    for chunk in streamer:
        output += chunk
        yield output

    thread.join()


# ============================================================================
# TAB 2: MODEL INSPECTOR
# ============================================================================
def fn_inspect(model_name: str):
    model_id, quant = MODEL_REGISTRY[model_name]
    try:
        model, _ = _load(model_id, quant)
    except Exception as e:
        return f"Error: {e}", ""

    mb     = model.get_memory_footprint() / 1e6
    params = sum(p.numel() for p in model.parameters()) / 1e6
    arch   = repr(model)

    stats = (
        f"Model:              {model_id}\n"
        f"Quantization:       {'4-bit NF4 (double quant, bfloat16 compute)' if quant else 'fp16 (no quantization)'}\n"
        f"Memory footprint:   {mb:,.1f} MB\n"
        f"Total parameters:   {params:,.1f} M"
    )
    return stats, arch


def fn_unload():
    name = _active["model_id"] or "none"
    _unload()
    return f"Unloaded: {name}\nVRAM released."


# ============================================================================
# GRADIO UI
# ============================================================================
DEFAULT_SYSTEM = "You are a helpful assistant"
DEFAULT_USER   = "Tell a light-hearted joke for a room of Data Scientists"

MODEL_NAMES = list(MODEL_REGISTRY.keys())


def build_ui():
    with gr.Blocks(title="Models Explorer — RTX 3060") as demo:
        gr.Markdown(
            "# Models Explorer\n"
            "Adapted from `Week_3_Day_4_models.ipynb`  •  "
            "Only one model lives in VRAM at a time — switching models auto-unloads the previous one."
        )

        with gr.Tabs():

            # ----------------------------------------------------------------
            with gr.Tab("1. Text Generation"):
                gr.Markdown(
                    "Loads the selected model (with optional 4-bit quantization) and streams "
                    "generated text token-by-token — equivalent to `TextStreamer` in the notebook."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gen_model  = gr.Dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="Model")
                        gen_system = gr.Textbox(
                            label="System Message (optional)",
                            value=DEFAULT_SYSTEM,
                            lines=2,
                        )
                        gen_user   = gr.Textbox(
                            label="User Message",
                            value=DEFAULT_USER,
                            lines=3,
                        )
                        gen_tokens = gr.Slider(
                            minimum=20, maximum=500, value=80, step=10,
                            label="max_new_tokens",
                        )
                        with gr.Row():
                            gen_btn   = gr.Button("Generate", variant="primary")
                            clear_btn = gr.Button("Clear")
                    with gr.Column(scale=2):
                        gen_out = gr.Textbox(
                            label="Model output (streaming)",
                            lines=20,
                            max_lines=40,
                        )

                gen_btn.click(
                    fn_generate,
                    inputs=[gen_model, gen_system, gen_user, gen_tokens],
                    outputs=[gen_out],
                )
                clear_btn.click(lambda: "", outputs=[gen_out])

            # ----------------------------------------------------------------
            with gr.Tab("2. Model Inspector"):
                gr.Markdown(
                    "Load a model and inspect its **layer architecture** and **memory footprint**.  \n"
                    "This mirrors the `model` printout and `model.get_memory_footprint()` cells in the notebook.  \n"
                    "Look for: embedding layer → N decoder layers (self-attention + MLP + norms) → LM head."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        insp_model = gr.Dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="Model")
                        insp_btn   = gr.Button("Load & Inspect", variant="primary")
                        unload_btn = gr.Button("Unload Current Model", variant="stop")
                        unload_out = gr.Textbox(label="Unload status", lines=2)
                    with gr.Column(scale=2):
                        insp_stats = gr.Textbox(label="Stats", lines=6)
                        insp_arch  = gr.Textbox(
                            label="Model Architecture (PyTorch repr)",
                            lines=30,
                            max_lines=60,
                        )

                insp_btn.click(
                    fn_inspect,
                    inputs=[insp_model],
                    outputs=[insp_stats, insp_arch],
                )
                unload_btn.click(fn_unload, outputs=[unload_out])

    return demo


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Exercise 3.4: Models Explorer — Local RTX 3060 + Gradio")
    print("=" * 70)
    print("Models and their quantization:")
    for name, (mid, quant) in MODEL_REGISTRY.items():
        print(f"  {'[4-bit]' if quant else '[fp16] '} {name}")
    print()
    print("Notes:")
    print("  - Llama 3.2 and Gemma 3 require HuggingFace terms acceptance.")
    print("  - Only one model loads at a time — switching auto-frees VRAM.")
    print()

    check_gpu()
    login_huggingface()

    demo = build_ui()
    demo.launch()
