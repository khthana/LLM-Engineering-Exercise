#!/usr/bin/env python3
"""
Exercise_3.3: Tokenizers Explorer — Local RTX 3060 + Gradio UI
Adapted from Week_3_Day_3_tokenizers.ipynb
"""

import os
import subprocess
import torch
import pandas as pd
from dotenv import load_dotenv
import gradio as gr
from transformers import AutoTokenizer

load_dotenv()

LLAMA_BASE     = "meta-llama/Meta-Llama-3.1-8B"
LLAMA_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI4           = "microsoft/Phi-4-mini-instruct"
DEEPSEEK       = "deepseek-ai/DeepSeek-V3.1"
QWEN_CODER     = "Qwen/Qwen2.5-Coder-7B-Instruct"

MODEL_DISPLAY = {
    "Llama 3.1 (Base)":     LLAMA_BASE,
    "Llama 3.1 (Instruct)": LLAMA_INSTRUCT,
    "Phi-4 Mini":           PHI4,
    "DeepSeek V3.1":        DEEPSEEK,
    "QwenCoder 2.5":        QWEN_CODER,
}

INSTRUCT_NAMES = ["Llama 3.1 (Instruct)", "Phi-4 Mini", "DeepSeek V3.1", "QwenCoder 2.5"]

_tok_cache = {}


def load_tokenizer(model_id: str) -> AutoTokenizer:
    if model_id not in _tok_cache:
        print(f"  Loading tokenizer: {model_id}...")
        _tok_cache[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return _tok_cache[model_id]


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
            print("[INFO] No GPU — tokenizers run on CPU (no GPU needed)")
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
        print("\n[WARN] HF_TOKEN not set — gated models (Llama) will require manual login\n")
        return False
    try:
        login(hf_token, add_to_git_credential=True)
        print("[OK] Logged in to HuggingFace")
        return True
    except Exception as e:
        print(f"[ERROR] HuggingFace login failed: {e}")
        return False


# ============================================================================
# DEFAULT VALUES
# ============================================================================
DEFAULT_TEXT         = "I am excited to show Tokenizers in action to my LLM engineers"
DEFAULT_COMPARE_TEXT = "I am curiously excited to show Hugging Face Tokenizers in action to my LLM engineers"
DEFAULT_SYSTEM       = "You are a helpful assistant"
DEFAULT_USER         = "Tell a light-hearted joke for a room of Data Scientists"
DEFAULT_CODE         = 'def hello_world(person):\n  print("Hello", person)\n'


# ============================================================================
# TAB 1: BASIC TOKENIZATION
# ============================================================================
def fn_basic_tokenize(text: str, model_name: str):
    model_id = MODEL_DISPLAY[model_name]
    try:
        tok = load_tokenizer(model_id)
    except Exception as e:
        return pd.DataFrame(), f"Error loading tokenizer: {e}", ""

    tokens    = tok.encode(text)
    fragments = tok.batch_decode(tokens)
    decoded   = tok.decode(tokens)

    stats = (
        f"Characters: {len(text)}  |  "
        f"Words: {len(text.split())}  |  "
        f"Tokens: {len(tokens)}"
    )
    table = pd.DataFrame({"Token ID": tokens, "Fragment": fragments})
    return table, stats, decoded


# ============================================================================
# TAB 2: VOCABULARY INFO
# ============================================================================
def fn_vocab_info(model_name: str):
    model_id = MODEL_DISPLAY[model_name]
    try:
        tok = load_tokenizer(model_id)
    except Exception as e:
        return f"Error: {e}", ""

    vocab_size = getattr(tok, 'vocab_size', len(tok.get_vocab()))
    added      = tok.get_added_vocab()
    preview    = dict(list(added.items())[:50])

    stats_str = f"Vocab size: {vocab_size}\nAdded vocab entries: {len(added)}"
    added_str = "\n".join(f"  {k!r}: {v}" for k, v in preview.items())
    if len(added) > 50:
        added_str += f"\n  ... and {len(added) - 50} more"

    return stats_str, added_str


# ============================================================================
# TAB 3: CHAT TEMPLATE (single model)
# ============================================================================
def fn_chat_template(system_msg: str, user_msg: str, model_name: str):
    model_id = MODEL_DISPLAY[model_name]
    try:
        tok = load_tokenizer(model_id)
    except Exception as e:
        return f"Error loading tokenizer: {e}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        return f"Error applying chat template: {e}"


# ============================================================================
# TAB 4: MULTI-MODEL COMPARISON
# ============================================================================
def fn_compare_tokens(text: str):
    targets = [
        ("Llama 3.1", LLAMA_INSTRUCT),
        ("Phi-4 Mini", PHI4),
        ("DeepSeek V3.1", DEEPSEEK),
    ]
    rows = []
    for name, model_id in targets:
        try:
            tok       = load_tokenizer(model_id)
            tokens    = tok.encode(text)
            fragments = tok.batch_decode(tokens)
            rows.append({
                "Model":                   name,
                "Token Count":             len(tokens),
                "Token IDs (first 20)":    str(tokens[:20]),
                "Fragments (first 10)":    str(fragments[:10]),
            })
        except Exception as e:
            rows.append({
                "Model":                name,
                "Token Count":          f"Error: {e}",
                "Token IDs (first 20)": "",
                "Fragments (first 10)": "",
            })
    return pd.DataFrame(rows)


def fn_compare_templates(system_msg: str, user_msg: str):
    targets = [
        ("Llama 3.1", LLAMA_INSTRUCT),
        ("Phi-4 Mini", PHI4),
        ("DeepSeek V3.1", DEEPSEEK),
    ]
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    parts = []
    for name, model_id in targets:
        try:
            tok    = load_tokenizer(model_id)
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            parts.append(f"### {name}\n```\n{prompt}\n```")
        except Exception as e:
            parts.append(f"### {name}\nError: {e}")
    return "\n\n---\n\n".join(parts)


# ============================================================================
# TAB 5: CODE TOKENIZATION (QwenCoder)
# ============================================================================
def fn_code_tokenize(code: str):
    try:
        tok = load_tokenizer(QWEN_CODER)
    except Exception as e:
        return pd.DataFrame(), f"Error: {e}"

    tokens = tok.encode(code)
    rows   = [{"Token ID": t, "Fragment": tok.decode(t)} for t in tokens]
    return pd.DataFrame(rows)


# ============================================================================
# GRADIO UI
# ============================================================================
def build_ui():
    all_names = list(MODEL_DISPLAY.keys())

    with gr.Blocks(title="Tokenizers Explorer — RTX 3060") as demo:
        gr.Markdown(
            "# Tokenizers Explorer\n"
            "Adapted from `Week_3_Day_3_tokenizers.ipynb`  •  Tokenizers run on CPU — no GPU required."
        )

        with gr.Tabs():

            # ----------------------------------------------------------------
            with gr.Tab("1. Basic Tokenization"):
                gr.Markdown(
                    "Encode text into token IDs and decode each token back to its fragment.  \n"
                    "This illustrates what an LLM actually receives as input."
                )
                with gr.Row():
                    with gr.Column():
                        bt_text  = gr.Textbox(label="Input Text", value=DEFAULT_TEXT, lines=3)
                        bt_model = gr.Dropdown(all_names, value="Llama 3.1 (Instruct)", label="Model")
                        bt_btn   = gr.Button("Tokenize", variant="primary")
                    with gr.Column():
                        bt_stats   = gr.Textbox(label="Statistics (chars / words / tokens)")
                        bt_decoded = gr.Textbox(label="Decoded round-trip", lines=2)
                bt_table = gr.DataFrame(
                    label="Token ID → Fragment",
                    headers=["Token ID", "Fragment"],
                )
                bt_btn.click(
                    fn_basic_tokenize,
                    inputs=[bt_text, bt_model],
                    outputs=[bt_table, bt_stats, bt_decoded],
                )

            # ----------------------------------------------------------------
            with gr.Tab("2. Vocabulary Info"):
                gr.Markdown(
                    "Inspect the vocabulary size and the special tokens added on top of the base vocab "
                    "(e.g. `<|begin_of_text|>`, `<|eot_id|>` for Llama)."
                )
                with gr.Row():
                    with gr.Column():
                        vi_model = gr.Dropdown(all_names, value="Llama 3.1 (Instruct)", label="Model")
                        vi_btn   = gr.Button("Load Vocab Info", variant="primary")
                    with gr.Column():
                        vi_stats = gr.Textbox(label="Vocab Stats", lines=3)
                        vi_added = gr.Textbox(label="Added Vocab (first 50 entries)", lines=12)
                vi_btn.click(fn_vocab_info, inputs=[vi_model], outputs=[vi_stats, vi_added])

            # ----------------------------------------------------------------
            with gr.Tab("3. Chat Template"):
                gr.Markdown(
                    "**Key insight from the notebook:** LLMs cannot receive Python dicts directly.  \n"
                    "The OpenAI-format `messages` list is first serialized into a flat string "
                    "with model-specific special tags, then tokenized into IDs."
                )
                with gr.Row():
                    with gr.Column():
                        ct_sys   = gr.Textbox(label="System Message", value=DEFAULT_SYSTEM, lines=2)
                        ct_user  = gr.Textbox(label="User Message",   value=DEFAULT_USER,   lines=3)
                        ct_model = gr.Dropdown(INSTRUCT_NAMES, value="Llama 3.1 (Instruct)", label="Model")
                        ct_btn   = gr.Button("Apply Chat Template", variant="primary")
                    with gr.Column():
                        ct_out = gr.Textbox(label="Formatted Prompt (what the model actually sees)", lines=18)
                ct_btn.click(fn_chat_template, inputs=[ct_sys, ct_user, ct_model], outputs=[ct_out])

            # ----------------------------------------------------------------
            with gr.Tab("4. Multi-Model Comparison"):
                gr.Markdown("Compare how Llama, Phi-4, and DeepSeek tokenize the same input.")

                gr.Markdown("#### Token counts & fragments")
                with gr.Row():
                    with gr.Column():
                        cmp_text = gr.Textbox(label="Input Text", value=DEFAULT_COMPARE_TEXT, lines=3)
                        cmp_btn  = gr.Button("Compare Tokens", variant="primary")
                cmp_table = gr.DataFrame(label="Comparison")
                cmp_btn.click(fn_compare_tokens, inputs=[cmp_text], outputs=[cmp_table])

                gr.Markdown("---\n#### Chat template formats side-by-side")
                with gr.Row():
                    with gr.Column():
                        cmp_sys  = gr.Textbox(label="System Message", value=DEFAULT_SYSTEM, lines=2)
                        cmp_usr  = gr.Textbox(label="User Message",   value=DEFAULT_USER,   lines=2)
                        cmp_btn2 = gr.Button("Compare Templates", variant="primary")
                cmp_tmpl = gr.Markdown()
                cmp_btn2.click(fn_compare_templates, inputs=[cmp_sys, cmp_usr], outputs=[cmp_tmpl])

            # ----------------------------------------------------------------
            with gr.Tab("5. Code Tokenization (QwenCoder)"):
                gr.Markdown(
                    "QwenCoder 2.5 is optimized for code.  \n"
                    "See how it fragments Python source line by line — indentation and keywords each get their own tokens."
                )
                with gr.Row():
                    with gr.Column():
                        code_in  = gr.Textbox(label="Input Code", value=DEFAULT_CODE, lines=8)
                        code_btn = gr.Button("Tokenize Code", variant="primary")
                code_table = gr.DataFrame(
                    label="Token ID → Fragment (QwenCoder 2.5)",
                    headers=["Token ID", "Fragment"],
                )
                code_btn.click(fn_code_tokenize, inputs=[code_in], outputs=[code_table])

    return demo


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Exercise 3.3: Tokenizers Explorer — Local RTX 3060 + Gradio")
    print("=" * 70)
    print("Note: Tokenizers run entirely on CPU — GPU is not required here.")
    print("      Llama models are gated; ensure HF_TOKEN is set in your .env")
    print()

    check_gpu()
    login_huggingface()

    demo = build_ui()
    demo.launch()
