#!/usr/bin/env python3
"""
Exercise_3.2: HuggingFace Pipelines — Local RTX 3060 + Gradio UI
Adapted from week_3_day_2_pipelines.ipynb
"""

import os
import torch
import subprocess
import zipfile
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Lazily-loaded model cache — avoids re-downloading on every button click
_cache = {}

def load_once(key, loader_fn):
    if key not in _cache:
        print(f"  Loading {key}...")
        _cache[key] = loader_fn()
    return _cache[key]


# ============================================================================
# GPU DETECTION  (reused from Exercise_3.1)
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
            print("[ERROR] No GPU detected")
            return False
    except FileNotFoundError:
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        return False


# ============================================================================
# HUGGINGFACE LOGIN  (reused from Exercise_3.1)
# ============================================================================
def login_huggingface():
    from huggingface_hub import login

    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("\n[WARN] HF_TOKEN not set — some models may require authentication\n")
        return False
    try:
        login(hf_token, add_to_git_credential=True)
        print("[OK] Logged in to HuggingFace")
        return True
    except Exception as e:
        print(f"[ERROR] HuggingFace login failed: {e}")
        return False


# ============================================================================
# DEFAULT TEXTS  (from original notebook cells)
# ============================================================================
DEFAULT_SENTIMENT  = "I'm super excited to be on the way to LLM mastery!"
DEFAULT_NER        = "AI Engineers are learning about the amazing pipelines from HuggingFace in Google Colab from Ed Donner"
DEFAULT_QA_Q       = "What are Hugging Face pipelines?"
DEFAULT_QA_C       = "Pipelines are a high level API for inference of LLMs with common tasks"
DEFAULT_SUMMARY    = (
    "The Hugging Face transformers library is an incredibly versatile and powerful tool "
    "for natural language processing (NLP). It allows users to perform a wide range of "
    "tasks such as text classification, named entity recognition, and question answering, "
    "among others. It's an extremely popular library that's widely used by the open-source "
    "data science community. It lowers the barrier to entry into the field by providing "
    "Data Scientists with a productive, convenient way to work with transformer models."
)
DEFAULT_TRANSLATE  = "The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API."
DEFAULT_CLASSIFY   = "Hugging Face's Transformers library is amazing!"
DEFAULT_LABELS     = "technology, sports, politics"
DEFAULT_GENERATE   = "If there's one thing I want you to remember about using HuggingFace pipelines, it's"
DEFAULT_IMAGE      = "A class of students learning AI engineering in a vibrant pop-art style"
DEFAULT_TTS        = "Hi to an artificial intelligence engineer, on the way to mastery!"


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def fn_sentiment(text, use_multilingual):
    from transformers import pipeline
    if use_multilingual:
        model = load_once("sentiment_multi", lambda: pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device="cuda",
        ))
    else:
        model = load_once("sentiment_default", lambda: pipeline(
            "sentiment-analysis", device="cuda"
        ))
    result = model(text)[0]
    return result["label"], f"{result['score']:.4f}"


def fn_ner(text):
    from transformers import pipeline
    ner = load_once("ner", lambda: pipeline("ner", aggregation_strategy="simple", device="cuda"))
    entities = ner(text)
    return [[e["word"], e["entity_group"], f"{e['score']:.4f}"] for e in entities]


def fn_qa(question, context):
    from transformers import pipeline
    qa = load_once("qa", lambda: pipeline("question-answering", device="cuda"))
    result = qa(question=question, context=context)
    return result["answer"], f"{result['score']:.4f}"


def fn_summarize(text, max_len, min_len):
    from transformers import pipeline
    summarizer = load_once("summarizer", lambda: pipeline("summarization", device="cuda"))
    result = summarizer(text, max_length=int(max_len), min_length=int(min_len), do_sample=False)
    return result[0]["summary_text"]


def fn_translate(text, lang_pair):
    from transformers import pipeline
    task_map = {
        "EN → FR": ("translation_en_to_fr", None),
        "EN → ES": ("translation_en_to_es", "Helsinki-NLP/opus-mt-en-es"),
    }
    task, model_name = task_map[lang_pair]
    key = f"translate_{lang_pair}"
    kwargs = {"device": "cuda"}
    if model_name:
        kwargs["model"] = model_name
    # capture locals for the lambda
    _task, _kwargs = task, kwargs
    translator = load_once(key, lambda: pipeline(_task, **_kwargs))
    return translator(text)[0]["translation_text"]


def fn_classify(text, labels_csv):
    from transformers import pipeline
    labels = [l.strip() for l in labels_csv.split(",") if l.strip()]
    classifier = load_once("classifier", lambda: pipeline("zero-shot-classification", device="cuda"))
    result = classifier(text, candidate_labels=labels)
    return [[label, f"{score:.4f}"] for label, score in zip(result["labels"], result["scores"])]


def fn_generate(prompt, max_new_tokens):
    from transformers import pipeline
    generator = load_once("generator", lambda: pipeline("text-generation", device="cuda"))
    result = generator(prompt, max_new_tokens=int(max_new_tokens), truncation=True)
    return result[0]["generated_text"]


def fn_image(prompt, steps):
    from diffusers import AutoPipelineForText2Image

    def _load():
        return AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

    pipe = load_once("image_pipe", _load)
    image = pipe(prompt=prompt, num_inference_steps=int(steps), guidance_scale=0.0).images[0]
    return image


def fn_tts(text):
    from transformers import pipeline as hf_pipeline
    from huggingface_hub import hf_hub_download
    import soundfile as sf

    synthesiser = load_once(
        "tts",
        lambda: hf_pipeline("text-to-speech", "microsoft/speecht5_tts", device="cuda"),
    )

    def _load_embedding():
        # load_dataset no longer supports trust_remote_code for this dataset;
        # read xvectors directly from the zip file instead.
        zip_path = hf_hub_download(
            repo_id="matthijs/cmu-arctic-xvectors",
            filename="spkrec-xvect.zip",
            repo_type="dataset",
        )
        with zipfile.ZipFile(zip_path) as z:
            npy_files = sorted(f for f in z.namelist() if f.endswith(".npy"))
            idx = min(7306, len(npy_files) - 1)
            with z.open(npy_files[idx]) as f:
                xvector = np.load(f)
        return torch.tensor(xvector).unsqueeze(0)

    speaker_embedding = load_once("speaker_emb", _load_embedding)
    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    output_path = str(OUTPUT_DIR / "output_3_2_speech.wav")
    sf.write(output_path, speech["audio"], speech["sampling_rate"])
    return output_path


# ============================================================================
# GRADIO UI
# ============================================================================
def build_ui():
    with gr.Blocks(title="HuggingFace Pipelines — Local RTX 3060") as demo:
        gr.Markdown("# HuggingFace Pipelines — Local RTX 3060\nAdapted from `week_3_day_2_pipelines.ipynb`")

        with gr.Tabs():

            # ----------------------------------------------------------------
            with gr.Tab("1. Sentiment Analysis"):
                with gr.Row():
                    with gr.Column():
                        sa_text  = gr.Textbox(label="Text", value=DEFAULT_SENTIMENT, lines=3)
                        sa_multi = gr.Checkbox(label="Use multilingual model (nlptown)", value=False)
                        sa_btn   = gr.Button("Analyse", variant="primary")
                    with gr.Column():
                        sa_label = gr.Textbox(label="Label")
                        sa_score = gr.Textbox(label="Score")
                sa_btn.click(fn_sentiment, inputs=[sa_text, sa_multi], outputs=[sa_label, sa_score])

            # ----------------------------------------------------------------
            with gr.Tab("2. Named Entity Recognition"):
                with gr.Row():
                    with gr.Column():
                        ner_text = gr.Textbox(label="Text", value=DEFAULT_NER, lines=3)
                        ner_btn  = gr.Button("Recognize", variant="primary")
                    with gr.Column():
                        ner_out  = gr.DataFrame(headers=["Word", "Entity", "Score"], label="Entities")
                ner_btn.click(fn_ner, inputs=[ner_text], outputs=[ner_out])

            # ----------------------------------------------------------------
            with gr.Tab("3. Question Answering"):
                with gr.Row():
                    with gr.Column():
                        qa_q   = gr.Textbox(label="Question", value=DEFAULT_QA_Q)
                        qa_c   = gr.Textbox(label="Context", value=DEFAULT_QA_C, lines=3)
                        qa_btn = gr.Button("Answer", variant="primary")
                    with gr.Column():
                        qa_ans   = gr.Textbox(label="Answer")
                        qa_score = gr.Textbox(label="Confidence")
                qa_btn.click(fn_qa, inputs=[qa_q, qa_c], outputs=[qa_ans, qa_score])

            # ----------------------------------------------------------------
            with gr.Tab("4. Summarization"):
                with gr.Row():
                    with gr.Column():
                        sum_text = gr.Textbox(label="Text", value=DEFAULT_SUMMARY, lines=6)
                        with gr.Row():
                            sum_max = gr.Slider(20, 200, value=50, step=5, label="Max length")
                            sum_min = gr.Slider(5, 100, value=25, step=5, label="Min length")
                        sum_btn = gr.Button("Summarize", variant="primary")
                    with gr.Column():
                        sum_out = gr.Textbox(label="Summary", lines=4)
                sum_btn.click(fn_summarize, inputs=[sum_text, sum_max, sum_min], outputs=[sum_out])

            # ----------------------------------------------------------------
            with gr.Tab("5. Translation"):
                with gr.Row():
                    with gr.Column():
                        tr_text = gr.Textbox(label="Text (English)", value=DEFAULT_TRANSLATE, lines=3)
                        tr_lang = gr.Dropdown(["EN → FR", "EN → ES"], value="EN → FR", label="Language pair")
                        tr_btn  = gr.Button("Translate", variant="primary")
                    with gr.Column():
                        tr_out = gr.Textbox(label="Translation", lines=3)
                tr_btn.click(fn_translate, inputs=[tr_text, tr_lang], outputs=[tr_out])

            # ----------------------------------------------------------------
            with gr.Tab("6. Zero-Shot Classification"):
                with gr.Row():
                    with gr.Column():
                        cl_text   = gr.Textbox(label="Text", value=DEFAULT_CLASSIFY, lines=3)
                        cl_labels = gr.Textbox(label="Candidate labels (comma-separated)", value=DEFAULT_LABELS)
                        cl_btn    = gr.Button("Classify", variant="primary")
                    with gr.Column():
                        cl_out = gr.DataFrame(headers=["Label", "Score"], label="Results")
                cl_btn.click(fn_classify, inputs=[cl_text, cl_labels], outputs=[cl_out])

            # ----------------------------------------------------------------
            with gr.Tab("7. Text Generation"):
                with gr.Row():
                    with gr.Column():
                        gen_prompt = gr.Textbox(label="Prompt", value=DEFAULT_GENERATE, lines=3)
                        gen_tokens = gr.Slider(20, 200, value=60, step=10, label="Max new tokens")
                        gen_btn    = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        gen_out = gr.Textbox(label="Generated text", lines=6)
                gen_btn.click(fn_generate, inputs=[gen_prompt, gen_tokens], outputs=[gen_out])

            # ----------------------------------------------------------------
            with gr.Tab("8. Image Generation"):
                with gr.Row():
                    with gr.Column():
                        img_prompt = gr.Textbox(label="Prompt", value=DEFAULT_IMAGE, lines=3)
                        img_steps  = gr.Slider(1, 8, value=4, step=1, label="Inference steps (SDXL Turbo)")
                        img_btn    = gr.Button("Generate Image", variant="primary")
                    with gr.Column():
                        img_out = gr.Image(label="Generated Image")
                img_btn.click(fn_image, inputs=[img_prompt, img_steps], outputs=[img_out])

            # ----------------------------------------------------------------
            with gr.Tab("9. Text-to-Speech"):
                with gr.Row():
                    with gr.Column():
                        tts_text = gr.Textbox(label="Text", value=DEFAULT_TTS, lines=3)
                        tts_btn  = gr.Button("Synthesise", variant="primary")
                    with gr.Column():
                        tts_out = gr.Audio(label="Speech output")
                tts_btn.click(fn_tts, inputs=[tts_text], outputs=[tts_out])

    return demo


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Exercise 3.2: HuggingFace Pipelines — Local RTX 3060 + Gradio")
    print("=" * 70)

    check_gpu()
    login_huggingface()

    demo = build_ui()
    demo.launch()
