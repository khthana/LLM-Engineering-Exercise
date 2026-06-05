#!/usr/bin/env python3
"""
Exercise_3.1: Local GPU Diffusion & Text-to-Speech
Adapted from Week_3_Day_1_Colab.ipynb for local RTX 3060 execution
"""

import os
import torch
import subprocess
from datetime import datetime
from pathlib import Path

# ============================================================================
# 1. GPU DETECTION
# ============================================================================
def check_gpu():
    """Verify GPU is available and show GPU info."""
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
        print("nvidia-smi not found, but checking torch.cuda...")
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        return False


# ============================================================================
# 2. HUGGINGFACE LOGIN
# ============================================================================
def login_huggingface():
    """Login to HuggingFace using HF_TOKEN environment variable."""
    from huggingface_hub import login

    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("\n[WARN] HF_TOKEN not found in environment variables")
        print("   Set it with: $env:HF_TOKEN='hf_...' (PowerShell)")
        print("   Or: export HF_TOKEN='hf_...' (Bash)")
        print("   Skipping login, some models may require it...\n")
        return False

    try:
        login(hf_token, add_to_git_credential=True)
        print("[OK] Logged in to HuggingFace")
        return True
    except Exception as e:
        print(f"[ERROR] HuggingFace login failed: {e}")
        return False


# ============================================================================
# 3. SDXL TURBO (Fast, low VRAM)
# ============================================================================
def run_sdxl_turbo():
    """SDXL Turbo - fastest, smallest memory footprint."""
    print("\n" + "="*70)
    print("SDXL Turbo (4 steps, ~2GB VRAM)")
    print("="*70)

    try:
        from diffusers import StableDiffusionXLPipeline

        print("Loading model...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe.to("cuda")

        prompt = "A class of students learning AI engineering in a vibrant Studio Ghibli Inspired style"
        print(f"Prompt: {prompt}")

        image = pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0
        ).images[0]

        output_path = Path("output_sdxl_turbo.png")
        image.save(output_path)
        print(f"[OK] Saved to {output_path}")

        # Clean up
        del pipe
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[ERROR] Error: {e}")


# ============================================================================
# 4. SDXL BASE (Higher quality, more VRAM)
# ============================================================================
def run_sdxl_base():
    """SDXL Base - higher quality than Turbo, 30 steps."""
    print("\n" + "="*70)
    print("SDXL Base 1.0 (30 steps, ~8-10GB VRAM)")
    print("="*70)

    try:
        from diffusers import DiffusionPipeline

        print("Loading model...")
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        pipe.to("cuda")

        prompt = "A class of data scientists learning AI engineering in a vibrant high-energy Studio Ghibli Inspired style"
        print(f"Prompt: {prompt}")

        image = pipe(prompt=prompt, num_inference_steps=30).images[0]

        output_path = Path("output_sdxl_base.png")
        image.save(output_path)
        print(f"[OK] Saved to {output_path}")

        del pipe
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[ERROR] Error: {e}")


# ============================================================================
# 5. SDXL BASE + REFINER (Memory-optimized for RTX 3060)
# ============================================================================
def run_sdxl_base_refiner():
    """SDXL Base + Refiner with memory optimization for RTX 3060 (12GB VRAM)."""
    print("\n" + "="*70)
    print("SDXL Base + Refiner (Memory-optimized for 12GB VRAM)")
    print("="*70)

    try:
        from diffusers import DiffusionPipeline

        print("Loading base model...")
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )

        # Memory optimization for RTX 3060
        base.enable_attention_slicing()
        base.to("cuda")

        print("Loading refiner model...")
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.enable_attention_slicing()
        refiner.to("cuda")

        n_steps = 40
        high_noise_frac = 0.8
        prompt = "A class of data scientists learning AI engineering in a vibrant high-energy Studio Ghibli Inspired Style"

        print(f"Prompt: {prompt}")
        print(f"Running base model with {int(n_steps * high_noise_frac)} steps...")

        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        print(f"Running refiner with {int(n_steps * (1 - high_noise_frac))} steps...")
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        output_path = Path("output_sdxl_base_refiner.png")
        image.save(output_path)
        print(f"[OK] Saved to {output_path}")

        del base, refiner
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[ERROR] Error: {e}")


# ============================================================================
# 6. TEXT-TO-SPEECH
# ============================================================================
def run_text_to_speech():
    """Microsoft SpeechT5 - convert text to speech."""
    print("\n" + "="*70)
    print("Text-to-Speech (SpeechT5, low VRAM)")
    print("="*70)

    try:
        from transformers import pipeline
        from datasets import load_dataset
        import soundfile as sf

        print("Loading text-to-speech model...")
        synthesiser = pipeline(
            "text-to-speech",
            "microsoft/speecht5_tts",
            device='cuda'
        )

        print("Loading speaker embeddings...")
        embeddings_dataset = load_dataset(
            "matthijs/cmu-arctic-xvectors",
            split="validation",
            trust_remote_code=True
        )
        speaker_embedding = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0)

        text = "Hi to an artificial intelligence engineer, on the way to mastery!"
        print(f"Text: {text}")

        speech = synthesiser(
            text,
            forward_params={"speaker_embeddings": speaker_embedding}
        )

        output_path = Path("output_speech.wav")
        sf.write(str(output_path), speech["audio"], speech["sampling_rate"])
        print(f"[OK] Saved to {output_path}")

        del synthesiser
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[ERROR] Error: {e}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Exercise 3.1: Local GPU Diffusion & Text-to-Speech")
    print("Target: RTX 3060 (12GB VRAM)")
    print()

    # Check GPU
    if not check_gpu():
        print("\n[ERROR] No GPU detected. Some features will be slow or fail.")
        return

    # Login to HuggingFace
    login_huggingface()

    # Run models
    try:
        run_sdxl_turbo()
    except Exception as e:
        print(f"[ERROR] SDXL Turbo failed: {e}")

    try:
        run_sdxl_base()
    except Exception as e:
        print(f"[ERROR] SDXL Base failed: {e}")

    try:
        run_sdxl_base_refiner()
    except Exception as e:
        print(f"[ERROR] SDXL Base + Refiner failed: {e}")

    try:
        run_text_to_speech()
    except Exception as e:
        print(f"[ERROR] Text-to-Speech failed: {e}")

    print("\n" + "="*70)
    print("[OK] Exercise completed!")
    print("="*70)


if __name__ == "__main__":
    main()
