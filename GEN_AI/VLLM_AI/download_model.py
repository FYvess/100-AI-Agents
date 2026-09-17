#!/usr/bin/env python3
"""Download Qwen2.5-7B-Instruct model for VLLM."""

import subprocess
import sys
from pathlib import Path

def install_pkg(pkg):
    """Install a package if not already present."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def download_model():
    """Download Qwen2.5-7B-Instruct from Hugging Face."""
    print("\n" + "="*60)
    print("Downloading Qwen2.5-7B-Instruct for VLLM")
    print("="*60)
    print("This may take 5-15 minutes depending on your internet speed.\n")
    
    # Ensure huggingface_hub is installed
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface-hub...")
        install_pkg("huggingface-hub")
        from huggingface_hub import snapshot_download
    
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    local_dir = Path(__file__).parent / "Qwen2.5-7B-Instruct"
    
    print(f"Model: {model_id}")
    print(f"Destination: {local_dir}\n")
    
    try:
        path = snapshot_download(model_id, local_dir=str(local_dir))
        print("\n" + "="*60)
        print("✅ Model downloaded successfully!")
        print("="*60)
        print(f"\nModel location: {path}")
        print("\nNext step: Start VLLM server with:")
        print(f"  vllm serve {local_dir} --port 8000")
        print("\nOr use the default model name:")
        print("  vllm serve qwen2.5-7b-instruct")
        return 0
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Check disk space (model is ~14GB)")
        print("  3. Try manually: huggingface-cli download Qwen/Qwen2.5-7B-Instruct")
        return 1

if __name__ == "__main__":
    sys.exit(download_model())
