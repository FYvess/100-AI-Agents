# Download and prepare Qwen2.5-7B-Instruct for VLLM
# Run this before starting VLLM server

Write-Host "Downloading Qwen2.5-7B-Instruct model for VLLM..." -ForegroundColor Green
Write-Host "This may take 5-15 minutes depending on your internet speed." -ForegroundColor Yellow

# Option 1: Using huggingface-cli (recommended if installed)
try {
    Write-Host "`nAttempting download via huggingface-cli..." -ForegroundColor Cyan
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./Qwen2.5-7B-Instruct
    Write-Host "✅ Model downloaded successfully!" -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "huggingface-cli not found, trying alternative method..." -ForegroundColor Yellow
}

# Option 2: Using pip to install huggingface-hub
Write-Host "`nInstalling huggingface-hub..." -ForegroundColor Cyan
pip install huggingface-hub

Write-Host "`nDownloading Qwen2.5-7B-Instruct..." -ForegroundColor Cyan
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='./Qwen2.5-7B-Instruct')"

if ($?) {
    Write-Host "`n✅ Model downloaded successfully!" -ForegroundColor Green
    Write-Host "`nNext step: Start VLLM with:" -ForegroundColor Cyan
    Write-Host "  vllm serve ./Qwen2.5-7B-Instruct --port 8000" -ForegroundColor Yellow
}
else {
    Write-Host "`n❌ Download failed. Check your internet connection." -ForegroundColor Red
    exit 1
}
