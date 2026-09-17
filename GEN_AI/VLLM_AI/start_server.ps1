# Start VLLM server with Qwen2.5-7B-Instruct

Write-Host "Starting VLLM Server with Qwen2.5-7B-Instruct..." -ForegroundColor Green
Write-Host "Server will run on http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Check if model exists locally
$model_path = "$PSScriptRoot\Qwen2.5-7B-Instruct"
if (Test-Path $model_path) {
    Write-Host "Using local model: $model_path" -ForegroundColor Green
    vllm serve $model_path --port 8000
}
else {
    Write-Host "Local model not found, using model from HF cache..." -ForegroundColor Yellow
    vllm serve qwen2.5-7b-instruct --port 8000
}
