# GPU experiment workflow

All heavy experiments must run outside Dropbox.  The default runner uses:

- Experiment root: `C:\Users\User\codex-experiments\edgellm-router`
- Virtualenv: `C:\Users\User\codex-experiments\venvs\edgellm-gpu`
- Hugging Face cache: `C:\Users\User\codex-experiments\hf-cache`
- Results: `C:\Users\User\codex-experiments\edgellm-router\results`

## Smoke test

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gpu_experiments_outside_dropbox.ps1 -Smoke
```

## Paper pilot

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gpu_experiments_outside_dropbox.ps1 `
  -Models "Qwen/Qwen3-0.6B:edge,Qwen/Qwen3-4B:cloud" `
  -PromptLimit 24 `
  -BatchSizes "1,2" `
  -OutputTokens "16,32" `
  -ReplayRequests 2000 `
  -ReplaySeeds 3
```

## Full run

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gpu_experiments_outside_dropbox.ps1 `
  -Models "Qwen/Qwen3-0.6B:edge,Qwen/Qwen3-4B:cloud,Qwen/Qwen3-14B:cloud" `
  -PromptLimit 96 `
  -BatchSizes "1,2,4" `
  -OutputTokens "32,64,128" `
  -ReplayRequests 10000 `
  -ReplaySeeds 10
```

The full run can take hours because it downloads model weights and collects
latency/power traces.  Generated figures are created by:

```powershell
$env:EDGELLM_EXPERIMENT_DIR="C:\Users\User\codex-experiments\edgellm-router\results"
& "C:\Users\User\codex-experiments\venvs\edgellm-gpu\Scripts\python.exe" generate_figures.py
```

## Gemini API defaults

API-backed validation uses these defaults unless overridden:

```powershell
$env:EDGELLM_EDGE_MODEL="gemini-3.1-flash-lite"
$env:EDGELLM_CLOUD_MODEL="gemini-3.1-pro-preview"
$env:EDGELLM_JUDGE_MODEL="gemini-3.1-pro-preview"
```

Set `GOOGLE_API_KEY` or `GOOGLE_API_KEYS` before running API validation.
Never store API keys in repository files.
