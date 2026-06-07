param(
  [string]$ExperimentRoot = "C:\Users\User\codex-experiments\edgellm-router",
  [string]$VenvPath = "C:\Users\User\codex-experiments\venvs\edgellm-gpu",
  [string]$HfHome = "C:\Users\User\codex-experiments\hf-cache",
  [string]$Models = "Qwen/Qwen3-4B:edge,Qwen/Qwen3-14B:cloud",
  [int]$PromptLimit = 96,
  [string]$BatchSizes = "1,2,4",
  [string]$OutputTokens = "32,64,128",
  [int]$ReplayRequests = 10000,
  [int]$ReplaySeeds = 10,
  [switch]$Smoke
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $ExperimentRoot "source"
$results = Join-Path $ExperimentRoot "results"

New-Item -ItemType Directory -Force -Path $ExperimentRoot, $src, $results, $HfHome | Out-Null

$files = @(
  "edge_simulator.py",
  "experiments_extended_prompt_bench.py",
  "experiments_gpu_latency_trace.py",
  "experiments_trace_replay_routing.py",
  "experiments_energy_cost.py"
)

foreach ($file in $files) {
  Copy-Item -LiteralPath (Join-Path $repo $file) -Destination (Join-Path $src $file) -Force
}

$env:EDGELLM_EXPERIMENT_DIR = $results
$env:HF_HOME = $HfHome
$env:HF_HUB_CACHE = Join-Path $HfHome "hub"
$env:TRANSFORMERS_CACHE = $HfHome

$python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  & "C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv $VenvPath
  if ($LASTEXITCODE -ne 0) { throw "Failed to create venv" }
  & $python -m pip install --upgrade pip setuptools wheel
  if ($LASTEXITCODE -ne 0) { throw "Failed to update pip" }
}

if ($Smoke) {
  $Models = "Qwen/Qwen3-0.6B:edge,Qwen/Qwen3-0.6B:cloud"
  $PromptLimit = 4
  $BatchSizes = "1"
  $OutputTokens = "8"
}

Push-Location $src
try {
  & $python experiments_extended_prompt_bench.py --size 900 --output-dir $results
  if ($LASTEXITCODE -ne 0) { throw "Prompt bench generation failed" }
  $modelArgs = @()
  foreach ($spec in $Models.Split(",")) {
    $modelArgs += @("--model", $spec)
  }
  & $python experiments_gpu_latency_trace.py `
    @modelArgs `
    --prompt-csv (Join-Path $results "extended_prompt_bench.csv") `
    --prompt-limit $PromptLimit `
    --batch-sizes $BatchSizes `
    --output-tokens $OutputTokens `
    --output-dir $results
  if ($LASTEXITCODE -ne 0) { throw "GPU latency trace failed" }
  & $python experiments_trace_replay_routing.py `
    --trace-csv (Join-Path $results "gpu_latency_trace_raw.csv") `
    --output-dir $results `
    --n-requests $(if ($Smoke) { 200 } else { $ReplayRequests }) `
    --seeds $(if ($Smoke) { 1 } else { $ReplaySeeds })
  if ($LASTEXITCODE -ne 0) { throw "Trace replay failed" }
  & $python experiments_energy_cost.py `
    --trace-csv (Join-Path $results "gpu_latency_trace_raw.csv") `
    --replay-csv (Join-Path $results "gpu_trace_replay_raw.csv") `
    --output-dir $results
  if ($LASTEXITCODE -ne 0) { throw "Energy summarisation failed" }
}
finally {
  Pop-Location
}

Write-Host "GPU experiments complete. Results: $results"
