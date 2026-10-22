# FSOT Formal-GPU — full stack run (no gatekeeping)
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\set_env.ps1"

# Local locations: env vars win; defaults are relative to this repo (siblings = repos cloned next to it)
$Parent = Split-Path -Parent $Root
if (-not $env:FSOT_ARCHIVE_ROOT) { $env:FSOT_ARCHIVE_ROOT = Join-Path $Root "data_external\physical_archive" }
if (-not $env:FSOT_COMPUTE_PATH) { $env:FSOT_COMPUTE_PATH = Join-Path $Parent "FSOT-2.1-Lean\vendor\fsot_compute.py" }
if (-not $env:FSOT_LLM_ROOT) { $env:FSOT_LLM_ROOT = Join-Path $Parent "fsot-2.1-llm" }

Write-Host "======== 1) CUDA trinary pack (native) ========"
$cuDir = Join-Path $Root "phase2_native_gpu\cuda"
$exe = Join-Path $cuDir "trinary_pack_test.exe"
Push-Location $cuDir
nvcc -O3 -o trinary_pack_test.exe trinary_pack_main.cu 2>&1
if (Test-Path $exe) {
  & $exe 2>&1
} else {
  Write-Host "nvcc build failed"
}
Pop-Location

Write-Host "======== 2) FSOT GPU engine (kernel port + train) ========"
python (Join-Path $Root "phase2_native_gpu\python\fsot_gpu_engine.py") 2>&1

Write-Host "======== 3) Lean formal still green ========"
Push-Location (Join-Path $Root "phase1_formal_gpu\lean")
lake build 2>&1
Pop-Location

Write-Host "======== 4) F* boot ========"
$fstar = if ($env:FSTAR_EXE) { $env:FSTAR_EXE } else { Join-Path $env:FSOT_ARCHIVE_ROOT "07_Portable-Toolchain\fstar\bin\fstar.exe" }
& $fstar (Join-Path $Root "phase1_formal_gpu\fstar\FSOTGpuBoot.fst") 2>&1

Write-Host "======== DONE ========"
