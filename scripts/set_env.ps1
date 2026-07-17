# FSOT Formal-GPU experiment environment
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$PathsFile = Join-Path $Root "config\paths.json"
$Paths = Get-Content $PathsFile -Raw | ConvertFrom-Json

$env:FSOT_EXPERIMENT_ROOT = $Root
$env:FSOT_ARCHIVE_ROOT = $Paths.archive_root
$env:FSOT_LEAN_FULL = $Paths.fsot_lean_full
$env:FSOT_SRITE = $Paths.fsot_srite
$env:FSOT_PUBLIC_DATA = $Paths.public_data
$env:FSOT_PORTABLE_TOOLCHAIN = $Paths.portable_toolchain
$env:FSOT_EXTERNAL_DATA_ROOT = $Paths.public_data

# Prefer full CUDA 13.3 (has nvvm/cicc). Broken 13.2 stub remains on disk but not on PATH.
$Cuda13 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
if (Test-Path (Join-Path $Cuda13 "bin\nvcc.exe")) {
  $env:CUDA_PATH = $Cuda13
  $env:CUDA_PATH_V13_3 = $Cuda13
  $env:PATH = "$(Join-Path $Cuda13 'bin');$(Join-Path $Cuda13 'nvvm\bin');$env:PATH"
  $env:FSOT_CUDA_ROOT = $Cuda13
} else {
  Write-Warning "CUDA v13.3 not found at $Cuda13"
}

# Prefer archive formal tools on PATH for this session
$extra = @(
  (Join-Path $Paths.portable_toolchain "fstar\bin"),
  (Join-Path $Paths.portable_toolchain "rocq\bin"),
  (Join-Path $Paths.portable_toolchain "isabelle\bin"),
  (Join-Path $Paths.portable_toolchain "elan\bin")
)
foreach ($p in $extra) {
  if (Test-Path $p) {
    $env:PATH = "$p;$env:PATH"
  }
}

Write-Host "FSOT Formal-GPU env ready"
Write-Host "  EXPERIMENT: $env:FSOT_EXPERIMENT_ROOT"
Write-Host "  ARCHIVE:    $env:FSOT_ARCHIVE_ROOT"
Write-Host "  LEAN FULL:  $env:FSOT_LEAN_FULL"
Write-Host "  CUDA:       $env:FSOT_CUDA_ROOT"
