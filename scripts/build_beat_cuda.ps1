# Build FSOT beat-CUDA kernel suite (sm_120)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$Nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$CuDir = Join-Path $Root "phase2_native_gpu\cuda"
$Vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

$lines = @(
  "@echo off"
  "setlocal"
  "call `"$Vcvars`""
  "set `"CUDA_PATH=$CudaRoot`""
  "set `"PATH=$CudaRoot\bin;$CudaRoot\nvvm\bin;%PATH%`""
  "cd /d `"$CuDir`""
  "`"$Nvcc`" -O3 -arch=sm_120 -o fsot_beat_cuda.exe fsot_beat_cuda.cu"
  "if errorlevel 1 exit /b 1"
  "fsot_beat_cuda.exe"
  "exit /b %ERRORLEVEL%"
)
$bat = Join-Path $env:TEMP "fsot_build_beat.bat"
Set-Content -Path $bat -Value ($lines -join "`r`n") -Encoding ASCII
cmd /c $bat
exit $LASTEXITCODE
