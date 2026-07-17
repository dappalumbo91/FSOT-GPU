# Build FSOT attention CUDA DLL for Python ctypes
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$Nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$CuDir = Join-Path $Root "phase2_native_gpu\cuda"
$Vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

$lines = @(
  "@echo off"
  "call `"$Vcvars`""
  "set `"CUDA_PATH=$CudaRoot`""
  "set `"PATH=$CudaRoot\bin;$CudaRoot\nvvm\bin;%PATH%`""
  "cd /d `"$CuDir`""
  "`"$Nvcc`" -O3 -arch=sm_120 -shared -o fsot_attn_lib.dll fsot_attn_lib.cu"
  "if errorlevel 1 exit /b 1"
  "dir fsot_attn_lib.dll"
  "exit /b 0"
)
$bat = Join-Path $env:TEMP "fsot_build_dll.bat"
Set-Content $bat ($lines -join "`r`n") -Encoding ASCII
cmd /c $bat
exit $LASTEXITCODE
