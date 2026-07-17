# Build FSOT CUDA kernels with CUDA 13.3 + MSVC (RTX 5070 sm_120)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$Nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$Cicc = Join-Path $CudaRoot "nvvm\bin\cicc.exe"
$CuDir = Join-Path $Root "phase2_native_gpu\cuda"
$Vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $Nvcc)) { throw "nvcc missing: $Nvcc" }
if (-not (Test-Path $Cicc)) { throw "cicc missing: $Cicc - CUDA toolkit incomplete" }
if (-not (Test-Path $Vcvars)) { throw "MSVC vcvars64 missing: $Vcvars" }

Write-Host "Using CUDA: $CudaRoot"
& $Nvcc --version

$lines = @(
  "@echo off"
  "setlocal"
  "call `"$Vcvars`""
  "set `"CUDA_PATH=$CudaRoot`""
  "set `"CUDA_PATH_V13_3=$CudaRoot`""
  "set `"PATH=$CudaRoot\bin;$CudaRoot\nvvm\bin;%PATH%`""
  "cd /d `"$CuDir`""
  "`"$Nvcc`" -O3 -arch=sm_120 -o trinary_pack_test.exe trinary_pack_main.cu"
  "if errorlevel 1 exit /b 1"
  "trinary_pack_test.exe"
  "exit /b %ERRORLEVEL%"
)
$batPath = Join-Path $env:TEMP "fsot_build_cuda.bat"
Set-Content -Path $batPath -Value ($lines -join "`r`n") -Encoding ASCII
cmd /c $batPath
$code = $LASTEXITCODE
if ($code -ne 0) { throw "CUDA kernel build/run failed exit=$code" }
Write-Host "CUDA kernels OK"
exit 0
