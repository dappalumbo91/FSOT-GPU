# Smoke-test Lean / Coq / Isabelle / F* availability for the experiment
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\set_env.ps1"

$report = [ordered]@{
  timestamp = (Get-Date).ToString("o")
  lean = $null
  lake = $null
  fstar = $null
  coqc = $null
  isabelle = $null
  overall_ok = $false
}

function Try-Cmd($name, $scriptBlock) {
  try {
    $out = & $scriptBlock 2>&1 | Out-String
    return @{ ok = $true; output = $out.Trim().Substring(0, [Math]::Min(400, $out.Trim().Length)) }
  } catch {
    return @{ ok = $false; output = $_.Exception.Message }
  }
}

$report.lake = Try-Cmd "lake" { lake --version }
$report.lean = Try-Cmd "lean" { lean --version }
$report.fstar = Try-Cmd "fstar" {
  $bin = Join-Path $env:FSOT_PORTABLE_TOOLCHAIN "fstar\bin\fstar.exe"
  & $bin --version
}
$report.coqc = Try-Cmd "coqc" {
  $bin = Join-Path $env:FSOT_PORTABLE_TOOLCHAIN "rocq\bin\coqc.exe"
  & $bin --version
}
# Isabelle --version can be slow; just test path exists
$isa = Join-Path $env:FSOT_PORTABLE_TOOLCHAIN "isabelle\bin\isabelle"
if (Test-Path $isa) {
  $report.isabelle = @{ ok = $true; output = "present: $isa" }
} else {
  $report.isabelle = @{ ok = $false; output = "missing" }
}

$report.overall_ok = (
  $report.lake.ok -and $report.fstar.ok -and $report.coqc.ok -and $report.isabelle.ok
)

$outDir = Join-Path $env:FSOT_EXPERIMENT_ROOT "results\phase0"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outFile = Join-Path $outDir "formal_toolchain_smoke.json"
($report | ConvertTo-Json -Depth 6) | Set-Content -Path $outFile -Encoding UTF8

Write-Host ""
Write-Host "=== Formal toolchain smoke ==="
Write-Host "lake:     $($report.lake.ok)"
Write-Host "lean:     $($report.lean.ok)"
Write-Host "fstar:    $($report.fstar.ok)"
Write-Host "coqc:     $($report.coqc.ok)"
Write-Host "isabelle: $($report.isabelle.ok)"
Write-Host "overall:  $($report.overall_ok)"
Write-Host "wrote:    $outFile"

if (-not $report.overall_ok) { exit 1 }
