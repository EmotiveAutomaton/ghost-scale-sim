# V15 launcher. ALWAYS module form.
#
# The sibling project's tools/orphan_sweep.ps1 runs at every gear-script startup (about every
# thirty minutes) and taskkills any python whose command line matches `runners[\/]run_` and whose
# root pid is not in its lock files. That killed V14's runner seven times, cost that program its
# first 24-hour window, and retro-explains four of V13's "unexplained" silent deaths -- no
# traceback, empty stderr, no fault events, roughly 25 to 32 minutes after each launch.
#
# `python -X faulthandler -m runners.run_v15` does not match that pattern. Attack X24 reads this
# file and checks the module form is still here, so the fix cannot rot quietly.
#
# Each run writes its own wrapper log. A single shared log is held open by anything tailing it,
# and Add-Content then throws an IOException that looks like a runner failure and is not.

param(
    [string]$Stage = "all",
    [int]$Workers = 0,
    [string]$Tier = ""
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
$py = Join-Path $repo ".venv\Scripts\python.exe"
$logDir = Join-Path $repo "results\v15\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$wrap = Join-Path $logDir "wrapper_$stamp.log"
$out = Join-Path $logDir "run_$stamp.out.log"
$err = Join-Path $logDir "run_$stamp.err.log"

Add-Content -Path $wrap -Value "$(Get-Date -Format s) launching stage=$Stage workers=$Workers tier=$Tier"

$argList = @("-X", "faulthandler", "-m", "runners.run_v15", "--stage", $Stage)
if ($Workers -gt 0) { $argList += @("--workers", "$Workers") }
if ($Tier -ne "")   { $argList += @("--tier", $Tier) }

$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:CUDA_VISIBLE_DEVICES = ""

$p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $repo `
    -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -NoNewWindow
Add-Content -Path $wrap -Value "$(Get-Date -Format s) pid=$($p.Id)"
$p.WaitForExit()
Add-Content -Path $wrap -Value "$(Get-Date -Format s) exited code=$($p.ExitCode)"
exit $p.ExitCode
