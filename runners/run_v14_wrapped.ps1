# Runs the V14 program with a fault handler and records how the process ended (spec §18: a silent
# death must leave a trace). Relaunch is the watchdog's job; this script runs the program once.
#   powershell -File runners\run_v14_wrapped.ps1 -Workers 12
param([int]$Workers = 12, [string]$Stage = "all")
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
$logDir = Join-Path $repo "results\v14\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "MMdd_HHmmss"
$log = Join-Path $logDir "run_${Stage}_wrapped_$stamp.log"
$wrap = Join-Path $logDir "wrapper_$stamp.log"
$env:PYTHONFAULTHANDLER = "1"
$env:OMP_NUM_THREADS = "1"
Add-Content $wrap "$(Get-Date -Format s) START stage=$Stage workers=$Workers log=$log ppid=$PID"
$p = Start-Process -FilePath (Join-Path $repo ".venv\Scripts\python.exe") -ArgumentList @("-X", "faulthandler", "-m", "runners.run_v14", "--stage", $Stage, "--workers", "$Workers") `
    -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
Add-Content $wrap "$(Get-Date -Format s) launched pid=$($p.Id)"
$p.WaitForExit()
Add-Content $wrap "$(Get-Date -Format s) EXIT pid=$($p.Id) code=$($p.ExitCode) stage=$Stage"
