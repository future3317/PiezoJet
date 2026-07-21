param(
    [string]$Plan = "outputs/vnext_stage_a_hierarchical_fairness_n200_quick_v2/plan.json"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Python = "D:\Anaconda\envs\EGNN\python.exe"
$env:PYTHONPATH = Join-Path $Root "src"
$planPath = if ([IO.Path]::IsPathRooted($Plan)) { $Plan } else { Join-Path $Root $Plan }
$planDoc = Get-Content -Raw $planPath | ConvertFrom-Json
$rawSteps = $planDoc.steps
$stepCount = if ($null -eq $rawSteps) { 0 } elseif ($rawSteps -is [Array]) { $rawSteps.Length } else { 1 }
Write-Host "Loaded quick plan $planPath with $stepCount steps (type $($rawSteps.GetType().FullName))"
$cohortRoot = Split-Path -Parent $planPath
$logDir = Join-Path $cohortRoot "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

function Invoke-Step([object]$Step, [int]$Index) {
    $name = if ($Step.name) { $Step.name } else { $Step.architecture }
    $log = Join-Path $logDir ("{0:D2}_{1}.log" -f $Index, ($name -replace "[^A-Za-z0-9_.-]", "_"))
    $argv = @($Step.argv)
    $display = ($argv -join " ")
    Write-Host "[$(Get-Date -Format o)] START $name"
    Write-Host $display
    $savedErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $argv[0] $argv[1..($argv.Count - 1)] 2>&1 | Tee-Object -FilePath $log
    $ErrorActionPreference = $savedErrorAction
    if ($LASTEXITCODE -ne 0) { throw "Stage-A step failed ($LASTEXITCODE): $name" }
    Write-Host "[$(Get-Date -Format o)] DONE $name"
}

$steps = if ($rawSteps -is [Array]) { $rawSteps } else { @($rawSteps) }
if ($steps.Count -lt 2) { throw "Quick plan has too few executable steps (got $($steps.Count))" }
for ($i = 0; $i -lt $steps.Count; $i++) {
    if ($steps[$i].name -eq "register_and_compare_after_all_candidates_finish") { break }
    Invoke-Step $steps[$i] $i
}

Write-Host "[$(Get-Date -Format o)] Quick N=200 adjudication complete; registry refresh follows."
& $Python -m piezojet.experiment_registry 2>&1 | Tee-Object -FilePath (Join-Path $logDir "06_registry.log")
if ($LASTEXITCODE -ne 0) { throw "Registry refresh failed ($LASTEXITCODE)" }
