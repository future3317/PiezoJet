<#
.SYNOPSIS
Refresh the experiment ledger after an already-running process exits.

.DESCRIPTION
This is used when ledger support was added after a long replay had already
started. It does not alter the replay or its artifacts; it only waits for the
parent process and then rebuilds the registry from persisted files.
#>
param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [string]$Python = 'D:\Anaconda\envs\EGNN\python.exe',
    [string]$Repository = 'E:\CODE\PiezoJet'
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $Repository
$env:PYTHONPATH = (Resolve-Path 'src').Path

Wait-Process -Id $ProcessId -ErrorAction SilentlyContinue
$OutputRoot = 'outputs\exposure_matched_direct_u_v2_conditioning'
$PhysicalComplete = @(
    Get-ChildItem -LiteralPath (Join-Path $OutputRoot 'physical') -Recurse -Filter 'dfpt_test.json' -File -ErrorAction SilentlyContinue
).Count
$DirectComplete = @(
    Get-ChildItem -LiteralPath (Join-Path $OutputRoot 'direct') -Recurse -Filter 'test.json' -File -ErrorAction SilentlyContinue
).Count
$Complete = ($PhysicalComplete -eq 12 -and $DirectComplete -eq 12)
$Status = if ($Complete) { 'completed' } else { 'incomplete_after_process_exit' }
$Detail = "watched PID $ProcessId exited; physical evaluations=$PhysicalComplete/12, direct evaluations=$DirectComplete/12"
[ordered]@{
    time_utc = [DateTime]::UtcNow.ToString('o')
    status = $Status
    detail = $Detail
} | ConvertTo-Json -Compress | Add-Content `
    -LiteralPath (Join-Path $OutputRoot 'experiment_status_history.jsonl') -Encoding utf8

& $Python -m piezojet.experiment_registry `
    --outputs outputs `
    --json outputs\EXPERIMENT_REGISTRY.json `
    --markdown EXPERIMENT_REGISTRY.md `
    --artifact-index outputs\EXPERIMENT_ARTIFACT_INDEX.jsonl
if ($LASTEXITCODE -ne 0) {
    throw 'Experiment registry refresh failed after watched process exited.'
}
