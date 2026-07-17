<#
.SYNOPSIS
Validation-only matched direct-total control for the global-l3 candidate.

.DESCRIPTION
Runs the Cartesian direct piezo tensor baseline for the same 4,961 macro-train
records, formula-disjoint val10 panel, graph construction, structural
checkpoint, ten complete passes, seeds, and minimum-validation-loss selection
used by the global-l3 physical candidate.  It never invokes a test evaluator.
#>
param(
    [int[]]$Seeds = @(42, 7, 1729),
    [int]$Epochs = 10,
    [string]$Python = 'D:\Anaconda\envs\EGNN\python.exe',
    [string]$Config = 'config.yaml',
    [string]$Split = 'data\processed\full_corpus_multitask_train1603_v1.json',
    [string]$OutputRoot = 'outputs\global_l3_matched_direct_validation_v1'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = (Resolve-Path 'src').Path

if (Test-Path -LiteralPath $OutputRoot) {
    throw "Fresh matched-direct output root required: $OutputRoot"
}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$sourceFiles = @(
    $Config,
    $Split,
    'scripts\run_global_l3_matched_direct_validation.ps1',
    'src\piezojet\train_direct_baseline.py',
    'src\piezojet\baselines.py',
    'src\piezojet\model.py',
    'src\piezojet\data.py',
    'src\piezojet\tensor_ops.py',
    'src\piezojet\elastic_dielectric_ops.py',
    'src\piezojet\projectors.py',
    'src\piezojet\project_config.py'
)
$sourceEntries = foreach ($sourceFile in $sourceFiles) {
    $resolved = Resolve-Path -LiteralPath $sourceFile
    $item = Get-Item -LiteralPath $resolved
    [ordered]@{
        path = (Resolve-Path -Relative -LiteralPath $resolved)
        bytes = $item.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolved).Hash.ToLowerInvariant()
    }
}
$manifest = [ordered]@{
    schema = 1
    captured_at = (Get-Date).ToString('o')
    git_head = (& git rev-parse HEAD).Trim()
    git_worktree_dirty = [bool](& git status --porcelain)
    purpose = 'Exact source snapshot for the three-seed global-l3 matched direct validation control'
    scientific_boundary = 'Validation10 only; frozen test20 is never loaded or evaluated'
    files = @($sourceEntries)
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (
    Join-Path $OutputRoot 'training_source_manifest.json'
) -Encoding UTF8

foreach ($Seed in $Seeds) {
    $output = Join-Path $OutputRoot "direct_seed$Seed"
    if (Test-Path -LiteralPath $output) {
        throw "Fresh direct-control output directory required: $output"
    }
    & $Python -m piezojet.train_direct_baseline --config $Config `
        --splits-file $Split --seed $Seed --epochs $Epochs --family cartesian `
        --output-dir $output
    if ($LASTEXITCODE -ne 0) {
        throw "Matched direct validation control failed for seed $Seed"
    }
}
