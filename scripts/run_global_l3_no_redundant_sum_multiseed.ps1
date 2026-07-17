<#
.SYNOPSIS
Validation-only three-seed replay of the global-l3 independent-U candidate.

.DESCRIPTION
Runs the maintained train1603/val10 protocol with complete-shell graphs,
teacher-U AdamW continuity, first-order U/V consistency, and no redundant
electronic+ionic branch-sum optimization.  It never evaluates or loads test20
labels.  Seed42 is retained in the adjudication cohort; the default invocation
runs the two missing replication seeds.
#>
param(
    [int[]]$Seeds = @(7, 1729),
    [string]$Python = 'D:\Anaconda\envs\EGNN\python.exe',
    [string]$Config = 'config.yaml',
    [string]$Split = 'data\processed\full_corpus_multitask_train1603_v1.json',
    [string]$OutputRoot = 'outputs\global_l3_no_redundant_sum_multiseed_v1'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = (Resolve-Path 'src').Path

foreach ($Seed in $Seeds) {
    $output = Join-Path $OutputRoot "factorized_seed$Seed"
    if (Test-Path -LiteralPath (Join-Path $output 'summary.json')) {
        throw "Refusing to overwrite completed cohort: $output"
    }
    & $Python -m piezojet.train --config $Config --splits-file $Split `
        --seed $Seed --output-dir $output --factor-pretrain-patience 0 `
        --early-stopping-patience 0 --checkpoint-selection-metric loss
    if ($LASTEXITCODE -ne 0) {
        throw "Global-l3 no-redundant-sum validation replay failed for seed $Seed"
    }
}
