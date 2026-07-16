<#
.SYNOPSIS
Run the formula-disjoint full-GMTNet / strict-DFPT multitask replay.

.DESCRIPTION
The training pool contains every GMTNet material whose formula is disjoint
from the frozen strict-completion validation/test formulas.  Only records with
JARVIS DFPT labels activate factor and branch losses.  Each learned condition
uses the same full-train structural checkpoint, 50 direct-factor updates and
100 joint updates (25 updates followed by validation at each of two/four
epochs).  Test is read only after validation-selected training ends.
#>
param(
    [int[]]$Seeds = @(42, 7, 1729),
    [string]$Python = 'D:\Anaconda\envs\EGNN\python.exe',
    [string]$Config = 'config.yaml',
    [string]$Split = 'data\processed\full_corpus_multitask_train1603_v1.json',
    [string]$OutputRoot = 'outputs\full_corpus_multitask_independent_response_v1'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = (Resolve-Path 'src').Path

if (-not (Test-Path -LiteralPath $Split)) {
    $canonical = Get-Content -LiteralPath 'data\processed\canonical_datasets.json' -Raw | ConvertFrom-Json
    & $Python -m piezojet.build_full_corpus_multitask_split `
        --data-root $canonical.roles.gmtnet_source `
        --strict-split-file $canonical.roles.strict_split `
        --output $Split
    if ($LASTEXITCODE -ne 0) { throw 'Full-corpus formula-disjoint split construction failed' }
}

$pretrain = "$OutputRoot\pretrain_full_corpus_seed42"
& $Python -m piezojet.pretrain --config $Config --splits-file $Split --output-dir $pretrain --epochs 3 --updates 100
if ($LASTEXITCODE -ne 0) { throw 'Full-corpus structural pretraining failed' }

& $Python scripts\evaluate_constant_baselines.py --config $Config --splits-file $Split --output "$OutputRoot\constant_baselines.json"
if ($LASTEXITCODE -ne 0) { throw 'Full-corpus constant-baseline evaluation failed' }

foreach ($Seed in $Seeds) {
    $factorized = "$OutputRoot\factorized_seed$Seed"
    & $Python -m piezojet.train --config $Config --splits-file $Split --seed $Seed --output-dir $factorized `
        --pretrained-encoder "$pretrain\best_encoder.pt" `
        --factor-pretrain-epochs 2 --factor-train-updates-per-epoch 25 `
        --epochs 4 --train-updates-per-epoch 25 --factor-pretrain-patience 0 --early-stopping-patience 0
    if ($LASTEXITCODE -ne 0) { throw "Full-corpus factorized training failed for seed $Seed" }
    & $Python -m piezojet.evaluate_dfpt --checkpoint "$factorized\loss_best.pt" --splits-file $Split --split test --device auto --output "$factorized\dfpt_test.json"
    if ($LASTEXITCODE -ne 0) { throw "Full-corpus factorized evaluation failed for seed $Seed" }

    $direct = "$OutputRoot\direct_seed$Seed"
    & $Python -m piezojet.train_direct_baseline --config $Config --splits-file $Split --seed $Seed --output-dir $direct `
        --pretrained-encoder "$pretrain\best_encoder.pt" --epochs 4 --updates-per-epoch 25
    if ($LASTEXITCODE -ne 0) { throw "Full-corpus direct control failed for seed $Seed" }
    & $Python -m piezojet.evaluate_direct_baseline --checkpoint "$direct\loss_best.pt" --splits-file $Split --split test --device auto --output "$direct\test.json"
    if ($LASTEXITCODE -ne 0) { throw "Full-corpus direct evaluation failed for seed $Seed" }
}
