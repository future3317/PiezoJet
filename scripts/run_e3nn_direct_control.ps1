<#!
.SYNOPSIS
Run the independent PBC e3nn direct-tensor control on the frozen panel.

.DESCRIPTION
The e3nn encoder is structurally pretrained once on the same train-only
subset used by the Cartesian checkpoint. Its three direct-head fine-tunes use
the same 100-update budget and validation-loss checkpoint rule.
#>
param(
    [int[]]$Seeds = @(42, 7, 1729),
    [string]$Python = 'D:\Anaconda\envs\EGNN\python.exe',
    [string]$Config = 'config.yaml',
    [string]$Split = 'data\processed\strict_completion_benchmark_train_v10_full_public.json',
    # Structural pretraining has no response labels, but it must nevertheless
    # use precisely the same train-only structure set as the direct control.
    # Keeping this explicit prevents a capacity confound when the strict
    # training panel grows.
    [string]$PretrainSplit = 'data\processed\strict_completion_benchmark_train_v10_full_public.json'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = (Resolve-Path 'src').Path
$pretrain = 'outputs\e3nn_direct_control_current\pretrain_seed42'
& $Python -m piezojet.pretrain_e3nn --config $Config --splits-file $PretrainSplit --output-dir $pretrain --seed 42 --batch-size 8 --accumulate-to-one-update
if ($LASTEXITCODE -ne 0) { throw 'e3nn structural pretraining failed' }

foreach ($Seed in $Seeds) {
    $output = "outputs\e3nn_direct_control_current\seed$Seed"
    & $Python -m piezojet.train_direct_baseline --family e3nn --config $Config --splits-file $Split --seed $Seed --pretrained-encoder "$pretrain\best_encoder.pt" --output-dir $output --epochs 100 --batch-size 8 --accumulate-to-one-update
    if ($LASTEXITCODE -ne 0) { throw "e3nn direct training failed for seed $Seed" }
    & $Python -m piezojet.evaluate_direct_baseline --checkpoint "$output\loss_best.pt" --splits-file $Split --split test --device auto --output "$output\test.json"
    if ($LASTEXITCODE -ne 0) { throw "e3nn direct evaluation failed for seed $Seed" }
}
