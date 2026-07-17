<#
.SYNOPSIS
Run matched same-ID 1/8/32 operator-learning capacity probes.

.DESCRIPTION
Compares the unchanged direct-factor objective with the preregistered,
gradient-balanced direct-operator bundle.  It reads only the declared strict
training IDs, disables the historical inverse/resolvent-action loss and the
homogeneous normal equation, and never evaluates frozen validation/test IDs.
Every run uses a fresh directory and retains negative or failed attempts.
#>
param(
    [string]$Python = 'D:\Anaconda\envs\EGNN\python.exe',
    [string]$Config = 'config.yaml',
    [string]$IDsRoot = 'data\processed\capacity_probe_ids',
    [string]$OutputRoot = 'outputs\operator_learning_capacity_v1',
    [int]$Seed = 42,
    [int]$FactorEpochs = 100,
    [int]$DisplacementEpochs = 100,
    [int]$JointEpochs = 300,
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = (Resolve-Path 'src').Path
$env:PYTHONWARNINGS = 'ignore::UserWarning:ast,ignore::DeprecationWarning:torch.jit._script'
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

foreach ($variant in @('baseline', 'operator')) {
    foreach ($count in @(1, 8, 32)) {
        $ids = Join-Path $IDsRoot "samples$count`_ids.json"
        if (!(Test-Path -LiteralPath $ids)) { throw "Missing capacity IDs: $ids" }
        $run = Join-Path $OutputRoot "$variant\samples$count"
        if (Test-Path -LiteralPath (Join-Path $run 'overfit_dfpt_train.json')) {
            Write-Output "Skipping completed $variant samples$count"
            continue
        }
        if (Test-Path -LiteralPath $run) {
            throw "Partial run exists at $run; preserve it and use a fresh output root."
        }
        $arguments = @(
            '-m', 'piezojet.train', '--config', $Config,
            '--seed', "$Seed", '--material-ids-file', $ids,
            '--material-ids-split', 'same', '--allow-noninductive-overfit',
            '--batch-size', "$count", '--factor-pretrain-epochs', "$FactorEpochs",
            '--displacement-pretrain-epochs', "$DisplacementEpochs",
            '--displacement-consistency-warmup-epochs', "$JointEpochs",
            '--displacement-consistency-ramp-epochs', '0', '--epochs', "$JointEpochs",
            '--early-stopping-patience', '0', '--device', $Device,
            '--output-dir', $run
        )
        if ($variant -eq 'operator') { $arguments += '--operator-learning-capacity' }
        & $Python @arguments
        if ($LASTEXITCODE -ne 0) { throw "$variant samples$count training failed" }

        & $Python -m piezojet.evaluate_dfpt --checkpoint (Join-Path $run 'loss_best.pt') `
            --material-ids-file $ids --material-ids-split same --split train `
            --device $Device --output (Join-Path $run 'overfit_dfpt_train.json') `
            --bootstrap-resamples 1
        if ($LASTEXITCODE -ne 0) { throw "$variant samples$count evaluation failed" }
    }
}

& $Python -m piezojet.summarize_operator_learning_capacity `
    --root $OutputRoot --output (Join-Path $OutputRoot 'summary.json')
if ($LASTEXITCODE -ne 0) { throw 'Operator-learning capacity summary failed.' }

& $Python -m piezojet.experiment_registry
if ($LASTEXITCODE -ne 0) { throw 'Experiment registry refresh failed.' }
