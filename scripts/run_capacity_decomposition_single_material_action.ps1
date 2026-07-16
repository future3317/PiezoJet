<#
.SYNOPSIS
Run the preregistered eight single-material Phi-action capacity probes.

.DESCRIPTION
Each strict-train material gets a separate, fresh, noninductive fit with the
already fixed (not tuned) response-operator action loss weight of 0.1.  This
is a capacity diagnostic only: no frozen validation/test ID is passed to the
trainer or evaluator, normal-equation/resolvent-action losses are disabled,
and every completed or failed run is retained under a unique directory.
#>
param(
    [string]$Python = 'D:\Anaconda\envs\EGNN\python.exe',
    [string]$Config = 'config.yaml',
    [string]$MaterialIDs = 'data\processed\capacity_probe_ids\samples8_ids.json',
    [string]$OutputRoot = 'outputs\capacity_decomposition_v1\single_material_action_v1',
    [int]$Seed = 42,
    [int]$FactorEpochs = 100,
    [int]$DisplacementEpochs = 100,
    [int]$JointEpochs = 300,
    [string]$Device = 'auto'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = (Resolve-Path 'src').Path
$env:PYTHONWARNINGS = 'ignore::UserWarning:ast,ignore::DeprecationWarning:torch.jit._script'
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$parsedIds = Get-Content -LiteralPath $MaterialIDs -Raw | ConvertFrom-Json
# Windows PowerShell 5 serializes a JSON list as one pipeline object; unwrap
# it explicitly before validating the preregistered eight-material cohort.
$ids = @([string[]]$parsedIds)
if ($ids.Count -ne 8) { throw "Expected exactly eight declared strict-train material IDs; got $($ids.Count)." }

foreach ($material in $ids) {
    $label = ([string]$material).ToLower()
    $root = Join-Path $OutputRoot $label
    $idsPath = Join-Path $OutputRoot "$label`_ids.json"
    ConvertTo-Json -InputObject @([string]$material) | Set-Content -LiteralPath $idsPath -Encoding utf8
    if (Test-Path (Join-Path $root 'overfit_dfpt_train.json')) {
        Write-Host "Skipping completed single-material action probe $label"
        continue
    }
    if (Test-Path $root) {
        throw "Partial run exists at $root. Preserve it and choose a new output root."
    }

    & $Python -m piezojet.train --config $Config --seed $Seed `
        --material-ids-file $idsPath --material-ids-split same `
        --allow-noninductive-overfit --batch-size 1 `
        --factor-pretrain-epochs $FactorEpochs `
        --displacement-pretrain-epochs $DisplacementEpochs `
        --normal-equation-warmup-epochs $JointEpochs `
        --normal-equation-ramp-epochs 0 `
        --factor-pretrain-response-operator-action-weight 0.1 `
        --response-operator-action-loss-weight 0.1 `
        --epochs $JointEpochs --early-stopping-patience 0 `
        --device $Device --output-dir $root
    if ($LASTEXITCODE -ne 0) { throw "Single-material action probe failed for $material" }

    & $Python -m piezojet.evaluate_dfpt --checkpoint "$root\loss_best.pt" `
        --material-ids-file $idsPath --material-ids-split same --split train --device $Device `
        --output "$root\overfit_dfpt_train.json" --bootstrap-resamples 1
    if ($LASTEXITCODE -ne 0) { throw "Single-material action evaluation failed for $material" }
}

& $Python -m piezojet.experiment_registry
if ($LASTEXITCODE -ne 0) { throw 'Experiment registry refresh failed.' }
