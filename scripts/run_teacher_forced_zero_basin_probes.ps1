<#
.SYNOPSIS
Run explicitly non-inductive 1-, 8-, and 32-material memorization probes.

.DESCRIPTION
These probes answer a narrow falsifiable question: with the maintained model
class and teacher-forced U_eta curriculum, can every supervised physical factor
and the resulting ionic response be fit on 1, 8, or 32 strict-complete
materials?  They are not a benchmark, never evaluate the frozen test panel,
and write an explicit non-inductive marker into their resolved configuration.
#>
param(
    [string]$Python = 'D:\Anaconda\envs\EGNN\python.exe',
    [string]$Config = 'config.yaml',
    [string]$StrictSplit = 'data\processed\strict_completion_benchmark_train_v10_full_public.json',
    [string]$OutputRoot = 'outputs\teacher_forced_zero_basin_v1',
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

$split = Get-Content -LiteralPath $StrictSplit -Raw | ConvertFrom-Json
$ids = @($split.splits.train)
if ($ids.Count -lt 32) {
    throw "Strict split requires at least 32 train IDs; found $($ids.Count)."
}

# A 1/8/32 ladder distinguishes literal single-example memorization from a
# modest multi-material capacity failure without consulting held-out data.
foreach ($count in @(1, 8, 32)) {
    $label = "samples$count"
    $root = Join-Path $OutputRoot $label
    $idsPath = Join-Path $OutputRoot "$label`_ids.json"
    [object[]]$selectedIds = @($ids | Select-Object -First $count)
    ConvertTo-Json -InputObject $selectedIds | Set-Content -LiteralPath $idsPath -Encoding utf8
    if (Test-Path (Join-Path $root 'overfit_dfpt_train.json')) {
        Write-Host "Skipping completed zero-basin probe $label"
        continue
    }
    if (Test-Path $root) {
        throw "Partial probe output exists; preserve it and choose a fresh OutputRoot: $root"
    }

    & $Python -m piezojet.train --config $Config --seed $Seed `
        --material-ids-file $idsPath --material-ids-split same `
        --allow-noninductive-overfit --batch-size $count `
        --factor-pretrain-epochs $FactorEpochs `
        --displacement-pretrain-epochs $DisplacementEpochs `
        --normal-equation-warmup-epochs $JointEpochs `
        --normal-equation-ramp-epochs 0 `
        --epochs $JointEpochs --early-stopping-patience 0 `
        --device $Device `
        --output-dir $root
    if ($LASTEXITCODE -ne 0) {
        throw "Teacher-forced zero-basin probe failed for $label"
    }

    # The selected strict-train IDs are also in the global train partition.
    # Evaluation is therefore train-only and labeled non-inductive by the run.
    & $Python -m piezojet.evaluate_dfpt --checkpoint "$root\loss_best.pt" `
        --material-ids-file $idsPath --material-ids-split same --split train --device $Device `
        --output "$root\overfit_dfpt_train.json" --bootstrap-resamples 1
    if ($LASTEXITCODE -ne 0) {
        throw "Teacher-forced zero-basin evaluation failed for $label"
    }
}

& $Python -m piezojet.summarize_teacher_forced_probe `
    --root $OutputRoot --output "$OutputRoot\capacity_probe_summary.json" --threshold 0.99
if ($LASTEXITCODE -ne 0) {
    throw 'Teacher-forced zero-basin summary failed.'
}

& $Python -m piezojet.experiment_registry
if ($LASTEXITCODE -ne 0) {
    throw 'Experiment registry refresh failed after zero-basin probes.'
}
