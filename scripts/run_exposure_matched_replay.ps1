<#
.SYNOPSIS
Run preregistered 1/5/10/20-pass macro/branch/strict exposure checkpoints.

.DESCRIPTION
Each training epoch is one complete pass over all three independent streams.
Runs start from the same inductive structural checkpoint and frozen split.
No optimizer-update cap is accepted by the multistream trainer.
#>
param(
    [int[]]$Passes = @(1, 5, 10, 20),
    [int[]]$Seeds = @(42, 7, 1729),
    [string]$Python = 'D:\Anaconda\envs\EGNN\python.exe',
    [string]$Config = 'config.yaml',
    [string]$Split = 'data\processed\full_corpus_multitask_train1603_v1.json',
    [string]$OutputRoot = 'outputs\exposure_matched_direct_u_v2_conditioning'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = (Resolve-Path 'src').Path
# Suppress only the two inventoried upstream import-time warnings. Pytest keeps
# the exact-message allowlist and fails on every new warning category.
$env:PYTHONWARNINGS = 'ignore::UserWarning:ast,ignore::DeprecationWarning:torch.jit._script'

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$PlanPath = Join-Path $OutputRoot 'experiment_plan.json'
$StatusHistoryPath = Join-Path $OutputRoot 'experiment_status_history.jsonl'
if (-not (Test-Path $PlanPath)) {
    [ordered]@{
        schema_version = 1
        experiment_id = (Split-Path $OutputRoot -Leaf)
        driver = 'scripts/run_exposure_matched_replay.ps1'
        passes = $Passes
        seeds = $Seeds
        conditions = @('physical', 'matched_direct_total')
        config = $Config
        split = $Split
        checkpoint_selection = 'validation_loss_only'
        frozen_validation_count = 10
        frozen_test_count = 20
        interpretation = 'Physical branch and isolated macro negative control are separate experiments.'
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $PlanPath -Encoding utf8
}

function Add-ExperimentStatus([string]$Status, [string]$Detail) {
    [ordered]@{
        time_utc = [DateTime]::UtcNow.ToString('o')
        status = $Status
        detail = $Detail
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $StatusHistoryPath -Encoding utf8
}

function Update-ExperimentRegistry {
    & $Python -m piezojet.experiment_registry `
        --outputs outputs `
        --json outputs\EXPERIMENT_REGISTRY.json `
        --markdown EXPERIMENT_REGISTRY.md `
        --artifact-index outputs\EXPERIMENT_ARTIFACT_INDEX.jsonl
}

trap {
    Add-ExperimentStatus 'failed_or_interrupted' $_.Exception.Message
    Update-ExperimentRegistry
    Write-Error $_
    exit 1
}

Add-ExperimentStatus 'running' "registered grid passes=$($Passes -join ',') seeds=$($Seeds -join ',')"
Update-ExperimentRegistry

foreach ($PassCount in $Passes) {
    foreach ($Seed in $Seeds) {
        $Output = "$OutputRoot\physical\passes$PassCount`_seed$Seed"
        if (Test-Path "$Output\dfpt_test.json") {
            Write-Host "Skipping completed physical run passes=$PassCount seed=$Seed"
        } elseif (Test-Path $Output) {
            throw "Partial physical output exists; preserve it and choose a fresh OutputRoot: $Output"
        } else {
        & $Python -m piezojet.train --config $Config --splits-file $Split `
            --seed $Seed --epochs $PassCount --factor-pretrain-epochs $PassCount `
            --early-stopping-patience 0 --output-dir $Output
        if ($LASTEXITCODE -ne 0) {
            throw "Exposure replay failed for passes=$PassCount seed=$Seed"
        }
        & $Python -m piezojet.evaluate_dfpt `
            --checkpoint "$Output\loss_best.pt" --splits-file $Split `
            --split test --device auto --output "$Output\dfpt_test.json"
        if ($LASTEXITCODE -ne 0) {
            throw "Exposure evaluation failed for passes=$PassCount seed=$Seed"
        }
        }
        $DirectOutput = "$OutputRoot\direct\passes$PassCount`_seed$Seed"
        if (Test-Path "$DirectOutput\test.json") {
            Write-Host "Skipping completed direct run passes=$PassCount seed=$Seed"
            continue
        } elseif (Test-Path $DirectOutput) {
            throw "Partial direct output exists; preserve it and choose a fresh OutputRoot: $DirectOutput"
        }
        & $Python -m piezojet.train_direct_baseline --config $Config --splits-file $Split `
            --seed $Seed --epochs $PassCount --family cartesian `
            --pretrained-encoder 'outputs\full_corpus_multitask_detached_lift_v2\pretrain_full_corpus_seed42\best_encoder.pt' `
            --output-dir $DirectOutput
        if ($LASTEXITCODE -ne 0) {
            throw "Matched direct exposure replay failed for passes=$PassCount seed=$Seed"
        }
        & $Python -m piezojet.evaluate_direct_baseline `
            --checkpoint "$DirectOutput\loss_best.pt" --splits-file $Split `
            --split test --device auto --output "$DirectOutput\test.json"
        if ($LASTEXITCODE -ne 0) {
            throw "Matched direct test evaluation failed for passes=$PassCount seed=$Seed"
        }
    }
}

& $Python -m piezojet.summarize_exposure_replay `
    --output-root $OutputRoot `
    --passes ($Passes -join ',') --seeds ($Seeds -join ',') `
    --output "$OutputRoot\report\hierarchical_summary.json"
if ($LASTEXITCODE -ne 0) {
    throw "Exposure replay summary failed"
}

Add-ExperimentStatus 'completed' 'all requested physical/direct points and hierarchical summary completed'
Update-ExperimentRegistry
