# PyTorch emits non-fatal compatibility warnings on stderr.  Native-command
# stderr must not abort this queue; every Python invocation is instead checked
# explicitly through ``$LASTEXITCODE`` below.
$ErrorActionPreference = 'Continue'

# Registered execution order for the fourth-feedback diagnostics.  All runs use
# the same frozen 69/10/20 factor panel; no command changes its test IDs.
$root = 'E:\CODE\PiezoJet'
$python = 'D:\Anaconda\envs\EGNN\python.exe'
$env:PYTHONPATH = Join-Path $root 'src'
Set-Location $root

New-Item -ItemType Directory -Force 'outputs\feedback4_execution_v1' | Out-Null

& $python -m piezojet.protocol_ablation `
  --config config.yaml `
  --splits-file outputs\strict_learning_curve_v1\splits\strict_lambda_n69.json `
  --output-root outputs\optimization_ablation_v1 `
  --protocol all --seeds 42,43,44 --factor-updates 50 --joint-updates 50 `
  --diagnostics-every 10 *>&1 | Tee-Object -FilePath 'outputs\feedback4_execution_v1\protocol_ablation.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

foreach ($index in 0..4) {
  $split = "outputs\stratified_subset_resampling_v1\splits\strict_lambda_n35_subset$('{0:D2}' -f $index).json"
  $output = "outputs\stratified_subset_resampling_v1\runs\subset$('{0:D2}' -f $index)_seed42"
  & $python -m piezojet.train `
    --config config.yaml --splits-file $split --seed 42 --output-dir $output `
    --factor-pretrain-epochs 50 --factor-pretrain-patience 0 `
    --epochs 50 --early-stopping-patience 0 --freeze-factors-during-joint *>&1 |
      Tee-Object -FilePath "outputs\feedback4_execution_v1\subset$('{0:D2}' -f $index).log"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  & $python -m piezojet.evaluate_dfpt `
    --checkpoint "$output\loss_best.pt" --splits-file $split --split test `
    --output "$output\dfpt_test.json" *>&1 |
      Tee-Object -FilePath "outputs\feedback4_execution_v1\subset$('{0:D2}' -f $index)_eval.log"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $python -m piezojet.train `
  --config config.yaml `
  --splits-file outputs\strict_learning_curve_v1\splits\strict_lambda_n69.json `
  --seed 42 --output-dir outputs\mode_aware_smoke_v1\seed42 `
  --factor-pretrain-epochs 50 --factor-pretrain-patience 0 `
  --factor-pretrain-mode-aware-strain-weight 0.1 `
  --epochs 50 --early-stopping-patience 0 --freeze-factors-during-joint *>&1 |
    Tee-Object -FilePath 'outputs\feedback4_execution_v1\mode_aware_smoke.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m piezojet.evaluate_dfpt `
  --checkpoint outputs\mode_aware_smoke_v1\seed42\loss_best.pt `
  --splits-file outputs\strict_learning_curve_v1\splits\strict_lambda_n69.json `
  --split test --output outputs\mode_aware_smoke_v1\seed42\dfpt_test.json *>&1 |
    Tee-Object -FilePath 'outputs\feedback4_execution_v1\mode_aware_smoke_eval.log'
exit $LASTEXITCODE
