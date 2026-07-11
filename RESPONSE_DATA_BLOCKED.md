# PiezoJet implementation blocked

## Milestone

M5 response coefficient training (dielectric / elastic / piezo unified potential)

## Command

```text
conda activate EGNN
python -m piezojet.audit_responses --data-root data/raw/gmtnet --output outputs/response_audit
```

## Git commit

`8c18e49215baa4fc619baf1ed11964081f83d24a`

## Data file and SHA256

- `data/raw/gmtnet/data/jarvis_diele_piezo.pkl`
- `2a57e081f0072b2ac7fca7769095adcded1d299d2cd971db5c93fd25eb66929d`
- `data/raw/gmtnet/data/jarvis_elastic.pkl` (hash recorded in `outputs/response_audit/response_fields.json`)

## Exact error / ambiguity

The `dielectric` and `dielectric_ionic` fields are observed as symmetric 3x3 arrays, but the official GMTNet loader does not declare a unit conversion or an authoritative susceptibility convention. The task requires units and tensor conventions to be uniquely confirmed before M5 heads or scalar-potential coefficient training.

## What was already verified

- piezo records: 5,000 raw, 4,998 loader-valid
- elastic records: 14,500
- piezo/elastic material-ID intersection: 3,346
- response field shapes and finite-value statistics are in `outputs/response_audit/response_fields.json`

## No fallback taken

No guessed dielectric unit, alternative dataset, synthetic label, or M5 training was used.

## Needed from the user

Authoritative dielectric unit/susceptibility convention and confirmation of the intended tensor field (`dielectric` or `dielectric_ionic`) for M5.

## Safe resume command

```text
python -m piezojet.audit_responses --data-root data/raw/gmtnet --output outputs/response_audit
```
