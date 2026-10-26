# Model / system card

## Intended use

Measure how a neural morphology classifier changes across synthetic acquisition conditions. Educational local experiment; no production deployment is implied.

## Data

Synthetic arrays, manually written documents or replayed JSON. No real user records. Seeds and sample inputs are recorded in app.py and sample_input.json.

## Method

Use hand-engineered intensity histogram and local-gradient summaries as inputs to a two-layer tanh classifier. Train on bright imaging domains and compare raw versus per-image normalized features on a shifted domain.

## Evaluation protocol

Split entire acquisition domains. Report held-out-domain accuracy for both preprocessing choices with a majority baseline, avoiding pixel-level leakage.

## Known limits

Labels are fictional sparse versus dense patterns, not diseases or diagnoses. Normalization can erase genuine biology and must be assessed on real data.

## Verification status

Local tests and default experiment are executed during packaging. Live model generation is not exercised in the build. A successful validation result is limited to the checks implemented in code.

## Further work

Use licensed morphology data, specimen-group splits, stain or instrument perturbations, calibration and domain-adaptation baselines.
