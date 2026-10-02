# Element-held-out set-regression optimization campaign

## Objective and limits

Improve physical-property regression for unseen elements using unordered sets
of simulations. Run at most 20 training trials. The target is validation
set-level macro R2 greater than 0.8, but results must not be overstated if the
available 26 training and 5 validation elements do not support it.

## Evaluation contract

- Use the existing `element-held-out-seed42` train/validation/test manifest.
- Fit and reuse only the existing train-fitted scalers.
- Never use the six test elements for architecture selection or early stopping.
- Primary metric: physical-unit macro R2 over repeated fixed-size validation
  sets. Treat material-ensemble R2 as secondary because it uses more than one
  set per material.
- Select checkpoints by scaled validation Smooth L1 loss and report the R2 of
  the restored checkpoint.
- Preserve the epoch-zero mean-regressor baseline whenever a pretrained model
  is used.
- Record configuration, code commit, seed, best epoch, runtime, metrics, and
  failure diagnosis for every attempt.
- Prefer controlled changes motivated by the preceding result. Repeat
  promising configurations with additional seeds before interpreting small
  improvements.

## Established baselines

| Approach | K | Validation macro R2 | Interpretation |
|---|---:|---:|---|
| Single-video regressor | 1 | about 0.478 | Existing held-out checkpoint |
| Averaged pretrained predictions | 32 | 0.632 | Protected epoch-zero baseline |
| Frozen pretrained branch plus Deep Set residual | 32 | best observed about 0.653 | Modest gain; run stopped before final report |
| Joint Deep Set from random initialization | 32 | -0.300 restored | Severe element-level overfitting |

## Trial ledger

| Trial | Status | Main change | Primary validation result | Decision |
|---:|---|---|---|---|
| 1 | queued | Pretrained K=32 Deep Set; jointly fine-tune regressor at 0.1x; per-video loss weight 0.2 | pending | pending |

