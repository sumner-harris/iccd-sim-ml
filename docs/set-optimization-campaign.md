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
- Write `progress.json` after every epoch and atomically replace
  `best_checkpoint.pt` after each validation improvement so interrupted trials
  retain auditable state.
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
| 1 | running | Pretrained K=32 Deep Set; jointly fine-tune regressor at 0.1x; per-video loss weight 0.2 | epoch-zero R2 0.6507; epochs 1--2 R2 0.0859/0.0469 | Continue to early stopping; protected baseline retained |
| 2 | queued | Trial 1 with base and pretrained learning rates reduced 10x | pending | Tests whether immediate degradation is an optimization-step-size problem |
| 3 | planned | Trial 2 plus fixed pretrained BatchNorm running statistics | pending | Tests whether correlated same-material bags, rather than gradient step size alone, are overwriting mixed-material encoder statistics |
| 4 | planned | Trial 3 with a five-epoch frozen-regressor set-head warmup before joint fine-tuning | pending | Prevents an untrained set residual and the mature video encoder from adapting simultaneously on the first batch |
| 5 | candidate | Trial 4 with per-property convex attention over the K pretrained predictions | pending | Tests learned experiment reliability while preserving the exact mean baseline at initialization and constraining the combined prediction |
