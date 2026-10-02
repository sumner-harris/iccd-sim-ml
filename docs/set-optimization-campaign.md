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
| 1 | stopped for futility after epoch 8 | Pretrained K=32 Deep Set; jointly fine-tune regressor at 0.1x; per-video loss weight 0.2 | protected epoch-zero R2 0.6507; trained epochs ranged from 0.0859 to -0.5867 | Training loss fell from 0.412 to 0.152 while every validation epoch was much worse than baseline; stopped to avoid seven additional non-improving epochs |
| 2 | stopped for futility after epoch 1 | Trial 1 with base and pretrained learning rates reduced 10x | protected epoch-zero R2 0.6507; epoch 1 R2 0.0682 | Nearly identical collapse to Trial 1 despite 10x smaller optimizer steps; strongly implicates learning-rate-independent BatchNorm-statistics drift |
| 3 | stopped for futility after epoch 5 | Trial 2 plus fixed pretrained BatchNorm running statistics | protected epoch-zero R2 0.6507; epochs 1--5 declined from 0.6021 to 0.3386 | Fixed statistics removed the catastrophic collapse, confirming the diagnosis, but joint weights still overfit as training loss fell from 0.171 to 0.026 |
| 4 | stopped after first joint epoch | Trial 3 with a five-epoch frozen-regressor set-head warmup before joint fine-tuning | warmup reached R2 0.6566 at epoch 4 and best loss 0.231607/R2 0.6558 at epoch 5; first joint epoch fell to 0.5395 | New set layers can add a small gain while the pretrained branch is fixed, but unfreezing immediately harms held-out performance |
| 5 | complete; early stopped at epoch 20 | Permanently frozen pretrained regressor; train only the K=32 Deep Set residual at 3e-5 | selected epoch 5: set R2 0.6558 versus 0.6507 baseline; material ensemble 0.6624 versus 0.6560 | Small net gain with target tradeoffs; confirms the stable regime but remains far below 0.8 |
| 6 | stopped for futility after epoch 10 | Trial 5 plus per-property convex attention over the K pretrained predictions | best selected loss 0.232694/R2 0.6511 at epoch 5; below Trial 5 | Per-target weighting added complexity but did not improve the stable residual model |
| 7 | complete; early stopped at epoch 20 | Frozen pretrained K=32 Set Transformer residual at 3e-5 | selected epoch 5: set R2 0.6912 and material ensemble R2 0.6974 | First substantial gain; improves five of seven targets and becomes the leading configuration |
| 8 | stopped for futility after epoch 10 | Trial 7 with K increased from 32 to 64 | K=64 baseline R2 0.6596; best trained R2 0.6866 at epoch 5 | Below K=32's 0.6912 while doubling epoch time; more videos alone did not improve the learned aggregate |
| 9 | complete; early stopped at epoch 27 | Trial 7 with set-layer learning rate reduced from 3e-5 to 1e-5 | selected epoch 12: set R2 0.7048 and material ensemble R2 0.7104 | New leader; slower optimization gives a smoother, higher peak while the pretrained branch stays frozen |
| 10 | stopped for futility after epoch 5 | Trial 9 with training bags per material increased from 4 to 16 | best R2 0.7042/loss 0.215812 at epoch 3 | Statistically indistinguishable from Trial 9 while tripling epoch time; repeated subsets do not add independent material information |
| 11 | running | Trial 9 with weight decay increased from 1e-4 to 1e-3 | pending | Tests stronger regularization against rapid overfitting in the 26-material training set |
