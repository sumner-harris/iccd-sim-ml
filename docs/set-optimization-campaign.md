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
| Averaged pretrained predictions | 32 | 0.6507 | Protected epoch-zero baseline |
| Frozen pretrained branch plus Deep Set residual | 32 | 0.6558 | Small but stable improvement |
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
| 11 | stopped for futility after epoch 12 | Trial 9 with weight decay increased from 1e-4 to 1e-3 | epoch-by-epoch losses and R2 match Trial 9 to printed precision | This AdamW change is negligible at the short duration and 1e-5 learning rate |
| 12 | complete; early stopped at epoch 37 | Trial 9 with a compact 64-token, one-layer Set Transformer branch | selected epoch 22: set R2 0.6993 and material ensemble R2 0.7048 | Learns more slowly and uses fewer parameters, but remains below the standard Trial 9 model |
| 13 | complete; early stopped at epoch 26 | Trial 9 with one learned Set Transformer pooling query and scalar residual head per property | selected epoch 11: set R2 0.7416 and material ensemble R2 0.7471 | Clear architecture gain; all seven properties improve over the pretrained mean baseline |
| 14 | complete; early stopped at epoch 29 | Trial 13 with 2x training-loss weight on cp, kappa, density, and critical temperature | selected epoch 14: set R2 0.74175 and material ensemble R2 0.74718 | Only 0.00015 above Trial 13; improves weighted targets but trades away similar performance elsewhere |
| 15 | complete; early stopped at epoch 28 | Repeat unweighted Trial 13 with model seed 43 and fixed bag seed 42 | selected epoch 13: set R2 0.7440 | Closely reproduces seed 42 and confirms the target-specific pooling gain is not a one-seed artifact |
| 16 | complete; early stopped at epoch 28 | Repeat unweighted Trial 13 with model seed 44 and fixed bag seed 42 | selected epoch 13: set R2 0.7452 | Seeds 42--44 give mean 0.7436 with sample standard deviation 0.00185 on identical validation bags |
| 17 | stopped for futility after epoch 12 | Trial 13 with K increased from 32 to 64 | best observed R2 0.7397 at epoch 11, below every K=32 seed while taking about twice as long per epoch | Larger bags do not justify their compute cost for this architecture |
| 18 | stopped for futility after epoch 25 | Trial 13 with a 12-epoch frozen warmup, then joint fine-tuning at an ultra-low 1e-8 pretrained learning rate, fixed pretrained BatchNorm statistics, and per-video auxiliary loss 0.2 | best and final R2 0.7089; post-unfreeze epochs took about 175 s versus 64 s frozen | Avoided collapse, but remained 0.033 below the replicated frozen result after 13 costly joint epochs; auxiliary individual loss changed only from about 0.2191 to 0.2169 |
| 19 | stopped for futility after epoch 13 | Trial 13 with space-filling training bags selected by farthest-point sampling over laser power and spot size; validation bags unchanged | final R2 0.6728 versus Trial 13's 0.7410 near the same epoch | Space-filling bags removed useful random-subset diversity; training loss fell while validation improved far too slowly |
| 20 | stopped for futility after epoch 6 | Target-specific K=32 model jointly fine-tuned from epoch 1 at a 1e-8 pretrained learning rate with fixed BatchNorm and no per-video auxiliary loss | final R2 0.6583 versus Trial 13's 0.6978 at epoch 6; epochs cost about 174 s | Removing the auxiliary loss did not remove the joint-training slowdown; frozen features remain both faster and more accurate |

## Seed stability and checkpoint ensembling

The target-specific K=32 Set Transformer was repeated with model seeds 42, 43,
and 44 while holding the split and all bag memberships fixed. Its validation
set-level macro R2 was 0.74160, 0.74404, and 0.74522 (mean 0.74362, sample
standard deviation 0.00185). This is a stable architectural gain rather than a
favorable single initialization.

A physical-unit average of the three checkpoint predictions gives validation
set-level macro R2 0.74490 and material-ensemble R2 0.75025. Adding the weighted
loss checkpoint reduces these values to 0.74420 and 0.74957, respectively, so
that checkpoint is excluded. The small ensemble gain shows that the three
models' errors are highly correlated.

## Pre-registered final selection and test rule

The three-seed unweighted ensemble is the default final model. A single Trial
18--20 checkpoint will replace it only if its macro R2 on the unchanged
validation bags exceeds the ensemble by more than 0.005. This margin is about
three times the observed across-seed standard deviation and limits selection
on noise after a 20-trial campaign. Validation bags use seed 43. After this
choice is written to `selection.json`, the selected method is evaluated once
on the locked test elements using pre-declared bag seed 44, K=32, and eight
bags per material. No architecture, checkpoint, or ensemble membership may be
changed in response to the test result.

## Final selection

The three-seed ensemble scored 0.74490 on the fixed validation bags. The best
single member scored 0.74522, only 0.00032 higher and therefore far below the
pre-registered 0.005 replacement margin. Trials 18--20 were all stopped below
the reference configuration and were ineligible. The selected method is thus
the physical-unit average of the Trial 13, 15, and 16 checkpoints.

## Locked test result

After selection was written to disk, the ensemble was evaluated once with
K=32 and eight seed-44 bags for each of the six locked elements: As, Cu, Pr,
Si, Te, and Zr. The bag-level macro R2 is **-0.70938** and the prediction-average
per-material macro R2 is **-0.70504**. This is a genuine failure to generalize
to the locked materials, not an R2 above 0.8.

| Property | Test bag R2 | Test MAE |
|---|---:|---:|
| heat capacity | 0.5619 | 85.12 |
| vaporization enthalpy | -0.2747 | 142,886.76 |
| thermal conductivity | 0.2044 | 73.01 |
| laser reflectivity | -0.8722 | 0.1699 |
| mass density | -3.6664 | 3,449.73 |
| boiling temperature | -0.1841 | 1,256.68 |
| critical temperature | -0.7346 | 3,525.00 |

Independent recomputation from `predictions.npz` reproduces every metric. The
saved targets match the manifest's element-property rows; all arrays are finite
and correctly aligned. Five of the six test materials lie within every
training target range; only Cu thermal conductivity is outside it. The result
therefore cannot be dismissed as a scaler error or simple target-range
extrapolation.

As a post-hoc diagnostic only, the protected mean of the pretrained K=32
single-video predictions has test macro R2 -0.58084. The learned set residual
improves heat capacity, vaporization enthalpy, conductivity, and boiling
temperature, but worsens reflectivity, critical temperature, and especially
density enough to reduce the macro score. A constant training-target-mean
baseline scores -0.12451 on these six materials. Neither diagnostic was used
for model selection, and no tuning was performed after the test was opened.

The scientifically supported conclusion is that target-specific set pooling
substantially improves the chosen five-element validation split, but the
available 26 training materials are insufficient for robust unknown-element
property prediction. A follow-up study should use nested repeated
element-held-out evaluation and more independent materials; this locked test
must not be reused as a tuning set.
