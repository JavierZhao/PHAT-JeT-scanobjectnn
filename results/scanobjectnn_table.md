# PHAT-JeT on ScanObjectNN PB_T50_RS — results

Protocol: 1024 points, coordinates only, unit-sphere normalization, no voting.
Test OA/mAcc reported at the **best-validation epoch** (validation = a stratified
10% split of train; the test set never drives any decision). Mean ± std over
seeds. Status as of the latest cluster sync — see `notes.md` for caveats.

## 1. Main table — PHAT-JeT vs published baselines

All baseline numbers **verified against the original papers** (see §4). Every
row is PB_T50_RS, the hardest variant.

**Protocol note — read before comparing.** Published baselines report
max-over-test-curve: PointNeXt's "validation set" *is* the test set and it
reports the best-epoch test score, and PointMLP has no validation set at all
and reports a running max. Our default protocol (test at best *held-out*
validation epoch) is strictly stricter. The **comparable** column is therefore
"OA (matched)"; "OA (strict)" is our own rule and is reported alongside for
honesty. Measured difference between the two: +0.65 (§5).

| Method | #Params | GFLOPs | OA (matched) | OA (strict) | mAcc |
|---|---|---|---|---|---|
| PointNet [1,2] | 3.5M | 0.9 | 68.2 | — | 63.4 |
| **PHAT-JeT-XS** (δ*, no height) | **0.157M** | 0.23 | 75.6 | 74.96 ± 0.82 | 70.74 |
| **PHAT-JeT-S** (δ*, no height) | **0.612M** | 0.90 | 76.5 | 75.78 ± 0.55 | 71.45 |
| **PHAT-JeT-M** (δ*, no height) | **1.214M** | 1.81 | 77.8 | 77.02 ± 0.80 | 73.01 |
| **PHAT-JeT-L** (δ*, no height) | **2.713M** | 4.02 | 77.9 | 77.15 ± 0.25 | 73.24 |
| PointNet++ [1,2] | 1.5M | 1.7 | 77.9 | — | 75.4 |
| DGCNN [1,2] | 1.8M | 4.8 | 78.1 | — | 73.6 |
| PointCNN [1] | 0.6M | — | 78.5 | — | 75.1 |
| BGA-DGCNN [2] | — | — | 79.7 | — | 75.7 |
| **PHAT-JeT-L + height** | **2.713M** | 4.02 | 80.09 ± 1.31 | 79.69 ± 1.21 | 76.58 |
| **PHAT-JeT-XS + height** | **0.157M** | 0.23 | 80.66 ± 0.16 | 80.00 ± 0.25 | 77.08 |
| **PHAT-JeT-M + height** | **1.214M** | 1.81 | 81.11 ± 0.59 | 80.46 ± 0.71 | 77.15 |
| SimpleView [1,2] | 0.8M | — | 80.5 ± 0.3 | — | — |
| **PHAT-JeT-S + height** | **0.612M** | 0.90 | 81.23 ± 0.48 | 80.55 ± 0.29 | 77.23 |
| **PHAT-JeT-S + xyz offsets** ★ | **0.612M** | **0.90** | **82.29 ± 0.41** | 81.45 ± 0.53 | **78.12** |
| MVTN [1] | 3.5M | 1.8 | 82.8 | — | — |
| PointMLP-elite [2] | 0.68M | — | 83.8 ± 0.6 | — | 81.8 |
| PointMLP [1] | 13.2M | 31.3 | 85.4 ± 1.3 | — | 83.9 |
| PointNet++ *w/ PointNeXt training* [1] | 1.5M | 1.7 | 86.1 ± 0.7 | — | 84.2 |
| PointNeXt-S [1] | 1.4M | 1.6 | 87.7 ± 0.4 | — | 85.8 |

All PHAT-JeT rows are 3 seeds at 250 epochs. "+ height" appends the object's
height above its own base as a 4th input feature (§3d); everything else is
identical.

### Honest assessment

- **Best result: PHAT-JeT-S + height, 80.55 ± 0.29 at 0.612M parameters.** That
  matches SimpleView (80.5 at 0.8M) with 24% fewer parameters, and beats
  PointNet++ (77.9), DGCNN (78.1), PointCNN (78.5) and BGA-DGCNN (79.7).
- Against PointNet the efficiency claim is strong: **XS + height is +11.8 OA
  with 22× fewer parameters**.
- It remains **3.2 behind PointMLP-elite** (83.8 at 0.68M) at matched budget,
  and 7.2 behind PointNeXt-S.
- Against the plan's success criteria this sits between the two defined
  outcomes: better than "competitive but behind", short of "within striking
  distance of PointMLP-elite/PointNeXt-S".

**Essential caveat.** PointNeXt [1] shows PointNet++ gains **+8.2 OA
(77.9 → 86.1) from training strategy alone, with no architecture change**. We
have closed part of that kind of gap ourselves (+4.8 from one input feature),
which is direct evidence that the remaining difference is **not demonstrably
architectural**. No architectural conclusion should be drawn from this table
without first exhausting training and input-representation changes.

## 2. Ablation — GMP and grid resolution (config S, 3 seeds)

| Condition | grid side | OA (%) | mAcc (%) |
|---|---|---|---|
| GMP on, δ=0.5 | 5 | 59.61 ± 0.44 | 54.15 ± 0.14 |
| GMP **off** | — | 65.63 ± 0.77 | 59.86 ± 0.85 |
| GMP on, δ=0.25 | 9 | 65.17 ± 0.48 | 60.66 ± 0.40 |
| GMP on, δ=0.125 | 18 | 73.98 ± 0.32 | 69.59 ± 0.51 |
| GMP on, δ=0.09375 | 24 | 75.80 ‡ | 71.62 ‡ |

**The GMP prior is only useful at fine grid resolution.** At δ=0.25 it is
statistically indistinguishable from having no prior at all (the 0.46-point gap
is smaller than the GMP-off seed spread of 0.77), and at δ=0.5 it is clearly
*worse* than no prior. At δ=0.125 it wins by 8.4 points.

Any statement of the form "GMP helps" is therefore incomplete without stating δ.
Note also that GMP-off still uses Morton ordering, so patches remain spatially
coherent — this ablation removes the explicit GMP module, not all geometric
structure.

## 3. Scaling ladder — both δ values

| Config | Params | OA @ δ=0.125 | OA @ δ=0.09375 | Δ |
|---|---|---|---|---|
| XS | 0.157M | 72.73 ± 1.18 | 74.96 ± 0.82 | +2.2 |
| S | 0.612M | 73.98 ± 0.32 | 75.78 ± 0.55 | +1.8 |
| M | 1.214M | 74.36 ± 0.17 | 77.02 ± 0.80 | +2.7 |
| L | 2.713M | 74.67 ± 0.57 | 77.15 ± 0.25 | +2.5 |

All 3 seeds, 250 epochs (one S seed at 241). **M nearly matches L** at δ\*
(77.02 vs 77.15, overlapping error bars) despite less than half the parameters —
the ladder is flat above ~1.2M.

Two observations:

1. **δ=0.09375 beats δ=0.125 at every rung**, by 1.0–2.3 points. δ\* is finer
   than the pre-registered anchors {0.5, 0.25, 0.125} would have selected;
   probing between anchors was necessary to find it.
2. **Scaling is nearly flat**: 17× the parameters (XS→L) buys ~+2 points.
   Accuracy is limited by grid resolution far more than by capacity — which is
   what motivates the GMP-efficiency work now in progress.

## 3b. δ is optimal at 0.09375 — the curve turns over (config M)

Finding finer δ required sparse GMP; the answer is that finer does **not** keep
helping. δ\* is bracketed on both sides.

| δ | grid side | GMP impl | test OA |
|---|---|---|---|
| 0.125 | 18 | dense (3 seeds) | 74.36 |
| **0.09375** | **24** | dense (3 seeds) | **77.02** |
| 0.03125 | 71 | sparse (1 seed) | 74.64 |
| 0.015625 | 141 | sparse (1 seed, 226 ep) | 69.85 |

The monotonic rise from δ=0.5 to δ=0.09375 does not continue: 0.03125 is 2.4
points worse and 0.015625 is 7.2 points worse. Plausible mechanism — at very
fine resolution each voxel holds roughly one point, so the 3×3×3 neighbourhood
spans a tiny physical region and GMP's message passing degenerates into local
noise instead of useful context.

This is a genuine negative result and worth reporting as one: the sparse
implementation was built to reach finer grids, and what it established is that
finer grids are not the answer. The optimum is now bracketed rather than open.

## 3c. Height appending — the largest single gain found (config M, δ=0.09375)

| arm | test OA | mAcc | vs baseline |
|---|---|---|---|
| baseline, no height (3 seeds) | 77.02 ± 0.80 | 73.01 | — |
| height = **unit** (CONTROL) | 76.04 | 71.26 | −1.0 |
| height = **raw** | 79.77 | 76.26 | **+2.8** |
| height = **shifted** | 80.71 / 81.33 | 77.55 / 78.87 | **+4.0** |

The `unit` control appends the normalized y channel — information the model
already has — and lands *below* baseline. So the gain is not from widening the
input embedding; it is the absolute-size information that unit-sphere
normalization destroys, which is highly discriminative among furniture classes.

**PHAT-JeT-M with shifted height reaches ~81.0 OA at 1.21M parameters**, which
would place it above PointNet++ (77.9), DGCNN (78.1), PointCNN (78.5),
BGA-DGCNN (79.7) and SimpleView (80.5) — see §1.

### Variance caveat that limits all single-run claims

The two `shifted` runs are the **same seed** on different hardware and differ by
0.62 points; the two sparse-parity runs, also same seed, differ by 2.0 points
(75.88 vs 77.90). That hardware/nondeterminism spread is as large as the
three-seed spread of the dense baseline (76.20–78.11). Consequently:

- "shifted beats raw" (~1.5 points apart) is **not** established.
- "height beats baseline and control" (+3 to +4 points) is comfortably outside
  that noise and is established.

Any arm intended for the paper needs three seeds. These are one seed each.

## 3d. FINAL height results — the gain shrinks as capacity grows

3 seeds, 250 epochs, δ=0.09375, `shifted` height.

| Config | Params | no height | **+ height** | gain |
|---|---|---|---|---|
| XS | 0.157M | 74.96 ± 0.82 | **80.00 ± 0.25** | **+5.0** |
| S | 0.612M | 75.78 ± 0.55 | **80.55 ± 0.29** | **+4.8** |
| M | 1.214M | 77.02 ± 0.80 | **80.46 ± 0.71** | **+3.4** |
| L | 2.713M | 77.15 ± 0.25 | **79.69 ± 1.21** | **+2.5** |

Two things fall out:

1. **The benefit is inversely proportional to capacity** (+5.0 at XS down to
   +2.5 at L). Absolute size information *substitutes* for model capacity — a
   small model given the right information matches a 17× larger one denied it.
2. **With height the ladder is completely flat, even slightly inverted**:
   XS 80.00, S 80.55, M 80.46, L 79.69. Scaling now buys nothing at all. The
   earlier flat-scaling finding was not a quirk; the architecture genuinely
   saturates, and once the information gap is closed it saturates immediately.

`raw` vs `shifted` at config M: 79.61 ± 0.15 vs 80.46 ± 0.71. Shifted is ahead
by 0.85, roughly one standard deviation — suggestive, **not** established.

**Best configuration: PHAT-JeT-S + height, 80.55 ± 0.29 OA at 0.612M params.**
That matches SimpleView (80.5 at 0.8M) with 24% fewer parameters, and beats
PointNet++ (77.9), DGCNN (78.1), PointCNN (78.5) and BGA-DGCNN (79.7).
It remains 3.2 behind PointMLP-elite (83.8 at 0.68M).

## 3e. Phase D — patch size does NOT help (config M, δ\*, 2 seeds)

| Patch size P | test OA |
|---|---|
| 16 | 77.35 |
| 32 (default) | 77.02 (3 seeds) |
| 64 | 76.63 |
| 128 | 76.41 |

Enlarging the local-attention window makes things slightly *worse*, and the
whole range spans 0.94 points — within seed noise.

This is evidence **against** the hypothesis that local receptive field limits
the model. It was the leading explanation for the remaining gap to PointNeXt,
and the pre-registered patch sweep does not support it. Whatever hierarchy
buys, it is unlikely to be simply "each token sees more neighbours".

## 3f. Phase C — ordering robustness (config M, δ\*, 3 seeds)

| Ordering | test OA | mAcc |
|---|---|---|
| Morton | 77.02 ± 0.80 | 73.01 |
| fixed random | 76.38 ± 0.51 | 72.66 |

Morton is ahead by 0.64 points, consistent in direction but within about one
standard deviation. The transfer analogue of the paper's Table 5: the model is
**largely robust to point ordering**, needing only that the ordering be
consistent between train and test. Notably, destroying spatial coherence in the
patches costs well under a point, which further undercuts the receptive-field
explanation — if patch contents barely matter, patch geometry is not the
bottleneck.

## 3g. Hierarchy and receptive-field scaling — NO effect

Config M, δ\*, height on, 2 seeds per arm, against the height baseline of
80.46 ± 0.71 (3 seeds).

| arm | mechanism | test OA | vs baseline |
|---|---|---|---|
| **baseline** (height only) | — | **80.46 ± 0.71** | — |
| `ds2-none` | downsample 1024→512, δ fixed | 80.52 | +0.1 |
| `ds2-density` | downsample + δ × 2^⅓ | 81.06 | +0.6 |
| `ds2-double` | downsample + δ × 2 | 80.64 | +0.2 |
| `coarse-pool2` | parallel coarse path, full resolution | 80.21 | −0.3 |
| `rf-kernel5` | GMP extent ±0.094 → ±0.188 | 79.51 | **−1.0** |
| `rf-kernel7` | GMP extent → ±0.281 | incomplete | trending down |

**Every arm falls inside the baseline's own seed spread (±0.71).** The largest
effect, `ds2-density` at +0.6, is smaller than that spread and rests on two
seeds. Nothing here is a result.

Widening the GMP kernel is, if anything, mildly **harmful** (−1.0 at kernel 5,
with kernel 7 trending worse still before completion).

### What this means, taken with Phases C and D

Three independent probes of spatial aggregation now agree:

| probe | knob | effect |
|---|---|---|
| Phase D | local-attention patch size (16→128) | none, slightly negative |
| Phase C | Morton vs random ordering | 0.64, ~1σ |
| §3g | hierarchy, GMP extent, coarse path | none, kernel widening negative |

**The architecture is insensitive to how information is spatially aggregated.**
Meanwhile the two large gains — GMP grid resolution (+8.4) and absolute height
(+4.8) — are both about *what information is available at all*.

A plausible reading: PHAT's patch-token mechanism already provides a global
communication path (every patch talks to every other through its token), so
additional hierarchy or wider kernels are redundant, and over-smoothing costs a
little. If so, the jet architecture's design is doing its job on point clouds —
the bottleneck is input representation, not information flow.

This also means the remaining ~7 points to PointNeXt-S are **not** explained by
hierarchy or receptive field. The untested candidate with the strongest prior is
the rest of the training recipe: PointNeXt demonstrated +8.2 OA on PointNet++
from training strategy alone, and we have so far adopted only one element of it.

## 3h. Gap-closing arms (config S, δ\*, height on) — interim

Against the S + height baseline of **80.55 ± 0.29**. Best available run per arm;
epoch counts noted where a run is unfinished.

| arm | change | OA | epochs |
|---|---|---|---|
| baseline | height (shifted) | 80.55 ± 0.29 | 250, 3 seeds |
| **`xyzabs`** | absolute offsets on **all three axes** | **82.27** | 250 ✓ |
| `gmpmean` | occupancy-normalized GMP | 81.68 | 196 |
| `gmptri` | trilinear sub-voxel position | 80.43 | 143 |
| `pn2` | PointNeXt's literal recipe settings | **79.08** | 250 ✓ |

**`xyzabs` is the best result of the study.** Extending absolute offsets from
one axis to three adds **+1.7** on top of height's +4.8. Two independent
research passes converged on this before it was run.

Every large gain now has the same shape: **restoring information that
normalization destroyed** (grid resolution +8.4, height +4.8, three-axis
offsets +1.7). Nothing from capacity or spatial aggregation has ever moved the
number.

### The PointNeXt recipe is worse for this architecture, not better

`pn2` adopts their released ScanObjectNN settings verbatim — lr 2e-3,
ε=0.3 (Keras 0.32143), cosine floor 1e-4 over t_max=200, gradient clip 10 —
and lands **1.5 points below** our existing recipe.

This closes the training-recipe hypothesis from the opposite direction to the
one expected. Our recipe was already better tuned for this architecture than
theirs. Combined with the fact that we already matched every item in their
additive study, the remaining gap to PointNeXt-S is **not** a training-strategy
gap.

*Caveats: one complete seed for `xyzabs` (its second is at 189/250 tracking
81.85); `gmpmean` still has 54 epochs to run. Not yet three-seed claims. The
two arms motivated by the overfitting diagnostic (`poolmax`, `reg`) are only
5–8 epochs in.*

## 3i. Attention ablation — the attention machinery contributes ~1 point

Config S, δ\*, height on, vs the 80.55 ± 0.29 baseline. 2 seeds × 2 GPU pools.

| arm | params | test OA (range) |
|---|---|---|
| full model | 612K | 80.55 ± 0.29 |
| no local attention | 480K | 79.0 – 80.2 |
| no patch messages | 447K | 79.6 – 80.5 |
| **neither (GMP → FFN only)** | **314K** | **79.1 – 79.6** |

Removing the *entire* attention machinery — both the within-patch attention and
the patch-token global path — costs about **1–1.5 points**, while removing GMP
costs 8.4. The model is essentially an information engine (GMP + input
features) with attention as a small refinement. A 314K-parameter GMP→FFN stack
reaches ~79.3.

This reframes the transfer claim: what transfers *usefully* from the jet
architecture is chiefly the GMP positional prior, not the hierarchical
attention that gives PHAT its name. That must be stated honestly in the paper.

## 3j. Further nulls: max pooling and regularization

- `poolmax` (max readout, the jet model's own choice): 79.4–80.1 — **no gain**,
  contradicting the PointNet max-pooling expectation for this architecture.
- `reg` (dropout 0.3/0.1 + 256-wide head): 77.6–78.6 — **hurt**, despite the
  falling-train-loss diagnostic suggesting overfitting. The 8–11 point
  val–test gap is therefore likely a train/test distribution shift in
  PB_T50_RS itself (val is drawn from the training distribution; test objects
  differ), not classical overfitting. Regularizing cannot close a shift.
- `pn2` (PointNeXt's literal recipe): 77.8–79.1 — confirmed worse across all
  4 runs.

## 3k. FINAL arm comparison — all complete 250-epoch runs, config S (0.612M)

Two protocols shown: ours (test at best-validation epoch) and the baselines'
(max over the test curve). See §5 for why both are reported.

| arm | n | test@best-val | curve max | final epoch |
|---|---|---|---|---|
| height baseline | 3 | 80.55 ± 0.29 | 81.23 ± 0.48 | 80.73 ± 0.43 |
| **xyz_shifted** | 4 | **81.45 ± 0.53** | **82.29 ± 0.41** | **81.70 ± 0.37** |
| sparse_mean GMP | 4 | 80.93 ± 0.65 | 81.39 ± 0.34 | 80.93 ± 0.35 |
| trilinear GMP | 4 | 80.99 ± 0.69 | 81.80 ± 0.31 | 81.22 ± 0.12 |
| no attention at all | 4 | 79.08 ± 0.53 | 79.84 ± 0.34 | 79.40 ± 0.45 |

**Corrections to earlier interim reporting in this file.** Two numbers quoted
mid-campaign were single best runs, not means, and both come down with full
seeds:

- `xyz_shifted` was reported at 82.27; the 4-run mean is **81.45 ± 0.53** on
  our protocol (82.29 ± 0.41 on the baselines'). The 82.27 was one run.
- `sparse_mean` was reported at 81.68 from a partial run; complete, it is
  **80.93 ± 0.65** — only +0.4 over baseline and **inside seed noise**. It is
  not a confirmed gain, and neither is trilinear GMP (80.99 ± 0.69).

So the only change beyond height that survives full seeds is **xyz_shifted**,
at about **+0.9** (roughly 2σ on the pooled spread, consistent across 4 runs).

## 3l. Training on 100% of the data buys nothing

Both arms, 3 seeds, `--val_fraction 0` (no holdout, matching PointNeXt/PointMLP):

| config | 90% train (curve max) | 100% train (curve max) | Δ |
|---|---|---|---|
| xyz_shifted | 82.29 ± 0.41 | 82.14 ± 0.13 | −0.15 |
| height only | 81.23 ± 0.48 | 80.94 ± 0.37 | −0.29 |

**The 10% holdout costs nothing measurable** — the estimate of +0.3 to +0.8 was
wrong. So the protocol difference against published numbers is purely the
*selection rule* (+0.65), not the training-set size, and our held-out
validation split is free.

Worth noting: the full-training-set runs have markedly tighter spread
(±0.13 vs ±0.41), so the extra 10% buys stability rather than accuracy.

## 3m. Final round — combination, multi-scale GMP and reallocation are all null

All complete, config S unless noted, against `xyz_shifted` at 81.45 ± 0.53.

| arm | n | test@best-val | note |
|---|---|---|---|
| **xyz_shifted** | 4 | **81.45 ± 0.53** | best configuration found |
| multi-scale GMP ×3 | 6 | 81.22 ± 0.96 | null, and higher variance |
| combo xyz + sparse_mean | 6 | 81.01 ± 0.54 | **worse than xyz alone** |
| GMP-only, config M | 5 | 80.71 ± 0.23 | reallocation does not pay |
| height baseline | 3 | 80.55 ± 0.29 | — |

Three hypotheses died here, and all three were mine:

1. **The two "winners" do not stack.** `combo` (81.01) is *below* `xyz_shifted`
   alone (81.45). That is consistent with `sparse_mean` never having been a
   real gain — it was a partial-run artifact — and combining it costs a little.
2. **Multi-scale GMP is null.** A parallel bank of voxel resolutions
   (0.0625 / 0.09375 / 0.125) scores 81.22 ± 0.96, indistinguishable from a
   single scale and noisier. δ\* being sharply peaked did not mean neighbouring
   scales carried complementary information.
3. **Reallocating capacity from attention to GMP does not pay.** Config M with
   attention stripped (619K params, essentially the same budget as full S at
   612K, with 4 GMP blocks instead of 2) reaches 80.71 ± 0.23 — below full S
   with attention. So although attention is worth only ~1 point, spending its
   parameters on more GMP is worth less. It has the tightest variance of any
   arm (±0.23), which is the one thing it does buy.

**The gap-closing effort has plateaued.** The trajectory was
75.78 → 80.55 (height) → 81.45 (three-axis offsets); nothing since has moved it
outside noise.

## 3n. Research-derived arms — FPS is the one that worked

Config S, δ\*, xyz_shifted input, against that arm's 82.29 ± 0.41 (matched).

| arm | n | matched OA | strict OA | verdict |
|---|---|---|---|---|
| xyz_shifted (reference) | 4 | 82.29 ± 0.41 | 81.45 ± 0.53 | — |
| FPS point sampling (n=8) | 8 | 82.15 ± 0.52 | 81.55 ± 0.52 | **ties — not a gain** |
| concat(max, mean) readout | 3 | 81.23 ± 0.19 | 81.04 ± 0.20 | null |
| sub-voxel position | 2 | 81.26 ± 0.10 | 80.86 ± 0.19 | null (2 seeds) |

**FPS ties the reference and is not a gain.** At n=3 it looked like the best
configuration found (81.62 strict vs 81.45); extended to **n=8** it settles at
81.55 ± 0.52 strict and 82.15 ± 0.52 matched — statistically identical to
xyz_shifted alone (81.45 ± 0.53 / 82.29 ± 0.41).

This is the third time in this campaign an arm looked positive at 2–3 seeds and
regressed to the reference once seeded properly. On this benchmark, with a
per-arm σ near 0.5, **no difference below roughly 1 point is a result at n≤4**.
PointMLP's own published σ is ±1.3, which says the same thing.

Its final-epoch score remains marginally higher (81.91 vs 81.70), so the "more
stable at convergence" reading survives weakly, but not as an accuracy claim.

The other two research leads are null. Notably, **radius-normalized sub-voxel
position did not help** despite a well-argued mechanism (PointNeXt reports +0.3
for the analogous change, and our GMP discards in-voxel position entirely). At
2 seeds this is provisional, but it is not trending positive.

## 3o. FLOP-matched scaling confirms we are not compute-limited

| config | params | GFLOPs | matched OA |
|---|---|---|---|
| **S + xyz** | 0.612M | **0.90** | **82.29 ± 0.41** |
| M + xyz (FLOP-matched to PointNeXt-S) | 1.215M | 1.81 | 81.24 ± 0.22 |
| XS + xyz | 0.157M | 0.23 | 81.22 ± 0.16 |
| PointNeXt-S | 1.4M | 1.6 | 87.7 |

Config M **exceeds** PointNeXt-S's compute (1.81 vs 1.6 GFLOPs) at 13% fewer
parameters and scores **1.05 points below our own S**. Spending PointNeXt's FLOP
budget buys nothing.

Equally striking at the other end: **XS matches M within noise** (81.22 vs
81.24) at **8× fewer parameters and 8× fewer FLOPs**. Accuracy here is
essentially independent of capacity across a 17× range, so the efficient
configuration is the one to report.

## 4. Sources for baseline numbers

[1] Qian et al., *PointNeXt: Revisiting PointNet++ with Improved Training and
Scaling Strategies*, NeurIPS 2022 — Table 2.
https://proceedings.neurips.cc/paper_files/paper/2022/file/9318763d049edf9a1f2779b2a59911d3-Paper-Conference.pdf

[2] Ma et al., *Rethinking Network Design and Local Geometry in Point Cloud: A
Simple Residual MLP Framework* (PointMLP), ICLR 2022 — Table 3.
https://arxiv.org/pdf/2202.07123

PointMLP states it does **not** use the voting strategy, matching our protocol.

**Correction to the handoff's from-memory table:** PointMLP has **13.2M**
parameters, not 12.6M. The other entries it listed were accurate.

## 5. Not yet run

- Phase C (ordering: Morton vs fixed-random at config M) — the transfer analogue
  of the paper's Table 5.
- Phase D (patch-size sweep P ∈ {16, 32, 64, 128} at config M).
- FLOPs column: profiling works on the cluster's TF 2.13 but raises on local
  TF 2.21, so it is captured per-run in `metrics.json` and not yet collated.
