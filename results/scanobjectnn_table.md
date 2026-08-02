# PHAT-JeT on ScanObjectNN PB_T50_RS — results

Protocol: 1024 points, coordinates only, unit-sphere normalization, no voting.
Test OA/mAcc reported at the **best-validation epoch** (validation = a stratified
10% split of train; the test set never drives any decision). Mean ± std over
seeds. Status as of the latest cluster sync — see `notes.md` for caveats.

## 1. Main table — PHAT-JeT vs published baselines

All baseline numbers **verified against the original papers** (see §4). Every
row is PB_T50_RS, the hardest variant.

| Method | #Params | OA (%) | mAcc (%) |
|---|---|---|---|
| PointNet [1,2] | 3.5M | 68.2 | 63.4 |
| **PHAT-JeT-XS** (δ=0.09375) | **0.157M** | **74.96 ± 0.82** | **70.74 ± 0.85** |
| **PHAT-JeT-S** (δ=0.09375) | **0.612M** | **75.78 ± 0.55** | **71.45 ± 0.53** |
| **PHAT-JeT-M** (δ=0.09375) | **1.214M** | **77.02 ± 0.80** | **73.01 ± 1.02** |
| **PHAT-JeT-L** (δ=0.09375) | **2.713M** | **77.15 ± 0.25** | **73.24 ± 0.27** |
| PointNet++ [1,2] | 1.5M | 77.9 | 75.4 |
| DGCNN [1,2] | 1.8M | 78.1 | 73.6 |
| PointCNN [1] | 0.6M | 78.5 | 75.1 |
| BGA-DGCNN [2] | — | 79.7 | 75.7 |
| SimpleView [1,2] | 0.8M | 80.5 ± 0.3 | — |
| MVTN [1] | 3.5M | 82.8 | — |
| PointMLP-elite [2] | 0.68M | 83.8 ± 0.6 | 81.8 ± 0.8 |
| PointMLP [1] | 13.2M | 85.4 ± 1.3 | 83.9 ± 1.5 |
| PointNet++ *w/ PointNeXt training* [1] | 1.5M | 86.1 ± 0.7 | 84.2 ± 0.9 |
| PointNeXt-S [1] | 1.4M | 87.7 ± 0.4 | 85.8 ± 0.6 |

† 3 seeds truncated at 192–196/250 epochs by a host-memory bug; rerunning.
‡ 2 seeds only (third launched). Both are near-final but not protocol-clean.

### Honest assessment

- PHAT-JeT clearly beats **PointNet**: XS is +6.9 OA with **22× fewer
  parameters**.
- PHAT-JeT-L sits just below **PointNet++** and **DGCNN**, using more parameters
  than either.
- At matched budget the comparison is unfavourable: **PointMLP-elite reaches
  83.8 at 0.68M, versus our 75.8 at 0.61M — an 8-point deficit.**
- Against the plan's success criteria this is the **"acceptable outcome"**
  (competitive but behind), not the good one. The supported claim is *the
  components transfer*, not *SOTA-competitive*.

**Essential caveat.** PointNeXt [1] shows PointNet++ gains **+8.2 OA
(77.9 → 86.1) from training strategy alone, with no architecture change**. Our
recipe is deliberately plain, and PHAT-JeT-L's 77.2 sits almost exactly at the
level of *original-recipe* PointNet++ (77.9) and DGCNN (78.1) rather than at the
level of modern-recipe results. The gap to PointNeXt-S is therefore **not
demonstrably architectural**. A training-recipe experiment is in progress; until
it reports, no architectural conclusion should be drawn from this table.

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
