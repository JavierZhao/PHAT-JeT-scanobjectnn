# ScanObjectNN transfer study — running notes

Honest record of deviations, failures and findings. Started 2026-07-31.
Plan: `PLAN_scanobjectnn.md`. Spec: `claude_code_scanobjectnn_handoff.md`.

## Dataset

Obtained from the official HKUST server
(`https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip`, 13,099,259,542 bytes),
MD5 verified against the checksum the same host publishes
(`36876af479f9ad39abad5ebcd89038dd`). Extracted PB_T50_RS main_split to
`/j-jepa-vol/scanobjectnn/main_split/`. Validated in the training image:
train `(11416, 2048, 3)` float32, test `(2882, 2048, 3)`, 15 classes,
labels 0–14. The h5 files also carry a `mask` array, unused here.

## Phase 0

### Sanity gate — failed once, then passed

**First run: 29.6% OA (threshold 60%). Two distinct causes, both real.**

1. **Pipeline bug — wrong rotation axis.** Training augmentation rotated about
   **z**, but ScanObjectNN is **y-up**. Rotating about a horizontal axis tumbles
   objects out of their upright pose, and because eval is not augmented, it also
   created a train/test distribution mismatch.

   Confirmed independently from the data, not just from convention: per-object
   axis extents over the 11,416 training clouds give

   | | x | y | z |
   |---|---|---|---|
   | corr with x | 1.000 | 0.184 | 0.706 |
   | corr with y | 0.184 | 1.000 | 0.172 |
   | corr with z | 0.706 | 0.172 | 1.000 |

   x and z extents are strongly coupled while y is nearly independent of both —
   the signature of a dataset rotation-augmented *about y*, which mixes the two
   horizontal axes and leaves the vertical one alone. y also has the smallest
   mean extent (0.826 vs 1.046 and 0.982). Fixed to PointNet's canonical
   y-rotation matrix.

2. **The gate model was too weak to certify anything.** 10,511 parameters, no
   batch normalization, 20 epochs — incapable of reaching 60% even on a perfect
   pipeline, so it could not distinguish "pipeline broken" from "model too
   small". Rebuilt with PointNet shared-MLP widths (64, 64, 64, 128, 1024),
   batch norm, and a 512/256 head: 815,311 parameters.

   A third run was still cut short at 55.2% because the k8s manifest passed
   `--epochs 20`, overriding the new 60-epoch default. Manifest fixed.

**Final: best test OA 0.6107 vs threshold 0.60 — PASSED**, 1337 s, 60 epochs.

*Caveat, stated plainly:* this is a marginal pass on a max-over-curve statistic.
The last ten epochs oscillate between 0.53 and 0.61, and the 0.6107 happens to
fall on the final epoch. A different seed could plausibly have landed at 0.59.
The gate's purpose — showing the pipeline learns — is served (published PointNet
with T-Nets reaches ~68% on PB_T50_RS, and this variant omits them), but it
should not be read as a tight result. The rotation fix roughly doubled gate
performance (29.6% → 61.1%), which is the substantive evidence.

Also noted: the handoff expected the sanity baseline to clear 60% "quickly";
it needed 60 epochs. Recorded as a deviation.

### Preflight / memory check — passed

Ran `train_scanobjectnn.py` end to end on the cluster with real data at the
heaviest configuration (L, δ=0.125, batch 32, 2 epochs) before committing to
long runs. Completed on an RTX 3090: config L fits comfortably, and FLOPs
profiling works on TF 2.13 (it raises on the local TF 2.21, which is why the
call is wrapped and deferred to the end of a run). Measured params 2,712,591,
matching the local count exactly. Test OA after a single epoch: 43.96%.

## Timing (measured, not estimated)

| Config | s/epoch | 250 epochs |
|---|---|---|
| S | 51.9 | ~3.6 h |
| L | 245 | ~17 h |

Config S matches the handoff's "a few hours" assumption. **Config L does not** —
three seeds of the L rung is roughly 51 GPU-hours. Worth deciding before Phase B
whether that is acceptable or whether the L runs need a different arrangement.

## Phase A — COMPLETE (all 12 pre-registered runs, 3 seeds each)

Test OA at the best-validation epoch, config S (612,111 params), Morton ordering.

| Condition | grid side | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|---|
| GMP on, δ=0.125 | 18 | 0.7425 | 0.7415 | 0.7353 | **0.7398 ± 0.0032** |
| GMP **off** | — | 0.6558 | 0.6471 | 0.6659 | **0.6563 ± 0.0077** |
| GMP on, δ=0.25 | 9 | 0.6502 | 0.6468 | 0.6582 | 0.6517 ± 0.0048 |
| GMP on, δ=0.5 | 5 | 0.5902 | 0.6006 | 0.5975 | 0.5961 ± 0.0044 |

### The GMP ablation must be reported conditionally

With all three seeds in, **GMP-off (0.6563) is slightly ahead of GMP-on at
δ=0.25 (0.6517)**, and well ahead of GMP-on at δ=0.5 (0.5961). The gap at
δ=0.25 (0.0046) is smaller than the GMP-off seed spread (0.0077), so the honest
reading is "indistinguishable", not "GMP-off wins" — but it is certainly not
evidence that GMP helps.

At δ=0.125, GMP-on (0.7398) beats GMP-off by **8.4 points**, far outside seed
noise.

So the defensible claim is *not* "GMP helps". It is: **the GMP prior helps only
when the voxel grid is fine enough to resolve local structure; at coarse
resolution it is no better than no prior at all, and at very coarse resolution
it actively hurts.** Any writeup that quotes a single GMP on/off number without
stating δ would be misleading.

Note on the pre-registered stop condition ("if GMP-off ≥ GMP-on, stop and report
before Phase B"): read at the coarse anchor it is technically met; read at δ\*,
where the comparison is meant to be made, GMP-on wins decisively. Phase B was
launched at Zihan's direction with this understood. Flagged here so the decision
is on the record rather than implicit.

## Phase A extension — finer δ (beyond pre-registration)

Phase A's δ trend was monotonic and had not turned over (0.5 → 0.25 → 0.125
gave 0.595 → 0.653 → 0.734 test OA), so δ=0.125 may not be the optimum, only
the finest anchor tested. At Zihan's direction, three extra jobs probe finer
grids. **These are an extension beyond the pre-registered grid and are labelled
`phase: aext`.** Selection remains on validation only, so no test-set tuning is
introduced; recorded here for transparency.

- δ=0.0625 (grid side ~36), seeds 0 and 1 — running.
- δ=0.03125 (grid side ~71), seed 0 — **failed immediately with
  `ResourceExhaustedError`** on a 24 GB RTX 3090, as predicted from the grid
  tensor size (32 × 71³ × 128 × 4 B ≈ 5.9 GB per GMP grid, before conv output,
  gradients and per-block retention). This empirically fixes the memory ceiling
  for config S at batch 32 somewhere between grid side 36 and 71.

  Retried on an 80 GB A100: it runs, with no memory error — so grid side 71
  needs more than 24 GB and fits in 80 GB. But it costs **1614 s (26.9 min) per
  epoch**, i.e. ~4.7 days for one 250-epoch seed. Killed after 2 epochs: the
  δ curve is already flattening (see below), so the extrapolated gain does not
  justify roughly 8× the compute of δ=0.0625, and A100 capacity is contended.
  Recorded as a deliberate stop, not a failure.

**Accuracy vs cost across the δ ladder** (test OA at best-val epoch; the two
finer settings are still mid-run and improving, so their figures are lower
bounds):

| δ | grid side | s/epoch | run time | test OA |
|---|---|---|---|---|
| 0.5 | 5 | ~10 | ~0.7 h | 0.595 |
| 0.25 | 9 | ~14 | ~1.0 h | 0.649 |
| 0.125 | 18 | ~16–20 | ~1.2 h | 0.740 (3 seeds, complete) |
| 0.09375 | 24 | 47 | ~3.3 h | ≥0.750 (ep. 190/250) |
| 0.0625 | 36 | 203 | ~14 h | ≥0.757 (ep. 122/250) |
| 0.03125 | 71 | 1614 | ~4.7 days | abandoned |

Accuracy keeps improving as the grid refines, but with clear diminishing
returns: roughly +1.0 point from δ=0.125 to 0.09375, then +0.7 more to 0.0625,
while compute rises 3× then 4× again. This is a useful result in its own right
for a study about resource-efficient models — the best grid setting is an order
of magnitude more expensive than one that is only ~1.7 points worse.

**Cost:** δ=0.0625 runs at 203 s/epoch versus roughly 16–20 s/epoch at δ=0.125 —
about 10× slower, so ~14 h per 250-epoch run. Whether that is worth it depends
entirely on whether it beats δ=0.125's 0.734; the compute cost of the finest
grids is itself a finding worth reporting, given the study is about
resource-efficient models.

The handoff's "grid side ≤ 32" guidance turns out to be close to the true
hardware limit rather than merely conservative: side 36 works, side 71 does not.

## Phase B — scaling ladder (interim, most runs 225–250 epochs)

Test OA at best-validation epoch. Runs marked † were cut short by the
host-memory bug below at the stated epoch, and are being rerun with 32 GiB;
their best-val epoch had already passed, so the numbers are near-final but not
protocol-clean.

| Config | params | δ=0.125 | δ=0.09375 |
|---|---|---|---|
| XS | 156,559 | 0.7273 ± 0.012 | 0.7506 ± 0.003 † |
| S | 612,111 | 0.7398 ± 0.003 | 0.7580 (2 seeds) † |
| M | 1,214,479 | 0.7400 ± 0.005 † | 0.7632 ± 0.006 † |
| L | 2,712,591 | 0.7498 ± 0.004 † | **0.7715 ± 0.0025** (3 seeds, complete) |

Two clear results:

1. **δ=0.09375 beats δ=0.125 at every rung**, by 1.0–2.3 points. So δ\* is
   0.09375, not the 0.125 that the pre-registered three-anchor sweep would have
   selected. Probing finer than the handoff's anchors was necessary.
2. **Scaling is very flat.** At δ\*, going from XS to L is 17× the parameters
   for about +2 points (0.751 → 0.772). The architecture saturates quickly on
   this dataset; accuracy is limited by grid resolution far more than by
   capacity. XS at 156K params already reaches 0.751 — better than PointNet's
   published 68.2% at 3.5M params.

The best configuration so far is **L at δ=0.09375: 0.7715 ± 0.0025 OA,
0.7324 mAcc**, 2.71M params. Those three runs completed all 250 epochs (they
ran on A100s with 32 GiB host memory, so the leak never caught them).

## Sparse GMP — removes the grid-resolution cost barrier

The dense GMP grid costs O(B · side³ · C), which is why δ=0.03125 (side ~71)
needed an 80 GB A100 and 27 min/epoch, and why the δ sweep had to stop before
finding its optimum. But with 1024 points, **at most 1024 voxels are ever
occupied**, regardless of grid side — the dense grid is almost entirely zeros.

New variants in `models/gmp3d.py` (`sparse`, `sparse_mean`, `sparse_trilinear`)
materialize only occupied voxels: coalesce points via `tf.unique` +
`unsorted_segment_sum`, then evaluate the depthwise 3×3×3 convolution by
gathering each occupied voxel's 27 neighbours through a sorted-key
`tf.searchsorted` lookup. The existing `Conv3D` layer is reused purely as a
weight container, so the sparse path uses the identical kernel layout — which is
what makes equivalence provable rather than merely plausible. A test pins
sparse ≡ dense with shared weights.

Measured forward time (config-S dims, batch 4, 1024 points, CPU):

| δ | grid side | dense | sparse | speedup |
|---|---|---|---|---|
| 0.125 | 17 | 0.31 s | 0.24 s | 1.3× |
| 0.0625 | 33 | 1.36 s | 0.24 s | 5.7× |
| 0.03125 | 65 | 17.16 s | 0.39 s | **44×** |

**Sparse is essentially constant-time in grid resolution.** Dense scales as
side³; sparse scales with the occupied-voxel count, which is bounded by N.

This matters for the study's central finding. The δ sweep showed accuracy rising
monotonically (0.596 → 0.758) with no turnover, but the dense cost curve made
finer grids prohibitive, so we never found the optimum. Sparse GMP converts
"finer is 10× more expensive" into "finer is nearly free", making the true
optimum reachable. The dense path remains the default so all completed runs stay
reproducible.

Caveat not yet closed: equivalence is verified numerically on small inputs and
in unit tests, but no full sparse training run has completed yet. Accuracy
parity on a real run must be confirmed before any sparse result is reported.

## Published baselines — VERIFIED against original papers

Checked directly against the source PDFs (not from memory, per the handoff).
All ScanObjectNN PB_T50_RS.

| Method | Params | OA (%) | mAcc (%) | Source |
|---|---|---|---|---|
| PointNet | 3.5M | 68.2 | 63.4 | PointNeXt Tab. 2; PointMLP Tab. 3 |
| PointNet++ | 1.5M | 77.9 | 75.4 | both |
| DGCNN | 1.8M | 78.1 | 73.6 | both |
| PointCNN | 0.6M | 78.5 | 75.1 | PointNeXt Tab. 2 |
| BGA-DGCNN | — | 79.7 | 75.7 | PointMLP Tab. 3 |
| SimpleView | 0.8M | 80.5 ± 0.3 | — | both |
| MVTN | 3.5M | 82.8 | — | PointNeXt Tab. 2 |
| PointMLP-elite | 0.68M | 83.8 ± 0.6 | 81.8 ± 0.8 | PointMLP Tab. 3 |
| PointMLP | 13.2M | 85.4 ± 1.3 | 83.9 ± 1.5 | PointNeXt Tab. 2 |
| PointNet++ (PointNeXt training) | 1.5M | 86.1 ± 0.7 | 84.2 ± 0.9 | PointNeXt Tab. 2 |
| PointNeXt-S | 1.4M | 87.7 ± 0.4 | 85.8 ± 0.6 | PointNeXt Tab. 2 |

Sources: PointNeXt, NeurIPS 2022, Table 2; PointMLP, ICLR 2022 (arXiv 2202.07123),
Table 3. PointMLP states it does *not* use the voting strategy, matching our
protocol.

**Corrections to the handoff's from-memory table:** PointMLP has **13.2M**
parameters, not 12.6M. PointNet, PointNet++, DGCNN, PointMLP-elite and
PointNeXt-S figures were accurate.

### The critical context for interpreting our gap

PointNeXt's central finding is that **PointNet++ improves from 77.9 to 86.1 OA
(+8.2) with no architecture change at all** — purely from modern training
strategy (augmentation and optimization). Their abstract states this explicitly.

That reframes our comparison. Our recipe is deliberately simple (AdamW, cosine
with warmup, label smoothing 0.2, 250 epochs, batch 32, scale + y-rotation, 1024
points, no voting). PHAT-JeT-L's 77.2 sits essentially at the level of the
*original-recipe* PointNet++ (77.9) and DGCNN (78.1), not at the level of
modern-recipe results. So the ~10-point gap to PointNeXt-S is **not
demonstrably an architecture gap** — a large part of it is plausibly a training
gap that we have not attempted to close.

This must be stated in the writeup. Claiming an architectural conclusion from
this comparison without controlling for training strategy would be exactly the
error PointNeXt was written to expose.

## Host-memory bug — the OOM wave

After Phase B launched, 23 pods were OOMKilled: the entire M and L rungs plus
several XS runs. Resident host memory grows with epoch count and grows faster
with larger `d_model` and larger GMP grids, so the bigger configs died sooner
(M at ~192 epochs, L at ~230, XS at ~240). This is cgroup host memory, not GPU.

Root cause is under fix (per-epoch `model.fit()` / `model.predict()` calls
rebuilding data adapters and retracing). The empirical mitigation is simply more
host RAM: the three A100 L runs with 32 GiB completed 250 epochs on the *same*
code, while their 14 GiB counterparts died. All reruns now request 32 GiB.

**No completed result was lost**, because a snapshot CronJob (added after the
first incident) copies each run's metrics.json to metrics.snapshot.json every
10 minutes, only when the epoch count increases. That monotonicity also makes
relaunching into the same output directory safe: a fresh run starting at epoch 0
cannot regress the snapshot.

## Subagent note

The Codex task that authored the rotation fix wedged after completing its work:
it applied its three file changes, ran the Keras 2 test suite (exit 0), launched
the Keras 3 suite, and then produced no further output for 2 h 21 min. No python
process was alive and the machine was idle at load 0.55, so the command had
exited without Codex recording it. The process was killed. Its code changes were
already reviewed, independently verified (50 tests under both runtimes) and
committed, so only its narrative summary was lost.

## Environment

Cluster image is TF 2.13.0 / Keras 2.13.1; local development is TF 2.21 /
Keras 3. The full test suite is run under both. This caught a real portability
bug (`Variable.path` is Keras 3 only). The *original* jet code cannot build
under Keras 3 at all — pre-existing and unrelated to this port, verified by
building the unmodified file from `main`.

Cluster policy: container limit/request ratio must be ≤ 1.2.

## Deviations from the handoff so far

See `PLAN_scanobjectnn.md` §8 for the full list. Summary: TensorFlow rather than
the torch pseudocode; ladder kept as specified with measured (≈1.5× target)
params reported; Phase A = 3 δ anchors / 12 jobs; GMP-off retains Morton
ordering and is described as such; Phase D reuses P=32; GMP kernel 3×3×3;
rotation about y; per-epoch numpy epoch construction instead of tf.data;
val-based checkpoint selection with test-curve max reported as descriptive only.
