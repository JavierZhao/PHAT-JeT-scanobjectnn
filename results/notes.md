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

**Cost:** δ=0.0625 runs at 203 s/epoch versus roughly 16–20 s/epoch at δ=0.125 —
about 10× slower, so ~14 h per 250-epoch run. Whether that is worth it depends
entirely on whether it beats δ=0.125's 0.734; the compute cost of the finest
grids is itself a finding worth reporting, given the study is about
resource-efficient models.

The handoff's "grid side ≤ 32" guidance turns out to be close to the true
hardware limit rather than merely conservative: side 36 works, side 71 does not.

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
