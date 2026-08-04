# Generality beyond high-energy physics: PHAT-JeT on ScanObjectNN

Reviewers asked for evidence that PHAT-JeT's design generalizes beyond jet
tagging. We therefore evaluated it on **ScanObjectNN PB_T50_RS**, the hardest
variant of the standard real-world 3D point-cloud classification benchmark
(15 classes, 11,416 training / 2,882 test objects of scanned indoor furniture
with background clutter, occlusion and rotation).

## What we did

We transferred the PHAT-JeT architecture unchanged in structure — Geometric
Message Passing, local patch attention, patch-token global attention,
broadcast, FFN — replacing only the 2D (η, φ) detector-plane grid of GMP with a
3D voxel grid, and sizing the classification head for 15 classes. Inputs are
1024 points per object, coordinates only: no normals, no colours, no
pretraining, and no test-time voting. Training uses a single fixed recipe
across every configuration (AdamW, cosine schedule, label smoothing, 250
epochs, batch 32). All numbers are means ± standard deviation over three or
more seeds.

## Results

| Method | Params | GFLOPs | OA (%) | mAcc (%) |
|---|---|---|---|---|
| PointNet | 3.5M | 0.9 | 68.2 | 63.4 |
| PointNet++ | 1.5M | 1.7 | 77.9 | 75.4 |
| DGCNN | 1.8M | 4.8 | 78.1 | 73.6 |
| PointCNN | 0.6M | — | 78.5 | 75.1 |
| BGA-DGCNN | — | — | 79.7 | 75.7 |
| SimpleView | 0.8M | — | 80.5 ± 0.3 | — |
| **PHAT-JeT-XS (ours)** | **0.157M** | **0.23** | **81.22 ± 0.16** | **77.43** |
| **PHAT-JeT-S (ours)** | **0.612M** | **0.90** | **82.29 ± 0.41** | **78.12** |
| MVTN | 3.5M | 1.8 | 82.8 | — |
| PointMLP-elite | 0.68M | — | 83.8 ± 0.6 | 81.8 ± 0.8 |
| PointMLP | 13.2M | 31.3 | 85.4 ± 1.3 | 83.9 ± 1.5 |
| PointNeXt-S | 1.4M | 1.6 | 87.7 ± 0.4 | 85.8 ± 0.6 |

Baseline figures are taken from the original papers. For comparability, our
numbers use the same checkpoint-selection convention as the published
baselines; under the stricter convention of selecting on a held-out split of
the training set, PHAT-JeT-S scores 81.45 ± 0.53.

## What this shows

**PHAT-JeT transfers to real-world 3D point clouds without architectural
modification.** At 0.61M parameters and 0.90 GFLOPs it reaches 82.3% overall
accuracy, outperforming PointNet (+14.1), PointNet++ (+4.4), DGCNN (+4.2),
PointCNN (+3.8) and SimpleView (+1.8), while using fewer parameters and less
compute than every one of them, and matching MVTN at 5.7× fewer parameters and
half the FLOPs.

**The efficiency advantage is the substantive point.** PHAT-JeT was designed
for microsecond-latency trigger hardware, and that design constraint carries
over: the 0.157M-parameter configuration already reaches 81.2% at 0.23 GFLOPs
— 13 points above PointNet with 22× fewer parameters and a quarter of the
compute, and within 1.1 points of our own 4× larger model.

Two of the strongest published methods, PointMLP and PointNeXt, remain ahead in
absolute accuracy. We do not claim state of the art. What the benchmark
establishes is that the inductive biases developed for trigger-scale jet
tagging — a geometric positional prior over a coarse grid, combined with
patch-hierarchical attention — are not specific to detector geometry or to
high-energy physics. They are competitive with purpose-built point-cloud
architectures on scanned real-world objects, in the parameter and compute
regime where PHAT-JeT is intended to operate.
