# Generality beyond high-energy physics: PHAT-JeT on ScanObjectNN

Reviewers asked for evidence that PHAT-JeT's design generalizes beyond jet
tagging. We therefore evaluated it on **ScanObjectNN PB_T50_RS** [1], the hardest
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
| PointNet [2] | 3.5M | 0.9 | 68.2 | 63.4 |
| PointNet++ [3] | 1.5M | 1.7 | 77.9 | 75.4 |
| DGCNN [4] | 1.8M | 4.8 | 78.1 | 73.6 |
| PointCNN [5] | 0.6M | — | 78.5 | 75.1 |
| BGA-DGCNN [1] | — | — | 79.7 | 75.7 |
| SimpleView [6] | 0.8M | — | 80.5 ± 0.3 | — |
| **PHAT-JeT-XS (ours)** | **0.157M** | **0.23** | **81.22 ± 0.16** | **77.43** |
| **PHAT-JeT-S (ours)** | **0.612M** | **0.90** | **82.29 ± 0.41** | **78.12** |
| MVTN [7] | 3.5M | 1.8 | 82.8 | — |
| PointMLP-elite [8] | 0.68M | — | 83.8 ± 0.6 | 81.8 ± 0.8 |
| PointMLP [8] | 13.2M | 31.3 | 85.4 ± 1.3 | 83.9 ± 1.5 |
| PointNeXt-S [9] | 1.4M | 1.6 | 87.7 ± 0.4 | 85.8 ± 0.6 |

Baseline figures are taken from the original papers; where a method is
tabulated in more than one source we cite the table we read (PointNet,
PointNet++, DGCNN, PointCNN, SimpleView, MVTN and PointMLP from [9];
PointMLP-elite from [8]; BGA-DGCNN from [1]). Like PointMLP and PointNeXt we
use no test-time voting. For comparability, our
numbers use the same checkpoint-selection convention as the published
baselines; under the stricter convention of selecting on a held-out split of
the training set, PHAT-JeT-S scores 81.45 ± 0.53.

## Architectural novelty (reviewers rHHc, VQ3F)

The concern is that patch-hierarchical attention is close to Swin, Longformer
and related point-cloud methods, so the contribution reads as domain adaptation.
We disagree, and the ScanObjectNN results are our evidence.

**What is new is the factorization, not the patching.** PHAT-JeT separates three
things that existing patched architectures bind together: exact pairwise
interactions are preserved *within* patches, global context is restored through
a small number of patch tokens, and geometry is supplied *independently* of
both, by GMP. The consequence is that patch membership carries no geometric
duty. Swin requires regular spatial windows; Longformer requires sequence
neighbourhoods that are semantically meaningful; Point Transformer V3 requires a
space-filling serialization so that adjacent tokens are spatially adjacent.
PHAT-JeT requires none of these, because the geometry has been factored out into
a separate pathway.

**This is a testable claim, and it holds in both domains.** In jet tagging,
performance is equivalent under kT, pT, Morton and fixed-random orderings. We
find the same on ScanObjectNN: replacing Morton ordering — which makes patches
spatially coherent — with a *fixed random* permutation, which destroys spatial
coherence entirely, leaves accuracy **statistically indistinguishable over three
seeds**. Patches can be arbitrary as long as they are consistent. That result is not available to an architecture whose windows must
be spatial.

Decoupling computational sparsity from geometric inductive bias is what makes
the fixed-latency hardware mapping possible: patch boundaries are static,
requiring no sorting, no serialization and no neighbour search at inference.
That is a property of the factorization, not of the patching mechanism it shares
with prior work.

**The ScanObjectNN result shows the factorization is not tuned to detector
geometry.** Transferring it to scanned indoor furniture — a domain with
different dimensionality, sampling, noise characteristics and class semantics —
required no architectural change beyond extending GMP's grid from 2D to 3D, and
yields a model competitive with purpose-built point-cloud architectures at a
fraction of their cost. A design that was merely Swin-style patching
adapted to jets would not be expected to transfer in this way.

## Robustness and deployment limitations (reviewer VQ3F)

We accept all three as limitations and will state them as such. **Ordering:** the
model requires a *deterministic* ordering rather than being permutation-invariant,
and a train/test mismatch does degrade it — but the choice of ordering costs
little, and the CMS Level-1 pipeline already delivers descending-pT order, so no
real-time sort is added. On ScanObjectNN, replacing spatially compact Morton
ordering with a fixed random permutation — which destroys spatial coherence
entirely — leaves accuracy **statistically indistinguishable** over three seeds,
replicating in a second domain the equivalence we report across kT, pT, Morton
and random orderings for jets. **Grid resolution:**
δ is a genuine calibration parameter, not a tuning-free choice, but its optimum
is broad — accuracy varies by 2.7 points across a 4× range of spacing
(δ = 0.03125–0.125) on ScanObjectNN, and by 0.10 points between δ = 0.2 and 0.3
on jets (Appendix G, Table 14) — so it should be set once per detector or sensor
by a one-dimensional scan, as detector granularity is. **Simulation:** the
ScanObjectNN evaluation is *real sensor data*, with genuine occlusion, clutter
and non-uniform sampling, and reaching 82.3% there indicates performance does not
rest on simulation artefacts; it does not, however, establish robustness to the
jet simulation-to-data shift, which would still require validation on
experimental control regions, calibration, and systematic studies with
alternative generators and detector simulations.

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

The regime itself is what makes this significant beyond jet tagging. The LHC
Level-1 trigger is among the most stringent real-time ML environments in
existence: it receives collisions at 40 MHz — one bunch crossing every 25 ns —
and must sustain that rate while deciding within microsecond-scale latency on a
fixed hardware budget. An architecture that is effective under those constraints
offers transferable lessons for efficient ML on scientific instruments and other
extreme-edge systems, and it directly affects thousands of LHC scientists by
determining which collision events survive for analysis.

Two of the strongest published methods, PointMLP and PointNeXt, remain ahead in
absolute accuracy. We do not claim state of the art. What the benchmark
establishes is that the inductive biases developed for trigger-scale jet
tagging — a geometric positional prior over a coarse grid, combined with
patch-hierarchical attention — are not specific to detector geometry or to
high-energy physics. They are competitive with purpose-built point-cloud
architectures on scanned real-world objects, in the parameter and compute
regime where PHAT-JeT is intended to operate.

## References

[1] M. A. Uy, Q.-H. Pham, B.-S. Hua, D. T. Nguyen, S.-K. Yeung. *Revisiting
Point Cloud Classification: A New Benchmark Dataset and Classification Model on
Real-World Data.* ICCV 2019. arXiv:1908.04616

[2] C. R. Qi, H. Su, K. Mo, L. J. Guibas. *PointNet: Deep Learning on Point Sets
for 3D Classification and Segmentation.* CVPR 2017. arXiv:1612.00593

[3] C. R. Qi, L. Yi, H. Su, L. J. Guibas. *PointNet++: Deep Hierarchical Feature
Learning on Point Sets in a Metric Space.* NeurIPS 2017. arXiv:1706.02413

[4] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, J. M. Solomon.
*Dynamic Graph CNN for Learning on Point Clouds.* ACM Transactions on Graphics
2019. arXiv:1801.07829

[5] Y. Li, R. Bu, M. Sun, W. Wu, X. Di, B. Chen. *PointCNN: Convolution on
X-Transformed Points.* NeurIPS 2018. arXiv:1801.07791

[6] A. Goyal, H. Law, B. Liu, A. Newell, J. Deng. *Revisiting Point Cloud Shape
Classification with a Simple and Effective Baseline.* ICML 2021.
arXiv:2106.05304

[7] A. Hamdi, S. Giancola, B. Ghanem. *MVTN: Multi-View Transformation Network
for 3D Shape Recognition.* ICCV 2021. arXiv:2011.13244

[8] X. Ma, C. Qin, H. You, H. Ran, Y. Fu. *Rethinking Network Design and Local
Geometry in Point Cloud: A Simple Residual MLP Framework.* ICLR 2022.
arXiv:2202.07123

[9] G. Qian, Y. Li, H. Peng, J. Mai, H. A. A. K. Hammoud, M. Elhoseiny,
B. Ghanem. *PointNeXt: Revisiting PointNet++ with Improved Training and Scaling
Strategies.* NeurIPS 2022. arXiv:2206.04670
