# RibAssist 3D

**Biplanar rib-fracture detection, anatomical addressing, and selective 3D localization.**

RibAssist 3D is an end-to-end research and proof-of-concept review-assistance system for highlighting suspected rib fractures in AP and lateral CT-derived projections, assigning predicted side and rib level, and selectively localizing sufficiently confident findings on interactive 3D rib anatomy. When cross-view correspondence is uncertain, the system abstains from producing a 3D point while preserving the original detection and available anatomical addressing for human review.

The project combines staged failure analysis, detector retraining, frozen-policy evaluation, reproducibility checks, and an interactive clinician-review workflow.

> **Sealed-test result:** under a protocol frozen before opening the 55-case test cohort, the retrained lateral detector reconstructed **15 of 601 fractures within 10 mm** at **0.436 false 3D points per case**, compared with zero reconstructions from the original detector under its frozen policy satisfying the same false-output budget. Case-bootstrap recall was **2.50% (95% CI: 0.69%–4.48%)**.
>
> This is a directional research proof of concept, not a diagnostic or deployable clinical system.

![Interactive RibAssist demo](figures/ribassist_demo.gif)

- **Paper:** [RibAssist 3D: Biplanar 3D Rib-Fracture Reconstruction from CT-Derived Projections](paper/RibAssist_3D.pdf)


## What this project shows

The central question: *can fractures detected independently in AP and lateral projections be paired across views and triangulated into reliable 3D points at a pre-specified controlled false-output budget?*

The staged program (A to L3) answers it by **identifying the effective operational bottleneck** under the tested detector and correspondence methods:

![Locating the effective operational bottleneck](figures/fig_bottleneck.svg)

The initial failure arose from cross-view correspondence operating on weak lateral detections. Testing deterministic and learned local pair-scoring showed that neither moved the controlled-false-rate frontier on the frozen detector. This redirected the investigation toward lateral-detector quality, which turned out to be the effective lever.

**Main conclusion.** Geometry and detector localization were accurate when the correct cross-view observations were supplied. The operational failure occurred because the weak lateral detector produced low-confidence true observations within a noisy candidate graph. Candidate-field cleanup alone did not move the controlled-budget frontier; improving lateral-detector confidence did. Neither deterministic assignment nor learned local-appearance scoring produced meaningful controlled-budget recall on the frozen detections, whereas lateral retraining produced the first nonzero controlled-false-rate reconstructions, and the fixed-policy result transferred to the untouched sealed cohort.

![Sealed frozen-vs-L2 result](figures/main_results.png)

The 2×2 factorial attributes the observed gain to the **detector intervention** rather than to either of the tested correspondence methods: the learned appearance scorer does not beat detector confidence at the operational budget.

![Detector by correspondence attribution](figures/fig_2x2_attribution.svg)

**Reading the 2.5%.** The low recall means RibAssist 3D is not a comprehensive or deployable reconstructor. It does establish reproducible, geometrically accurate selective localization: the committed sealed-test points had a median matched distance of 1.49 mm and 93% rib-exactness. When 3D correspondence abstains, the underlying detection and rib-addressing outputs remain available for review.

## Assistive use

RibAssist 3D is designed as a proof-of-concept assistive review system rather than a standalone diagnostic method.

Even when cross-view correspondence is too uncertain for 3D reconstruction, the system can still assist a reviewer by:

1. highlighting suspected fracture regions in the AP and lateral projections;
2. assigning a predicted side and rib level to organize findings anatomically;
3. preserving uncommitted detections for human review instead of discarding them;
4. selectively placing high-confidence findings on interactive 3D rib anatomy;
5. showing the projection evidence and 3D location together during per-finding review;
6. allowing the reviewer to accept, reject, or mark findings as needing further review.

The 3D layer is therefore **additive**. An abstention means the system declines to make an unreliable 3D localization claim; it does not remove the underlying fracture detection or rib-level information.

The current evidence supports RibAssist 3D as a workflow proof of concept for fracture highlighting, anatomical organization, and selective localization. It does not establish improved diagnostic accuracy, reduced reading time, or clinical benefit, because those outcomes would require a clinician reader study.

## Interactive demo

The demo presents RibAssist 3D as an assistive review workflow. Detection remains available for every reported finding, and rib addressing is shown where produced by the addressing model, while 3D localization is added only when a cross-view match satisfies the frozen confidence policy.

> Both paths run the trained models, so they require the datasets and the pretrained checkpoints, which are prepared separately and are **not** committed to the repository. Paths such as `outputs/…` and `data/…` in the commands below refer to those local artifacts.

**Streamlit clinician-review app** (full, model-backed):

```bash
pip install -r requirements-demo.txt
streamlit run app.py
```

A three-stage workflow:

1. case overview with interactive 3D rib anatomy;
2. per-finding review with AP, lateral, and focused 3D evidence;
3. case summary with reviewer decisions and localization status.

Detection and addressing confidence, cross-view status, and accept / needs-review / reject actions are shown throughout.

**Static self-contained review page** (no server): generate a single HTML file with real model outputs.

```bash
python scripts/export_review_site.py \
    --champion-run outputs/detector_dev_scratch_c32_both_gated \
    --address-model outputs/addressing_model_ap_nopos \
    --data outputs/det_out_v2/det_test.npz \
    --pairs-npz outputs/sealed/L2_sealed_D0_pairs.npz --policy outputs/sealed/L2_policy.json \
    --image-dirs data/ribfrac_train data/ribfrac --seg-dir data/ribseg/ribseg_v2/seg --cl-dir data/ribseg/ribseg_v2/cl \
    --expected-data-sha256 <det_test sha> \
    --template scripts/ribassist_review_template.html --out outputs/ribassist_review.html
```

In both, a detection is **never** removed because the 3D correspondence abstains; it is retained with "3D localization unavailable, manual review recommended."

Model localization status and reviewer decisions are separate concepts in the interface:

- **Localized:** the model committed a cross-view pair and emitted a 3D point.
- **Candidate:** a compatible pair exists, but confidence was insufficient to commit.
- **Rib-level only:** a detection and anatomical address remain available without a reliable 3D point.
- **Accepted / Needs review / Rejected:** human-review decisions recorded in the interface.

![Pipeline architecture](figures/fig_architecture.svg)

## Repository layout

```
RibAssist-3D/
├── app.py                        # Streamlit clinician-review demo (entry point)
├── demo_app/                     # demo app package (viewers, 3D scene, pipeline, UI)
├── scripts/                      # pipeline, evaluation, and figure scripts (see below)
├── figures/                      # architecture / bottleneck / results / attribution figures
├── tests/                        # unit tests
└── outputs/sealed/               # headline sealed-evaluation result JSONs (tracked)
```

**Not committed:** the `data/` datasets, model checkpoints, and large `.npz` / `.pt` tensors under `outputs/`;
these are prepared separately. Only the small headline sealed-evaluation JSONs in `outputs/sealed/` are tracked,
so the main-results figure regenerates from a fresh clone without any download.

**Key scripts** (`scripts/`): `train_detector.py`, `train_address.py`, `train_lateral_L2.py` (models);
`run_ribassist.py` (detection + addressing inference); `eval_biplanar_geometry*.py` (Stages A-C),
`eval_correspondence_D0/D1/D2*.py` (candidate graph, deterministic and learned correspondence),
`eval_lateral_L0*/L01*.py` (lateral diagnosis and calibration); `build_sealed_det_npz.py` and
`run_sealed_stage{1,2}*.sh` (sealed evaluation); `make_main_results_figure.py`, `make_demo_figures.py`,
`export_review_site.py` (figures and demo).

## Quickstart

Requires Python 3.10+, PyTorch (Apple MPS or CUDA for training; CPU for evaluation), NumPy, SciPy, nibabel,
scikit-learn, and matplotlib. The demo additionally needs the packages in `requirements-demo.txt`.

The datasets and model checkpoints are prepared separately and are **not** committed (they exceed Git limits and
include licensed derived data). The headline sealed-evaluation result JSONs **are** included under
`outputs/sealed/`, so the main-results figure regenerates directly from the repository with no download. To obtain
the datasets and reproduce the full pipeline, follow [`DATA_SETUP.md`](DATA_SETUP.md) (licensed data) and
[`REPRODUCE.md`](REPRODUCE.md) (end-to-end steps, hashes, and the sealed pass). Once the artifacts are in place,
launch the demo with `streamlit run app.py`.

## Method (one paragraph)

A CT volume is rendered to AP and lateral orthographic projections. A per-view U-Net emits a fracture heatmap;
peaks are extracted per view and paired into a candidate graph gated by shared-axis (SI) agreement. A one-to-one
**assignment with abstention** commits a correspondence only when the configured geometric and detector-confidence
cost clears the frozen abstention threshold, and each committed pair is back-projected and triangulated to a 3D point. A separate addressing model
predicts each detection's rib level. Detector, extraction policy, correspondence configuration, and evaluation are
frozen before the sealed test; all provenance is hash-checked and fails closed.

## Model architecture (layer by layer)

Three trained networks. The biplanar detector is **two independent single-channel U-Nets** (one per view);
cross-view "fusion" is a geometric candidate union on the shared superior-inferior (SI) axis, not a learned
layer, so there is no fusion network to describe. The addressing network and the learned pair-scorer are the two
auxiliary models.

![Layer-by-layer model architecture](figures/fig_model_layers.svg)

### Per-view detector U-Net (base channels = 32)

Input is one 256×256 orthographic projection (single channel); output is a 256×256 fracture heatmap in [0, 1]
(sigmoid). Every `DoubleConv` is Conv 3×3 (pad 1), BatchNorm, ReLU, Conv 3×3 (pad 1), BatchNorm, ReLU.
Downsampling is 2×2 max-pool; upsampling is bilinear interpolation to the skip resolution followed by channel
concatenation.

| Stage | Operation | Output (C × H × W) | Params |
|---|---|---|---|
| Input | projection | 1 × 256 × 256 | 0 |
| `inc` | DoubleConv 1 -> 32 | 32 × 256 × 256 | 9,696 |
| `d1` | MaxPool, DoubleConv 32 -> 64 | 64 × 128 × 128 | 55,680 |
| `d2` | MaxPool, DoubleConv 64 -> 128 | 128 × 64 × 64 | 221,952 |
| `d3` (bottleneck) | MaxPool, DoubleConv 128 -> 256 | 256 × 32 × 32 | 886,272 |
| `u3` | Up + concat `d2`, DoubleConv 384 -> 128 | 128 × 64 × 64 | 590,592 |
| `u2` | Up + concat `d1`, DoubleConv 192 -> 64 | 64 × 128 × 128 | 147,840 |
| `u1` | Up + concat `inc`, DoubleConv 96 -> 32 | 32 × 256 × 256 | 37,056 |
| `outc` | Conv 1×1 32 -> 1, sigmoid | 1 × 256 × 256 | 33 |

Total: **1,949,121 parameters per view**. The deployed detector runs two independent copies (AP and lateral),
about 3.9M parameters combined. The L2 retrained lateral head has the identical shape (warm-started from the
champion lateral weights, BatchNorm, focal positive-branch weight 3.0, periodic hard-negative re-mining). A
training-only auxiliary rib-region head (a second 1×1 output conv on the shared decoder features) exists in some
checkpoints but is unused at inference; the deployed forward path returns only the fracture heatmap.

### Addressing network (rib-level assignment)

Two identical CNN streams (one per view), each 1 -> 16 -> 32 -> 64 channels over three [Conv 3×3 (pad 1),
BatchNorm, ReLU, MaxPool 2×2] blocks, then AdaptiveAvgPool to 1×1 and flatten to a 64-d embedding. The two
embeddings concatenate to 128-d and feed three linear heads. It runs on the RAW projections.

| Component | Operation | Output | Params |
|---|---|---|---|
| AP stream | 3x [Conv-BN-ReLU-MaxPool] 1 -> 16 -> 32 -> 64, AdaptiveAvgPool, Flatten | 64 | 23,520 |
| Lateral stream | same as AP stream | 64 | 23,520 |
| concat | [ap, lat] | 128 | 0 |
| side head | Linear 128 -> 1 (sigmoid) | 1 | 129 |
| rib head | Linear 128 -> 12 (softmax) | 12 | 1,548 |
| quality head | Linear 128 -> 1 (sigmoid) | 1 | 129 |

Total: **48,846 parameters**. The reported `address_score` is side confidence times rib probability. The auxiliary quality head is retained as implemented but is not the cross-view correspondence scorer (that is the separate learned pair-scorer below).

### Learned pair-scorer (correspondence, D2b)

Used only in the 2×2 attribution, to test whether learned local appearance beats detector confidence for
cross-view matching. A shared two-layer strided-conv tower embeds each 40×40 crop (AP and lateral) to 32-d; the
head scores the pair from [ap_emb, lat_emb, ap_emb × lat_emb] plus 6 geometric and confidence scalars
(normalized |dSI|, AP score, lateral score, and the min, product, and asymmetry of the two confidences).

| Component | Operation | Output | Params |
|---|---|---|---|
| crop (per view) | 40 × 40 patch | 1 × 40 × 40 | 0 |
| tower (shared) | Conv 3×3 s2 1 -> 16, BN, ReLU; Conv 3×3 s2 16 -> 32, BN, ReLU; AdaptiveAvgPool; Linear 32 -> 32 | 32 | 5,952 |
| head | Linear (32×3 + 6 = 102) -> 64, ReLU, Dropout 0.4, Linear 64 -> 1 | 1 | 6,657 |

Total: **12,609 parameters**. At the operational budget it does not beat detector confidence, which is why the
2×2 attributes the gain to the detector, not the correspondence method.

## Engineering and research areas

A compact map of the techniques this project exercises end to end:

- Medical-image computer vision
- Multi-view geometric reconstruction
- Detector calibration and hard-negative mining
- Assignment with abstention
- Case-level cross-validation and sealed evaluation
- Artifact provenance and reproducibility
- Interactive Streamlit and Plotly visualization
- Human-in-the-loop healthcare UX

## Status, limitations, and disclaimer

RibAssist 3D is a **research proof of concept**, not a medical device or clinical tool. It currently supports
fracture-region highlighting, rib-level addressing, and selective 3D localization for high-confidence cross-view
matches.

The sealed result remains limited: recall was 2.5%, and only 6 of 55 cases produced at least one correct
reconstruction. The cohort was small, contained no validated negative-scan safety evaluation, and used CT-derived
simulated orthographic projections rather than independently acquired clinical radiographs.

Within the tested detector and local-correspondence methods, lateral-detector availability and confidence were the
effective lever. The next research directions are stronger lateral detection, global anatomical correspondence
using rib identity and ordering, evaluation on negative cases, and validation on more realistic or independently
acquired biplanar inputs.

**Do not use RibAssist 3D for diagnosis, treatment, or patient care.**

## Data

Datasets are third-party and governed by their own licenses and terms; they are **not** redistributed here:

- [RibFrac dataset](https://ribfrac.grand-challenge.org/)
- [RibSeg v2](https://github.com/M3DV/RibSeg)

## License

- Original source code: [Apache License 2.0](LICENSE)
- Original documentation and figures: [CC BY 4.0](LICENSE-DOCS)
- Model checkpoints: not distributed; see [MODEL_TERMS.md](MODEL_TERMS.md)
- Third-party datasets: governed by their original licenses and not redistributed

## Citation

If you use RibAssist 3D, please cite this repository together with the RibFrac and RibSeg datasets:

```bibtex
@article{soboka2026ribassist,
  title   = {RibAssist 3D: Biplanar 3D Rib-Fracture Reconstruction from CT-Derived Projections},
  author  = {Soboka, Kabila Haile},
  year    = {2026}
}
```