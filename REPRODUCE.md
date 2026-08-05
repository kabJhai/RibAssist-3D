# RibAssist 3D: Reproducibility

This document is the end-to-end path from **licensed third-party data** to the **sealed headline result**
and the **Streamlit demo**. The repository ships all scripts, published case-id splits, and headline sealed
JSONs. It does **not** ship RibFrac/RibSeg volumes, derived `.npz` tensors, or model checkpoints. Obtain
those under their licenses ([DATA_SETUP.md](DATA_SETUP.md)) and run the steps below.

Every stage verifies dataset, checkpoint, and protocol hashes before producing numbers; a mismatch aborts
the run (fail-closed).

All commands run from the repository root with the project virtualenv active.

## 0. Quickest check (no data download)

The headline sealed-evaluation JSONs are committed under `outputs/sealed/`, so a fresh clone can regenerate the
main-results figure and confirm the reported numbers without any dataset, checkpoint, or GPU:

```bash
pip install -r requirements.txt
python scripts/validate_local_artifacts.py            # verifies the published split hashes
python scripts/make_main_results_figure.py \
    --sealed-dir outputs/sealed --out figures/main_results.png
```

The regenerated figure reports frozen `recall@10 0.0%` (0/601) versus L2 `recall@10 2.50%` (15/601) at 0.436
false-3D/case, case-bootstrap 95% CI 0.69%-4.48%. Everything below reproduces those JSONs from raw data.

## 1. What is published vs local

| In git | You prepare locally |
|---|---|
| All `scripts/` for retrain + sealed eval + demo | RibFrac CT + labels, RibSeg seg/cl |
| `frozen_split.json`, `geometry_split.json`, `split_manifest.json` | `outputs/det_out_v2/*.npz`, checkpoints |
| `outputs/sealed/*.json` (headline sealed results) | Rib atlas, addressing weights, sweep dirs |

**Use the published splits as-is** (`frozen_split.json` for the sealed test cohort; `geometry_split.json` for
the atlas holdout). Regenerating splits with `freeze_split.py` / `freeze_geometry_split.py` is optional and
will not match the published hashes unless you intentionally replace them.

Validate splits after clone:

```bash
python scripts/validate_local_artifacts.py
```

After you finish the pipeline, verify your local artifacts match the published fingerprints:

```bash
python scripts/validate_local_artifacts.py --demo --strict
python scripts/validate_local_artifacts.py --all --strict
```

## 2. Environment

- Python **3.10+** (see `pyproject.toml`), packages in `requirements.txt` (NumPy, SciPy, PyTorch, nibabel,
  scikit-learn, matplotlib). Demo: `requirements-demo.txt` (Streamlit, Plotly).
- Detector training uses Apple MPS or CUDA; evaluation and figure generation are CPU-only.
- Data layout after [DATA_SETUP.md](DATA_SETUP.md):
  - `data/ribfrac_train/`, `data/ribfrac/`: RibFrac CT + labels
  - `data/ribseg/ribseg_v2/seg/`, `data/ribseg/ribseg_v2/cl/`: RibSeg v2

## 3. Pinned artifact inventory

These are the **expected sha256 digests** once you complete the pipeline. The demo and sealed-evaluation
artifacts (assembled `det_test.npz`, both detector checkpoints, the addressing model, and the sealed policy /
pairs files) are pinned in `split_manifest.json` and checked by `validate_local_artifacts.py`. The `det_dev.npz`
and the two sealed source tensors (`det_test_inputs.npz`, `det_test_gt.npz`) are instead verified through
`det_manifest.json`, which `make_det_data.py` writes and every downstream stage re-checks (fail-closed).

| artifact | sha256 (prefix … suffix) |
|---|---|
| `det_dev.npz` | `66ce66cef9006a66…0c0eb114` |
| assembled `det_test.npz` | `75c62ab5286dedbf…fdb0514d` |
| sealed source `det_test_inputs.npz` | `61a2b5f74d8bcc27…eea44e79` |
| sealed source `det_test_gt.npz` | `0dc0c4afb3a156cc…d854c1f6` |
| champion AP `detector_ap.pt` | `4adeffbe3a231eab…edf1d125` |
| champion lateral `detector_lat.pt` (frozen head) | `580d6289dd3af082…a36f0d8ab` |
| L2 lateral `detector_lat.pt` | `afc1192a6dad744a…a2c57b04` |
| addressing model | `e6a427b679a11e53…1c097aa` |

**Frozen policies (selected on development, applied verbatim to sealed):**

- frozen detector: gate 6, cost `min_conf`, mutual_best `false`, u `0.0` (abstain by construction)
- L2 detector: gate 30, cost `geomean_conf`, mutual_best `false`, u `0.41666666666666663`
- Extraction: AP `nms 5 / floor 0.05`; lateral frozen `nms 3 / floor 0.06`; L2 `nms 3 / floor 0.10`

## 4. Build the detector dataset

Uses the **published** `frozen_split.json` (do not open the sealed test until Section 10).

```bash
python scripts/make_det_data.py \
    --ribfrac-dir data/ribfrac_train data/ribfrac \
    --split-json frozen_split.json \
    --out outputs/det_out_v2 \
    --size 256 --sigma 4 --overlays 12
```

Produces `det_dev.npz`, `det_test_inputs.npz`, `det_test_gt.npz`, and `det_manifest.json`. Training code
loads **only** `det_dev.npz`; sealed sources stay separate until assembly.

## 5. Train the champion detector (frozen head)

From-scratch dual-view U-Net, base channels 32, 80 epochs. Checkpoint selection uses the dev-internal val
slice only.

```bash
python scripts/train_detector.py \
    --data outputs/det_out_v2/det_dev.npz \
    --views both --epochs 80 --batch 8 --base-ch 32 --lr 0.001 \
    --bootstrap 1000 --out outputs/detector_dev_scratch_c32_both

python scripts/evaluate_detector.py \
    --dev-run outputs/detector_dev_scratch_c32_both \
    --data outputs/det_out_v2/det_dev.npz \
    --out outputs/detector_dev_scratch_c32_both_scored

python scripts/calibrate_fusion.py \
    --dev-run outputs/detector_dev_scratch_c32_both_scored \
    --data outputs/det_out_v2/det_dev.npz

python scripts/evaluate_detector.py \
    --dev-run outputs/detector_dev_scratch_c32_both_scored \
    --data outputs/det_out_v2/det_dev.npz \
    --lat-gate 0.07 \
    --out outputs/detector_dev_scratch_c32_both_gated_scored

python scripts/freeze_detector.py \
    --dev-run outputs/detector_dev_scratch_c32_both_gated_scored \
    --primary fusion \
    --overlay-qa-passed \
    --overlay-qa-cases "RibFrac19,RibFrac81,RibFrac257,RibFrac266" \
    --out outputs/detector_dev_scratch_c32_both_gated
```

The gated run (`detector_dev_scratch_c32_both_gated`) is the **frozen head** for correspondence evaluation.

## 6. Addressing model (demo + rib-level output)

Build detector-frame crops and train the addressing network:

```bash
python scripts/make_address_data_detframe.py \
    --dev outputs/det_out_v2/det_dev.npz \
    --image-dirs data/ribfrac_train data/ribfrac \
    --seg-dir data/ribseg/ribseg_v2/seg \
    --cl-dir data/ribseg/ribseg_v2/cl \
    --crop 96 --half 24 \
    --out outputs/det_out_v2/address_dataset_detframe.npz

python scripts/train_address.py \
    --data outputs/det_out_v2/address_dataset_detframe.npz \
    --folds 5 --epochs 40 --no-pos

python scripts/train_address_deploy.py \
    --data outputs/det_out_v2/address_dataset_detframe.npz \
    --views ap --no-pos --epochs 60 \
    --out outputs/addressing_model_ap_nopos
```

## 7. Rib atlas (3D demo context)

Uses the published `geometry_split.json` for atlas build / val holdout:

```bash
python scripts/build_rib_atlas.py \
    --cl-dir data/ribseg/ribseg_v2/cl \
    --image-dirs data/ribfrac_train data/ribfrac \
    --det-manifest outputs/det_out_v2/det_manifest.json \
    --build-cases geometry_split.json \
    --out outputs/rib_atlas_v1 --k 60
```

Per-case reconstruction for the demo uses `scripts/reconstruct_3d.py` (called from `run_ribassist.py` /
`demo_app/`).

## 8. Correspondence diagnostics (L0 → L1)

Optional but documented context for why lateral extraction was recalibrated:

```bash
python scripts/eval_lateral_L0_diagnosis.py \
    --detector-run outputs/detector_dev_scratch_c32_both_gated \
    --data outputs/det_out_v2/det_dev.npz \
    --out outputs/lateral_L0_diagnosis.json

python scripts/eval_lateral_L01_calibration.py \
    --detector-run outputs/detector_dev_scratch_c32_both_gated \
    --data outputs/det_out_v2/det_dev.npz \
    --out outputs/lateral_L01_calibration.json
```

L1 extraction-policy sweep (identifies `latN3_f060` as the best frozen-head policy):

```bash
bash scripts/run_L1_correspondence_sweep.sh
```

Outputs land in `outputs/L1_sweep/` (remove that directory to re-run cleanly).

## 9. Lateral retrain (L2) + floor sweep

```bash
python scripts/train_lateral_L2.py \
    --champion-run outputs/detector_dev_scratch_c32_both_gated \
    --data outputs/det_out_v2/det_dev.npz \
    --epochs 60 --pos-weight 3.0 --remine-every 10 \
    --hnm-weight 3.0 --hnm-sigma 4.0 --init champion \
    --out-run outputs/detector_L2_lateral_hnm

bash scripts/run_L2_floor_sweep.sh
```

The winning L2 operating point is `latN3_f010` (lateral floor 0.10). Minor run-to-run variation on MPS is
possible; compare checkpoint sha256 against `split_manifest.json`.

Optional L2 eval at the standing policy:

```bash
bash scripts/run_L2_eval.sh
```

## 10. Sealed confirmatory evaluation

**Prerequisites.** Stage 1 reads the two development candidate graphs produced earlier:
`outputs/L1_sweep/latN3_f060_D0_pairs.npz` (from the L1 sweep, Section 8) and
`outputs/L2_sweep/L2_latN3_f010_D0_pairs.npz` (from the L2 floor sweep, Section 9). Run Sections 5, 8, and 9
first, or Stage 1 will abort on the missing files.

**Stage 1** assembles the sealed cohort and freezes D1 deployment configs on development (no sealed metrics):

```bash
bash scripts/run_sealed_stage1_freeze.sh
```

Review the printed policies and `det_test.npz` sha256, then **Stage 2** (the one confirmatory pass):

```bash
bash scripts/run_sealed_stage2_apply.sh
```

Expected headline: frozen `recall@10 0.0%` (0/601); **L2 `recall@10 2.50%` (15/601) at 0.436 false-3D/case,
case-bootstrap 95% CI [0.69%, 4.48%]**. Results: `outputs/sealed/{frozen,L2}_sealed_D1.json`.

Alternatively assemble manually:

```bash
python scripts/build_sealed_det_npz.py \
    --inputs outputs/det_out_v2/det_test_inputs.npz \
    --gt outputs/det_out_v2/det_test_gt.npz \
    --manifest outputs/det_out_v2/det_manifest.json \
    --out outputs/det_out_v2/det_test.npz
```

## 11. Figures and demo

```bash
python scripts/make_main_results_figure.py \
    --sealed-dir outputs/sealed --out figures/main_results.png

python scripts/make_demo_figures.py \
    --detector-run outputs/detector_L2_lateral_hnm \
    --data outputs/det_out_v2/det_test.npz \
    --pairs-npz outputs/sealed/L2_sealed_D0_pairs.npz \
    --policy outputs/sealed/L2_policy.json \
    --sealed-d1-json outputs/sealed/L2_sealed_D1.json \
    --image-dirs data/ribfrac_train data/ribfrac \
    --seg-dir data/ribseg/ribseg_v2/seg \
    --cl-dir data/ribseg/ribseg_v2/cl \
    --expected-data-sha256 75c62ab5286dedbfd6e6d994f8b1ede6e969a9e505aaded43379f672fdb0514d \
    --out-dir outputs/demo --overwrite

pip install -r requirements-demo.txt
streamlit run app.py
```

Smoke tests (no UI):

```bash
bash scripts/smoke_demo.sh
```

## 12. Optional: learned pair-scorer (2×2 attribution)

Run only on the winning L1 policy to reproduce the attribution figure:

```bash
bash scripts/run_L1_D2_confirm.sh
```

This trains/evaluates `eval_correspondence_D2a_crops.py` + `eval_correspondence_D2b_learned.py`.

## 13. Provenance model

`sha256(det_dev.npz)` is embedded in every downstream artifact. The sealed path adds an external anchor:
assembled `det_test.npz` is verified against manifest-recorded source hashes before any sealed metric is
produced. See `scripts/make_integration_manifest.py` for a full integration manifest after freezing.

## 14. Legacy FROC sealed path

An earlier detector-only sealed evaluator exists (`scripts/eval_sealed_test.py`) for the
`train_detector → evaluate_detector → freeze_detector` FROC pipeline. The **reported headline result** uses
the correspondence path (Sections 8-10) instead. Keep `eval_sealed_test.py` for completeness if you need the
original FROC confirmatory pass.
