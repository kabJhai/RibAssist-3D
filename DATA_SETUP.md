# Data setup

RibAssist 3D does **not** ship clinical data or checkpoints. Obtain the datasets below under
their own licenses, then place them under `data/` (see `.gitignore`).

## 1. RibFrac (MICCAI 2020 challenge)

**Challenge:** [Rib Fracture Detection and Classification](https://ribfrac.grand-challenge.org/)  
**License:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (attribution required; non-commercial use only)

The organizers recommend the official train / validation / test split. Annotations are NIfTI
(`.nii.gz`). Due to Zenodo size limits, the training set is split into two parts.

| Subset | CT cases | Annotations | Download |
|---|---:|---|---|
| Training Part 1 | 300 | Yes | https://zenodo.org/records/3893508 |
| Training Part 2 | 120 | Yes | Zenodo records linked from https://ribfrac.grand-challenge.org/dataset/ |
| Validation | 80 | Yes | https://zenodo.org/records/3893496 |
| Test | 160 | Images only | https://zenodo.org/records/3993380 |

Suggested layout after unzip (filenames may vary):

```
data/ribfrac_train/     # training CT + labels (Part 1 and Part 2)
data/ribfrac/             # validation (+ test if needed)
```

Info CSV columns (`ribfrac-train-info-1.csv`, `ribfrac-train-info-2.csv`, `ribfrac-val-info.csv`):

| Column | Meaning |
|---|---|
| `public_id` | Anonymous case ID matching images and annotations |
| `label_id` | Label value in the NIfTI mask |
| `label_code` | Fracture type: 0 background; 1 displaced; 2 non-displaced; 3 buckle; 4 segmental; -1 unknown (ignore for classification) |

Example download (replace `<FILE>` with the exact name from the Zenodo file list):

```bash
wget "https://zenodo.org/records/3893508/files/<FILE>?download=1" -O part1.zip
unzip part1.zip -d data/ribfrac_train/
```

Mirror links for CT images in China mainland are listed on the challenge dataset page; annotations
still come from Zenodo.

**Citation:** Respect the RibFrac authors' effort. Cite the MICCAI 2020 RibFrac challenge and
the Zenodo record(s) you used, link to CC BY-NC 4.0, and note any changes to derived artifacts.

## 2. RibSeg v2 (rib segmentation + centerlines)

**Project:** [M3DV/RibSeg](https://github.com/M3DV/RibSeg)  
**License:** The RibSeg repository distributes software and related materials under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (SPDX: `Apache-2.0`).
Source CT images originate from RibFrac and remain subject to RibFrac's terms. Review the RibSeg
repository notices and upstream dataset conditions before redistributing annotations or derived
artifacts.

Download (Google Drive links from the RibSeg repo):

- Seg + centerlines archive: https://drive.google.com/file/d/1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm
- Filename map spreadsheet: https://docs.google.com/spreadsheets/d/1lz9liWPy8yHybKCdO3BCA9K76QH8a54XduiZS_9fK70

```bash
pip install gdown
gdown --id 1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm -O ribseg_v2.zip
unzip ribseg_v2.zip -d data/ribseg/
```

On gdown 5.0+ the `--id` flag is removed; use the file id directly (`gdown 1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm -O
ribseg_v2.zip`) or `gdown --fuzzy <share-url>`. If Google Drive rate-limits the download, open the share link in a
browser instead.

Expected paths for this codebase:

```
data/ribseg/ribseg_v2/seg/
data/ribseg/ribseg_v2/cl/
```

## 3. Validate local layout

After data and checkpoints are in place:

```bash
python scripts/validate_local_artifacts.py
python scripts/validate_local_artifacts.py --demo --strict
python scripts/validate_local_artifacts.py --all --strict
```

See `REPRODUCE.md` for the sealed evaluation and demo workflow.
