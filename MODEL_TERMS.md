# Model checkpoint terms

Model checkpoints for RibAssist 3D are **not** licensed under this repository's Apache-2.0
code license. They are trained on third-party medical datasets and must comply with **all**
dataset licenses that contributed to training.

This file describes compliance expectations for checkpoints you train locally and for any
**external distribution** (for example, a Hugging Face model repository). It is **not** an
offer to license weights that are not published: the GitHub repository does not commit
checkpoints by default.

## License by artifact

| Artifact | License / terms |
|----------|-----------------|
| RibAssist **source code** | [Apache-2.0](LICENSE) |
| RibAssist **documentation and figures** | [CC BY 4.0](LICENSE-DOCS) |
| RibAssist **checkpoints** (weights trained on RibFrac-derived data) | **CC BY-NC 4.0** / non-commercial research and educational use only |
| **RibFrac** CT volumes and labels | Not redistributed; [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) at source |
| **RibSeg** annotations and anatomy files | Not redistributed; RibSeg **code** is Apache-2.0, but annotations are derived from RibFrac CTs |
| **RibSeg code** (upstream) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |

Do **not** label published checkpoints as Apache-2.0. Source code and weights carry
different licenses.

## RibFrac (CC BY-NC 4.0): binding for checkpoints

The [MICCAI 2020 RibFrac challenge](https://ribfrac.grand-challenge.org/) dataset (Zenodo:
training, validation, and test subsets) is licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

RibAssist 3D detectors, addressing models, and projection tensors are trained on RibFrac-derived
images and labels. Whether neural-network weights legally constitute a copyright "adaptation"
of training data is not fully settled; the **conservative position** is to treat RibFrac
restrictions as carrying into the use and redistribution of those checkpoints. CC BY-NC
explicitly permits **non-commercial** sharing of adaptations, with attribution.

If you obtain, publish, or redistribute RibAssist checkpoints trained on RibFrac, you must:

- **Use them only for non-commercial research and educational purposes**, unless separate
  permission is obtained from the relevant rights holders.
- **Attribute** the RibFrac authors, link to CC BY-NC 4.0, and note any changes to derived
  artifacts. Do not imply endorsement.
- **Cite** the RibFrac challenge and the Zenodo records you used, as requested by the dataset
  maintainers.
- **Not use checkpoints for clinical diagnosis, treatment, or patient care.** RibAssist 3D is
  a research proof of concept, not a medical device.

Commercial use of the checkpoints, or incorporation into a commercial product or service, is
**not** authorized under these terms unless the relevant upstream rights holders grant separate
permission.

## RibSeg v2

The [RibSeg v2](https://github.com/M3DV/RibSeg) repository distributes **software and related
materials** under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The README
provides dataset, code, and models for research use and asks users to cite RibSeg.

**Important:** RibSeg segmentation and centerline data are built from **RibFrac source CT
scans**; the RibSeg README instructs users to obtain those CTs from RibFrac. RibSeg's
Apache-2.0 license on the **repository/code** does not automatically mean every hosted
RibSeg dataset artifact is unambiguously Apache-2.0 for all purposes, or that it overrides
RibFrac's terms on the underlying imaging.

RibSeg is used at inference and evaluation for 3D anatomy (not as the detector training set).
For **combined RibAssist checkpoints** trained with RibFrac labels and projections, the
**cleanest route** is CC BY-NC 4.0 / non-commercial research distribution, with RibFrac
compliance stated explicitly.

## Hugging Face checkpoint release

If checkpoints are published on Hugging Face or elsewhere, the release should:

- Use the model repository **[kabilasoboka/RibAssist-3D](https://huggingface.co/kabilasoboka/RibAssist-3D)** on Hugging Face (or an equivalent named repo under the same account).
- Set the model license to **CC BY-NC 4.0** (or equivalent "other" + link), **not** Apache-2.0.
- Publish a model card on the **Hub** (the Hub repo `README.md`) that matches the terms in this file.
- Prepare uploads from a **local** `huggingface/` staging folder (gitignored on GitHub; not part of this repository). Upload that folder’s contents to the Hub — weights, LICENSE, MODEL_TERMS, CITATION.cff, configs, and examples.
- Keep RibFrac and RibSeg **data out of the Hub repo** unless you have a separate basis to
  redistribute a minimal demo subset.
- State that users must comply with [RibFrac CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
  and obtain RibFrac/RibSeg data separately under their terms.

## This GitHub repository

- **Source code:** Apache-2.0 (see [LICENSE](LICENSE)).
- **RibFrac / RibSeg data:** not included; obtain separately under their licenses.
- **Checkpoints and Hub model card:** not committed here. The local `huggingface/` folder is
  staging-only for Hub upload; the canonical model release lives on Hugging Face.

Do not deploy trained weights in clinical or commercial products without independent legal
review and explicit permission from all relevant rights holders.
