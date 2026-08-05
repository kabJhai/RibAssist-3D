# Model checkpoint license

Model checkpoints for RibAssist 3D are **not** automatically licensed under this repository's
Apache-2.0 code license. They are trained on third-party medical datasets and must comply
with **all** dataset licenses that contributed to training.

## RibFrac (CC BY-NC 4.0): binding for checkpoints

The [MICCAI 2020 RibFrac challenge](https://ribfrac.grand-challenge.org/) dataset (Zenodo:
training, validation, and test subsets) is licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

RibAssist 3D detectors, addressing models, and projection tensors are trained on RibFrac-derived
data. If you obtain or redistribute checkpoints trained on RibFrac, you must:

- **Attribute** the RibFrac authors, link to CC BY-NC 4.0, and note any changes to derived
  artifacts. Do not imply endorsement.
- **Restrict use to non-commercial purposes.** Commercial use of weights trained on RibFrac
  (or of outputs produced with them) is not permitted under the dataset license unless you
  have a separate rights grant from the licensors.
- **Cite** the RibFrac challenge and the Zenodo records you downloaded, as requested by the
  dataset maintainers.

This is the **primary constraint** on sharing trained weights from this project.

## RibSeg v2 (Apache-2.0)

[RibSeg v2](https://github.com/M3DV/RibSeg) (rib segmentations and centerlines) is licensed
under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

RibSeg is used at inference and evaluation for 3D anatomy (not as the detector training set).
You may use and redistribute RibSeg data under Apache-2.0 (include the license, note changes,
retain notices). RibSeg's Apache license does **not** override RibFrac's NC restriction on
models trained with RibFrac labels and projections.

## This repository

- **Source code:** Apache-2.0 (see [LICENSE](LICENSE)).
- **RibFrac / RibSeg data:** not included; obtain separately under their licenses.
- **Checkpoints:** not committed until distribution terms are documented. Any future release
  of weights will be **non-commercial research use only**, compliant with RibFrac CC BY-NC 4.0,
  with RibFrac and RibSeg attribution as applicable.

Do not deploy trained weights in clinical or commercial products without independent legal
review and explicit permission from all relevant rights holders.
