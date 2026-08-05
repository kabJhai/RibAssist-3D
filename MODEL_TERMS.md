# Model checkpoint terms

Model checkpoints for RibAssist 3D are **not** automatically licensed under this repository's
Apache-2.0 code license. They are trained on third-party medical datasets and must comply
with **all** dataset licenses that contributed to training.

This file describes compliance expectations for checkpoints you train locally or any future
checkpoint distribution. It is **not** an offer to license committed model artifacts: this
repository does not currently redistribute trained weights.

## RibFrac (CC BY-NC 4.0): binding for checkpoints

The [MICCAI 2020 RibFrac challenge](https://ribfrac.grand-challenge.org/) dataset (Zenodo:
training, validation, and test subsets) is licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

RibAssist 3D detectors, addressing models, and projection tensors are trained on RibFrac-derived
data. If you obtain or redistribute checkpoints trained on RibFrac, you must:

- **Attribute** the RibFrac authors, link to CC BY-NC 4.0, and note any changes to derived
  artifacts. Do not imply endorsement.
- **Respect non-commercial constraints on the checkpoints.** Commercial use of the checkpoints,
  or incorporation of them into a commercial product or service, is not authorized under these
  terms unless the relevant upstream rights holders grant separate permission. Users are
  responsible for determining whether any intended use or distribution of model outputs complies
  with applicable dataset licenses and law.
- **Cite** the RibFrac challenge and the Zenodo records you downloaded, as requested by the
  dataset maintainers.

This is the **primary constraint** on sharing trained weights from this project.

## RibSeg v2

The [RibSeg v2](https://github.com/M3DV/RibSeg) repository distributes software and related
materials under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). RibSeg
segmentation and centerline annotations used here are derived from RibFrac source CTs; upstream
dataset terms may still apply to those annotations and to artifacts you build from them. Review
the RibSeg repository notices and upstream dataset conditions before redistributing annotations
or derived artifacts.

RibSeg is used at inference and evaluation for 3D anatomy (not as the detector training set).
RibSeg's Apache-2.0 terms do **not** override RibFrac's NC restriction on models trained with
RibFrac labels and projections.

## This repository

- **Source code:** Apache-2.0 (see [LICENSE](LICENSE)).
- **RibFrac / RibSeg data:** not included; obtain separately under their licenses.
- **Checkpoints:** not committed. Any future release of weights will document distribution
  terms compliant with RibFrac CC BY-NC 4.0 and applicable RibSeg notices, with attribution as
  required.

Do not deploy trained weights in clinical or commercial products without independent legal
review and explicit permission from all relevant rights holders.
