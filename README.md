# AdapToPASS

**Ambiguity-aware Adaptive Spherical Transformer for Panoramic Semantic Segmentation**

This repository contains an anonymous project page for a research submission on robust panoramic semantic segmentation. The project presents **AdapToPASS**, a bio-inspired spherical Transformer designed to improve 360° semantic segmentation under unseen spherical transformations.

## Overview

Panoramic semantic segmentation is challenging because real-world 360° imagery often undergoes camera motion, viewpoint changes, rotations, scale changes, and other spherical transformations. These perturbations introduce contextual, geometric, and boundary ambiguity, which can degrade models that assume a stable canonical spherical layout.

AdapToPASS addresses this by introducing ambiguity-aware adaptive spherical perception directly on the sphere. The method adaptively changes contextual aggregation according to local ambiguity, combines fine-detail and wide-field spherical representations, and uses boundary-aware supervision for sharper semantic segmentation.

## Key Ideas

- **Adaptive Spherical Attention:** dynamically modulates spherical attention using a learned local ambiguity signal.
- **Bifocal Spherical Representation:** combines an acuity stream for high-resolution local detail with a lateral stream for broad contextual understanding.
- **Boundary-aware Supervision:** uses signed distance field guidance to improve semantic boundary localization.
- **Transformation Robustness:** evaluates robustness under unseen spherical transformations including rotation, scale, translation, orientation shift, and viewpoint shift.

## Project Page

The project page is provided as:

```text
index.html
```

To view it locally, open `index.html` in a web browser.

## Anonymity Notes

This repository is prepared for anonymous review. Please ensure that the following are not included before submission:

- Author names
- Affiliations
- Email addresses
- Lab, university, or organization names
- Personal website links
- Non-anonymous GitHub/usernames
- Acknowledgments revealing identity
- PDF metadata containing author information
- File names or asset names that reveal identity

## Suggested Repository Structure

```text
.
├── index.html
├── README.md
├── assets/
│   ├── teaser.png
│   ├── method.png
│   └── results.png
└── paper.pdf
```

The asset names above are generic and anonymous. Replace them with the actual anonymous figures used in the project page.

## Results Highlight

AdapToPASS reports strong segmentation performance on indoor and outdoor panoramic semantic segmentation benchmarks, with improved robustness under unseen spherical transformations and no transformation-specific augmentation during training.

## Citation

The citation is intentionally anonymized for review:

```bibtex
@inproceedings{anonymous2026adaptopass,
  title     = {AdapToPASS: Ambiguity-aware Adaptive Spherical Transformer for Panoramic Semantic Segmentation},
  author    = {Anonymous Author(s)},
  booktitle = {Submitted for anonymous review},
  year      = {2026}
}
```

## License

License information is omitted during anonymous review. Add the appropriate license after the review period if the repository is released publicly.
