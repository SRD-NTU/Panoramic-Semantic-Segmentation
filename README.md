# AdapToPASS - NeurIPS

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

