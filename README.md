# Spatify
Spatify is a Python-based toolkit for integrating spatial transcriptomics data with nuclear imaging and single-cell RNA-seq analysis.

## data structure

**Resolve files/**
- **Stage/**
  - `DAPI.tiff` — DAPI channel image (nuclei)
  - `DAPI_mask.tiff` — Cellpose nuclei segmentation mask
  - `BF.tiff` — Brightfield image
  - `Apical_layer.tiff` — Apical layer mask
  - `sliceVertices.dat` — Tissue outline vertices data
  - `scored_Results.txt` — Spatial transcriptomics results
