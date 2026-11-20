# simple-ST: HER2ST Compact Leiden + Similarity-Weighted LOSO

This repository contains the code and data artifacts used for the HER2ST experiments in the manuscript **“Similarity-Weighted Hyperparameter Transfer in Spatial Transcriptomics: A Compact Leiden Clustering Approach for HER2ST.”**

## Dataset used in the manuscript

The experiments use the public HER2ST dataset and the seven sections **A1, B1, D1, E1, F1, G2, H1**. For each section the repository contains:

- raw spot-by-gene count data under `dataset_klasoru/ST-cnts/`;
- spatial spot metadata under `dataset_klasoru/ST-spotfiles/`;
- reference labels under `dataset_klasoru/ST-pat/lbl/`.

The label files are the original HER2ST pathologist-provided annotations. The upstream dataset README states that the labels were extracted from annotated H&E images. No new manual, crowdsourced, or model-generated relabeling was performed in this study. Missing/undetermined labels are excluded only from ARI calculation; the corresponding spots remain in preprocessing and clustering.

## Main code

- `compact_leiden.py`: data loading, preprocessing, cosine-neighborhood graph, Leiden resolution search, and one-pass spatial majority refinement.
- `final_weighted_loso.py`: 7 x 100 full grid, training-only standardization of section descriptors, inverse-distance (1/d) similarity weighting, LOSO hyperparameter selection, and fresh held-out evaluation.

The manuscript workflow uses:

- HVG: 1000, 2000, 3000, 4000, 5000
- PCA: 15, 25, 40, 60, 100
- scale clipping: 5, 10, 15, None
- k = 20 neighbors
- known-K target derived from the number of non-missing reference classes in each held-out section

## Software environment

Pinned package versions are listed in `requirements.txt`:

- Scanpy 1.10.3
- AnnData 0.10.9
- NumPy 1.26.4
- SciPy 1.13.1
- pandas 2.3.3
- scikit-learn 1.6.1
- igraph 1.0.0
- leidenalg 0.11.0
- scikit-misc 0.3.1
- Matplotlib 3.9.4

## Reproduction

From the repository root:

```bash
pip install -r requirements.txt
python final_weighted_loso.py
```

Outputs are written to `final_weighted_loso/`, including the 700-run grid, selected parameters, section weights, final ARIs, figures, and timing logs.

## Manuscript results

The final section-wise ARIs in `final_weighted_loso/final_results.csv` reproduce the manuscript values:

| Section | ARI |
|---|---:|
| A1 | 0.6927 |
| B1 | 0.4211 |
| D1 | 0.3986 |
| E1 | 0.0884 |
| F1 | 0.2136 |
| G2 | 0.3104 |
| H1 | 0.4190 |
| **Mean** | **0.3634** |

Timing is wall-clock and run-dependent. The manuscript reports a single timed run on the hardware described in the paper; those values are archived in `final_weighted_loso/manuscript_timing.csv`. The pre-existing `timing.csv` is retained as a separate rerun artifact.

## Pseudocode

Reviewer-facing pseudocode matching the implementation is provided at:

`scripts/HER2ST/Algorithm1_Pseudocode.md`

The compatibility directory `scripts/HER2ST/` is retained so the manuscript/revision link resolves to a concise entry point for the HER2ST workflow.
