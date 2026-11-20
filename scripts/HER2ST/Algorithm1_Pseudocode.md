# Algorithm 1 — Compact Leiden with Similarity-Weighted LOSO Hyperparameter Transfer

**Input:** HER2ST sections S = {A1, B1, D1, E1, F1, G2, H1}; grid Θ = HVG × PCA × scale; k = 20.

**Output:** Selected hyperparameters and held-out ARI for each target section.

1. **Build the full search matrix**
   1. For each section s in S:
      1. Load the raw count matrix, spatial coordinates, and original pathologist reference labels.
      2. Set K_s to the number of unique non-missing reference classes.
      3. Define the section descriptor z_s = [number of genes, number of spots / K_s].
      4. For each hyperparameter triple θ in Θ:
         1. Make gene names unique.
         2. Select highly variable genes using Scanpy `seurat_v3`.
         3. Normalize each spot to total count 10,000 and apply log(1+x).
         4. Standardize genes; optionally clip standardized values by the selected scale bound.
         5. Compute PCA using the selected number of components.
         6. Build a k-nearest-neighbor graph in PCA space with cosine distance.
         7. Search Leiden resolution in [0, 3] by binary search for at most 15 iterations, targeting K_s clusters.
         8. In physical coordinate space, find k Euclidean neighbors for every spot.
         9. Apply one synchronous majority-vote spatial refinement using the initial Leiden labels.
         10. Compute ARI only on spots with valid reference labels.
         11. Store ARI(s, θ).

2. **Similarity-weighted LOSO selection**
   1. For each target section T in S:
      1. Let R = S \ {T}.
      2. Compute feature-wise mean and standard deviation of z_s using only s in R.
      3. Standardize both training descriptors and z_T using the training statistics.
      4. For every training section s in R, compute Euclidean distance d_s,T.
      5. Compute inverse-distance weight w_s,T = 1 / (d_s,T + 1e-8) and normalize weights to sum to 1.
      6. For every θ in Θ, compute weighted training score Score(θ,T) = Σ_s∈R w_s,T · ARI(s,θ).
      7. Select θ*_T = argmax_θ Score(θ,T).
      8. Re-run the complete compact Leiden pipeline on held-out section T using θ*_T.
      9. Report the held-out ARI for T.

3. **Equal-weight control**
   1. Repeat the LOSO selection score with all six training sections assigned equal weight 1/6.
   2. Compare section-wise and mean ARI against the similarity-weighted strategy.

**Leakage control:** ARI of the held-out target section is never used when selecting its hyperparameters. Descriptor standardization is also fit on the six training sections only.
