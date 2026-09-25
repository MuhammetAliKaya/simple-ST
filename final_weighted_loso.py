import os
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from compact_leiden import load_her2st_data, run_leiden_clustering

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(PROJECT_DIR, "dataset_klasoru")
DEFAULT_OUT = os.path.join(PROJECT_DIR, "final_weighted_loso")
SLICES = ["A1", "B1", "D1", "E1", "F1", "G2", "H1"]

HVG_LIST = [1000, 2000, 3000, 4000, 5000]
PCA_LIST = [15, 25, 40, 60, 100]
SCALE_LIST = [5, 10, 15, None]
NEIGHBOR_K = 20

PLOT_POINT_SIZE = 90
PLOT_FIGSIZE = (14, 6)
PLOT_DPI = 350

def get_k(a):
    # Known-K: gerçek etiketlerde gözlenen sınıf sayısı.
    # Statik A1/B1/... haritası kullanılmaz.
    return int(a.obs["label"].dropna().nunique())


def scale_key(x):
    return "None" if x is None else str(x)


def scale_value(x):
    return None if str(x) == "None" else float(x)


def preprocess(a, hvg, pca, scale):
    a = a.copy()
    a.var_names_make_unique()
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=int(hvg))
    a = a[:, a.var["highly_variable"]].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.scale(a, max_value=scale_value(scale))
    sc.pp.pca(a, n_comps=int(pca))
    return a


def run_config(raw, hvg, pca, scale, return_result=False):
    target_k = get_k(raw)
    a = preprocess(raw, hvg, pca, scale)
    col = run_leiden_clustering(a, neighbor_k=NEIGHBOR_K, target_clusters=target_k)
    m = ~a.obs["label"].isna()
    ari = adjusted_rand_score(a.obs["label"][m], a.obs[col][m])
    predicted_k = int(a.obs[col].nunique())
    if return_result:
        return float(ari), predicted_k, a, col
    return float(ari), predicted_k


def save_result_png(a, cluster_col, slice_name, ari, path):
    fig, axes = plt.subplots(1, 2, figsize=PLOT_FIGSIZE, constrained_layout=True)
    spatial = a.obsm["spatial"]

    for ax, column, title in (
        (axes[0], "label", "Ground truth"),
        (axes[1], cluster_col, f"Weighted LOSO Leiden (ARI={ari:.3f})"),
    ):
        values = pd.Categorical(a.obs[column])
        mask = values.codes >= 0

        ax.scatter(
            spatial[mask, 0],
            spatial[mask, 1],
            c=values.codes[mask],
            cmap="tab20",
            s=PLOT_POINT_SIZE,
            linewidths=0
        )

        ax.set_title(title)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

    fig.suptitle(f"Slice {slice_name}")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def main(dataset_dir=DEFAULT_DATASET, output_dir=DEFAULT_OUT, slices=None):
    slices = SLICES if slices is None else slices
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    total_start = time.perf_counter()

    # ---------------------------------------------------------
    # 1) FULL GRID: 7 x 100 = 700
    # ---------------------------------------------------------
    grid_start = time.perf_counter()
    grid_rows = []
    desc = {}
    grid_slice_times = []

    print("\n=== 1/3 FULL GRID: 700 RUN ===")

    for s in slices:
        t0 = time.perf_counter()
        raw = load_her2st_data(dataset_dir, s)
        k = get_k(raw)

        # 1/d descriptor: [N_Genes_Total, N_Spots / known-K]
        desc[s] = np.array([raw.n_vars, raw.n_obs / k], dtype=float)

        print(f"{s}: detected K={k} | spots={raw.n_obs} | genes={raw.n_vars}")

        for hvg in HVG_LIST:
            for pca in PCA_LIST:
                for scale in SCALE_LIST:
                    ari, predicted_k = run_config(raw, hvg, pca, scale)
                    grid_rows.append({
                        "Slice": s,
                        "Target_K": k,
                        "Predicted_K": predicted_k,
                        "HVG": hvg,
                        "PCA": pca,
                        "Scale": scale_key(scale),
                        "ARI": ari,
                    })

        sec = time.perf_counter() - t0
        grid_slice_times.append({"Phase": "GRID_" + s, "Seconds": sec})
        print(f"  done: {sec:.2f} sec")

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(os.path.join(output_dir, "grid_700.csv"), index=False)
    grid_sec = time.perf_counter() - grid_start

    # ---------------------------------------------------------
    # 2) 1/d SIMILARITY-WEIGHTED LOSO PARAMETER SELECTION
    # ---------------------------------------------------------
    sel_start = time.perf_counter()
    selected_rows = []
    weight_rows = []

    print("\n=== 2/3 1/d WEIGHTED LOSO SELECTION ===")

    for test in slices:
        train = [s for s in slices if s != test]

        x = np.vstack([desc[s] for s in train])
        t = desc[test]
        mu, sd = x.mean(axis=0), x.std(axis=0)
        sd[sd == 0] = 1.0

        d = np.linalg.norm((x - mu) / sd - (t - mu) / sd, axis=1)
        w = 1.0 / (d + 1e-8)
        w = w / w.sum()
        weights = dict(zip(train, w))

        for s in train:
            weight_rows.append({
                "Test": test,
                "Train": s,
                "Weight": weights[s],
            })

        g = grid[grid["Slice"].isin(train)].copy()
        g["W"] = g["Slice"].map(weights)
        g["WA"] = g["ARI"] * g["W"]

        score = (
            g.groupby(["HVG", "PCA", "Scale"], dropna=False)
             .agg(WA=("WA", "sum"), W=("W", "sum"))
             .reset_index()
        )
        score["Weighted_ARI"] = score["WA"] / score["W"]
        best = score.loc[score["Weighted_ARI"].idxmax()]

        selected_rows.append({
            "Slice": test,
            "HVG": int(best["HVG"]),
            "PCA": int(best["PCA"]),
            "Scale": best["Scale"],
            "Train_Weighted_ARI": float(best["Weighted_ARI"]),
        })

        wtxt = ", ".join(f"{s}={weights[s]:.3f}" for s in train)
        print(
            f"{test}: {wtxt} -> "
            f"HVG={int(best['HVG'])}, PCA={int(best['PCA'])}, "
            f"Scale={best['Scale']}"
        )

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(os.path.join(output_dir, "selected_params.csv"), index=False)
    pd.DataFrame(weight_rows).to_csv(os.path.join(output_dir, "weights.csv"), index=False)
    selection_sec = time.perf_counter() - sel_start

    # ---------------------------------------------------------
    # 3) FRESH HELD-OUT EVALUATION
    # ---------------------------------------------------------
    eval_start = time.perf_counter()
    eval_rows = []

    print("\n=== 3/3 FRESH HELD-OUT EVALUATION ===")

    for _, r in selected.iterrows():
        s = r["Slice"]
        t0 = time.perf_counter()
        raw = load_her2st_data(dataset_dir, s)
        k = get_k(raw)

        ari, predicted_k, result, cluster_col = run_config(
            raw, r["HVG"], r["PCA"], r["Scale"], return_result=True
        )
        save_result_png(
            result, cluster_col, s, ari,
            os.path.join(plot_dir, f"{s}_weighted_loso.png"),
        )
        sec = time.perf_counter() - t0

        eval_rows.append({
            "Slice": s,
            "Target_K": k,
            "Predicted_K": predicted_k,
            "HVG": int(r["HVG"]),
            "PCA": int(r["PCA"]),
            "Scale": r["Scale"],
            "ARI": ari,
            "Eval_Seconds": sec,
        })

        print(
            f"{s}: targetK={k}, predictedK={predicted_k} | "
            f"{int(r['HVG'])}/{int(r['PCA'])}/{r['Scale']} | "
            f"ARI={ari:.4f} | {sec:.2f} sec"
        )
        if predicted_k != k:
            print(f"  WARNING: final refinement changed cluster count {k} -> {predicted_k}")

    results = pd.DataFrame(eval_rows)
    results.to_csv(os.path.join(output_dir, "final_results.csv"), index=False)
    eval_sec = time.perf_counter() - eval_start
    total_sec = time.perf_counter() - total_start

    timing = pd.DataFrame(
        grid_slice_times + [
            {"Phase": "GRID_TOTAL", "Seconds": grid_sec},
            {"Phase": "LOSO_SELECTION", "Seconds": selection_sec},
            {"Phase": "FINAL_EVAL_7_SLICES", "Seconds": eval_sec},
            {"Phase": "END_TO_END_TOTAL", "Seconds": total_sec},
        ]
    )
    timing["Minutes"] = timing["Seconds"] / 60.0
    timing.to_csv(os.path.join(output_dir, "timing.csv"), index=False)

    mean_ari = results["ARI"].mean()
    median_ari = results["ARI"].median()

    report = (
        f"Mean ARI={mean_ari:.4f}\n"
        f"Median ARI={median_ari:.4f}\n"
        f"Grid 700={grid_sec:.2f} sec ({grid_sec/60:.2f} min)\n"
        f"LOSO selection={selection_sec:.4f} sec\n"
        f"Final 7 eval={eval_sec:.2f} sec ({eval_sec/60:.2f} min)\n"
        f"End-to-end={total_sec:.2f} sec ({total_sec/60:.2f} min)\n"
    )

    with open(os.path.join(output_dir, "FINAL_REPORT.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print("\n=== FINAL ===")
    print(report)
    print(f"Saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Similarity-weighted LOSO evaluation")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--slices", nargs="+", default=SLICES)
    args = parser.parse_args()
    main(args.dataset_dir, args.output_dir, args.slices)
