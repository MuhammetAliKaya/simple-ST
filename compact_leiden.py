import os
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import anndata
from sklearn.neighbors import NearestNeighbors
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. VERİ YÜKLEME
# ============================================================
def load_her2st_data(dataset_dir, slice_name):
    cnt_path = os.path.join(dataset_dir, 'ST-cnts', f'{slice_name}.tsv')
    df_cnt = pd.read_csv(cnt_path, sep='\t', index_col=0)
    pos_path = os.path.join(dataset_dir, 'ST-spotfiles', f'{slice_name}_selection.tsv')
    df_pos = pd.read_csv(pos_path, sep='\t')
    lbl_path = os.path.join(dataset_dir, 'ST-pat/lbl', f'{slice_name}_labeled_coordinates.tsv')
    df_lbl = pd.read_csv(lbl_path, sep='\t')

    df_lbl = df_lbl.dropna(axis=0, how='any')
    df_lbl.loc[df_lbl['label'] == 'undetermined', 'label'] = np.nan
    df_lbl['x'] = (df_lbl['x']+0.5).astype(np.int64)
    df_lbl['y'] = (df_lbl['y']+0.5).astype(np.int64)

    df_pos['id'] = [f"{x}x{y}" for x, y in zip(df_pos['x'], df_pos['y'])]
    df_lbl['id'] = [f"{x}x{y}" for x, y in zip(df_lbl['x'], df_lbl['y'])]

    meta_pos = df_cnt.join(df_pos.set_index('id'))
    meta_lbl = df_cnt.join(df_lbl.set_index('id'))

    adata = anndata.AnnData(df_cnt, dtype=np.float32)
    adata.obsm['spatial'] = np.floor(meta_pos[['pixel_x','pixel_y']].values).astype(int)
    adata.obs['label'] = pd.Categorical(meta_lbl['label'])
    return adata

# ============================================================
# 2. LEIDEN KÜMELEME (k=20 & Refinement)
# ============================================================
def run_leiden_clustering(adata, neighbor_k, target_clusters):
    # PCA uzayında komşuluk grafiği oluştururken metriği 'cosine' yapıyoruz
    sc.pp.neighbors(adata, n_neighbors=neighbor_k, use_rep='X_pca', metric='cosine')
    
    # Çözünürlük (Resolution) araması (Aynı kalıyor)
    low_res, high_res = 0.0, 3.0
    for _ in range(15):
        res = (low_res + high_res) / 2
        sc.tl.leiden(adata, resolution=res, key_added='temp')
        n_clust = len(adata.obs['temp'].unique())
        if n_clust < target_clusters: low_res = res
        elif n_clust > target_clusters: high_res = res
        else: break
    
    # Uzamsal düzeltme (Spatial Refinement)
    # DİKKAT: Burada 'euclidean' kalması daha sağlıklıdır çünkü x,y koordinatları 
    # fiziksel mesafedir. Ama istersen burayı da 'cosine' yapabilirsin.
    raw_labels = np.array(adata.obs['temp']).copy()
    refined_labels = raw_labels.copy()
    nbrs = NearestNeighbors(n_neighbors=neighbor_k, metric='euclidean').fit(adata.obsm['spatial'])
    _, indices = nbrs.kneighbors(adata.obsm['spatial'])
    
    for i in range(len(raw_labels)):
        refined_labels[i] = Counter(raw_labels[indices[i]]).most_common(1)[0][0]
    
    cluster_col = "cluster_leiden"
    adata.obs[cluster_col] = pd.Categorical(refined_labels)
    return cluster_col

# ============================================================
# 3. ANA AKIŞ
# ============================================================
def main(args):
    # Veri Yükleme
    adata = load_her2st_data(args.dataset_dir, args.slice_name)
    adata.var_names_make_unique()
    
    # Standart Ön İşleme (Log-Scale & PCA)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=60)

    # Leiden Kümeleme (k=20)
    col = run_leiden_clustering(adata, args.k, args.pseudo_cluster)
    
    # ARI Skoru Hesaplama (Eğer etiket varsa)
    mask = ~adata.obs['label'].isna()
    if mask.sum() > 0:
        ari = adjusted_rand_score(adata.obs['label'][mask], adata.obs[col][mask])
        print(f"Slice: {args.slice_name} | Leiden ARI: {ari:.4f}")
    else:
        ari = 0.0
        print(f"Slice: {args.slice_name} | Etiket bulunamadı.")

    save_path = os.path.join(args.save_dir, f"{args.slice_name}_leiden.png")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. Scanpy spatial plot fonksiyonunu gelişmiş görsel parametrelerle çağırıyoruz
    sc.pl.spatial(
        adata, 
        color=col, 
        title=f"Leiden (ARI: {ari:.3f})", 
        spot_size=150,       # Noktaların dolgun ve bitişik durması için büyütüldü
        alpha=1.0,           # Tam opaklık sağlandı
        edgecolors='none',   # Noktaların etrafındaki ince çizgiler kaldırıldı
        frameon=False,       # Dış siyah kutu çerçevesi kaldırıldı
        show=False
    )
    
    # 3. Yüksek çözünürlüklü (HD 300 DPI) olarak diske kayıt
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--slice_name', type=str, default='A1')
    parser.add_argument('--pseudo_cluster', type=int, default=7)
    parser.add_argument('--k', type=int, default=20) 
    parser.add_argument('--save_dir', type=str, default='./output/')
    main(parser.parse_args())