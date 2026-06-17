import os
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
import warnings
import time

# compact_leiden.py'den temel fonksiyonları alıyoruz
from compact_leiden import load_her2st_data, run_leiden_clustering

warnings.filterwarnings("ignore")

def deep_grid_search(dataset_dir, tissue_map):
    # --- PARAMETRE UZAYI (Genişletilmiş) ---
    hvg_list = [1000, 2000, 3000, 4000, 5000]
    pca_list = [15, 25, 40, 60, 100]
    scale_list = [5, 10, 15, None]
    
    all_logs = []
    best_results = []
    
    output_log_file = "full_detailed_grid_search.csv"
    
    print(f"🔥 DEEP GRID SEARCH BAŞLATILIYOR...")
    print(f"📊 Toplam Kombinasyon: {len(hvg_list) * len(pca_list) * len(scale_list)} (per slice)")
    print("-" * 60)

    for slice_name, pseudo_cluster in tissue_map.items():
        start_time = time.time()
        print(f"\n📂 İşleniyor: {slice_name} (Target: {pseudo_cluster})")
        
        adata_raw = load_her2st_data(dataset_dir, slice_name)
        adata_raw.var_names_make_unique()
        
        slice_best_ari = -1
        
        for hvg in hvg_list:
            for pca_n in pca_list:
                for scale_val in scale_list:
                    try:
                        adata = adata_raw.copy()
                        
                        # 1. HVG
                        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg)
                        adata = adata[:, adata.var['highly_variable']].copy()
                        
                        # 2. Normalize & Log
                        sc.pp.normalize_total(adata, target_sum=1e4)
                        sc.pp.log1p(adata)
                        
                        # 3. Scale
                        sc.pp.scale(adata, max_value=scale_val)
                        
                        # 4. PCA
                        sc.pp.pca(adata, n_comps=pca_n)
                        
                        # 5. Leiden (k=20, Metric=Cosine sabit)
                        col = run_leiden_clustering(adata, neighbor_k=20, target_clusters=pseudo_cluster)
                        
                        # ARI Skorlama
                        mask = ~adata.obs['label'].isna()
                        ari = adjusted_rand_score(adata.obs['label'][mask], adata.obs[col][mask])
                        
                        # Log Kaydı
                        log_entry = {
                            "Slice": slice_name, "HVG": hvg, "PCA": pca_n, 
                            "Scale": "None" if scale_val is None else scale_val, "ARI": ari
                        }
                        all_logs.append(log_entry)
                        
                        # Anlık Ekrana Yaz (Sadece yüksek skorları veya süreci takip için)
                        if ari > slice_best_ari:
                            slice_best_ari = ari
                            print(f"  ⭐ Yeni En İyi! -> HVG:{hvg}, PCA:{pca_n}, Scale:{scale_val} | ARI: {ari:.4f}")
                            
                    except Exception as e:
                        print(f"  [!] Hata ({slice_name} - HVG:{hvg}): {e}")

        elapsed = (time.time() - start_time) / 60
        print(f"✅ {slice_name} tamamlandı. Süre: {elapsed:.2f} dk")

    # --- RAPORLAMA ---
    full_df = pd.DataFrame(all_logs)
    full_df.to_csv(output_log_file, index=False)
    
    # Her slice için en iyileri ayıkla
    summary_best = full_df.loc[full_df.groupby("Slice")["ARI"].idxmax()]
    
    print("\n" + "="*70)
    print("🏆 FINAL: TÜM DOKULAR İÇİN EN OPTİMAL PARAMETRELER")
    print("="*70)
    print(summary_best.to_string(index=False))
    print(f"\n🚀 GENEL ORTALAMA ARI: {summary_best['ARI'].mean():.4f}")
    print(f"📂 Tüm denemeler '{output_log_file}' dosyasına kaydedildi.")

if __name__ == "__main__":
    DATASET_PATH = "C:/Users/MLD/Desktop/BioInformatic/SpaCon/SpaConTDS/dataset/ST/her2st-master/data"
    HER2ST_MAP = {"A1": 5, "B1": 4, "D1": 3, "E1": 3, "F1": 3, "G2": 6, "H1": 6}
    
    deep_grid_search(DATASET_PATH, HER2ST_MAP)