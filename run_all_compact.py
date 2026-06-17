import json
import os
import subprocess
import sys

def run_compact_leiden(json_path):
    print(f"\n[*] İşleniyor: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # DİKKAT: Artık sadece compact_leiden.py dosyasını çağırıyoruz
    main_script_path = "compact_leiden.py" 
    
    cmd_args = [
        sys.executable, main_script_path,
        "--dataset_dir", config_dict["dataset_dir"],
        "--slice_name", config_dict["slice_name"],
        "--pseudo_cluster", str(config_dict.get("pseudo_cluster", 7)),
        "--k", "20",  # Sabit k=20
        "--save_dir", "./results_leiden_only"
    ]

    try:
        # Alt scripti çalıştır
        subprocess.run(cmd_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[-] Hata oluştu ({json_path}): {e}")

if __name__ == "__main__":
    # Test edilecek dokuların listesi
    json_files = [
        "./scripts/HER2ST/A1.json",
        "./scripts/HER2ST/B1.json",
        "./scripts/HER2ST/D1.json",
        "./scripts/HER2ST/E1.json",
        "./scripts/HER2ST/F1.json",
        "./scripts/HER2ST/G2.json",
        "./scripts/HER2ST/H1.json"
    ]

    print("🚀 SADECE LEIDEN (k=20) TESTİ BAŞLATILIYOR...")

    for j_file in json_files:
        if os.path.exists(j_file):
            run_compact_leiden(j_file)
        else:
            print(f"[-] Dosya bulunamadı: {j_file}")
            
    print("\n✅ İşlem tamamlandı. Sadece Leiden sonuçları üretildi.")