import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix

def run_evaluation(dataset_path):
    print("=== Memulai Evaluasi Komparatif: Isolation Forest vs LOF ===")
    print(f"Memuat data dari: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"Dimensi data awal: {df.shape}")
    
    # 1. Cek keberadaan kolom 'isFraud'
    if 'isFraud' not in df.columns:
        print("\n❌ ERROR: Kolom 'isFraud' tidak ditemukan di dataset ini!")
        print("Untuk menghitung metrik evaluasi jurnal (PR-AUC, Precision, Recall, F1), kolom 'isFraud' wajib ada.")
        print("Silakan ganti DATASET_PATH di baris bawah dengan file yang masih menyimpan kolom 'isFraud' (misal: train_transaction.csv).")
        return

    # 2. Pemisahan Fitur (X) dan Target (y)
    y = df['isFraud']
    X = df.drop(columns=['isFraud']).select_dtypes(include=[np.number]).fillna(0)
    
    # 3. Batasi Sampel Data (Mencegah Memory/RAM Error pada LOF)
    # Mengambil 25.000 sampel agar pembagian menjadi Train: 20.000 dan Test: 5.000
    if len(df) > 25000:
        print(f"\nData cukup besar ({len(df)} baris). Mengambil 25.000 sampel (stratified) agar RAM aman & LOF cepat...")
        X, _, y, _ = train_test_split(X, y, train_size=25000, random_state=42, stratify=y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"-> Jumlah Data Latih (Train) : {len(X_train)} sampel")
    print(f"-> Jumlah Data Uji (Test)    : {len(X_test)} sampel")
    
    # 4. Standardisasi Fitur (float32 untuk efisiensi memori)
    print("\nMelakukan standardisasi fitur...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype('float32'))
    X_test_scaled = scaler.transform(X_test.astype('float32'))
    
    # ==========================================
    # 5. PELATIHAN & EVALUASI ISOLATION FOREST
    # ==========================================
    print("\n[1/2] Melatih Isolation Forest...")
    if_model = IsolationForest(random_state=42, n_jobs=-1)
    
    t0 = time.time()
    if_model.fit(X_train_scaled)
    if_train_time = time.time() - t0
    
    t0 = time.time()
    if_scores = -if_model.decision_function(X_test_scaled)
    if_infer_time = time.time() - t0
    
    # ==========================================
    # 6. PELATIHAN & EVALUASI LOCAL OUTLIER FACTOR (LOF)
    # ==========================================
    print("[2/2] Melatih Local Outlier Factor (LOF)...")
    lof_model = LocalOutlierFactor(novelty=True, n_jobs=-1)
    
    t0 = time.time()
    lof_model.fit(X_train_scaled)
    lof_train_time = time.time() - t0
    
    t0 = time.time()
    lof_scores = -lof_model.score_samples(X_test_scaled)
    lof_infer_time = time.time() - t0
    
    # ==========================================
    # 7. HITUNG METRIK EVALUASI
    # ==========================================
    print("\nMenghitung metrik perbandingan...")
    if_pr_auc = average_precision_score(y_test, if_scores)
    lof_pr_auc = average_precision_score(y_test, lof_scores)
    
    prevalence = (y_train.sum() / len(y_train)) * 100
    if_threshold = np.percentile(if_scores, 100 - prevalence)
    lof_threshold = np.percentile(lof_scores, 100 - prevalence)
    
    if_preds = (if_scores >= if_threshold).astype(int)
    lof_preds = (lof_scores >= lof_threshold).astype(int)
    
    # ==========================================
    # 8. CETAK HASIL PERBANDINGAN
    # ==========================================
    print("\n" + "="*55)
    print(" HASIL EVALUASI: ISOLATION FOREST")
    print("="*55)
    print(f"Waktu Pelatihan : {if_train_time:.4f} detik")
    print(f"Waktu Inferensi : {if_infer_time:.4f} detik")
    print(f"PR-AUC (AP)     : {if_pr_auc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, if_preds))
    print("\nClassification Report:\n", classification_report(y_test, if_preds, digits=4))
    
    print("\n" + "="*55)
    print(" HASIL EVALUASI: LOCAL OUTLIER FACTOR (LOF)")
    print("="*55)
    print(f"Waktu Pelatihan : {lof_train_time:.4f} detik")
    print(f"Waktu Inferensi : {lof_infer_time:.4f} detik")
    print(f"PR-AUC (AP)     : {lof_pr_auc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, lof_preds))
    print("\nClassification Report:\n", classification_report(y_test, lof_preds, digits=4))

if __name__ == '__main__':
    # Pastikan file ini adalah file yang MASIH MEMILIKI kolom 'isFraud'
    DATASET_PATH = 'train_transaction.csv' 
    run_evaluation(DATASET_PATH)