# ==============================================================================
# File: preprocess.py (Optimized Memory Usage for Low RAM)
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def find_file(filename):
    paths = [
        os.path.join(DATA_DIR, filename),
        os.path.join(BASE_DIR, filename)
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return None

tx_path = find_file("train_transaction.csv")
id_path = find_file("train_identity.csv")

if not tx_path:
    print("\n[ERROR] File 'train_transaction.csv' tidak ditemukan!\n")
    sys.exit(1)

print(f"--> Memuat 'train_transaction.csv'...")
df_tx = pd.read_csv(tx_path)

if id_path:
    print(f"--> Memuat 'train_identity.csv'...")
    df_id = pd.read_csv(id_path)
    print("--> Menggabungkan dataset...")
    df = pd.merge(df_tx, df_id, on="TransactionID", how="left")
    del df_tx, df_id  # HAPUS VARIABEL LAMA DARI RAM
else:
    df = df_tx

print(f"    Ukuran total dataset: {df.shape[0]} baris, {df.shape[1]} kolom")

# Simpan train_raw.csv untuk evaluasi.py
train_raw_path = os.path.join(DATA_DIR, "train_raw.csv")
print(f"--> Menyimpan salinan data mentah ke '{train_raw_path}'...")
df.to_csv(train_raw_path, index=False)

print("--> Pra-pemrosesan & hemat memori...")
y = df['isFraud'].astype(np.int8)  # Hemat RAM untuk label

drop_cols = ['TransactionID', 'TransactionDT', 'isFraud']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])

# KUNCI UTAMA: Ambil numerik lalu ubah float64 -> float32 untuk menghemat RAM 50%
X = X.select_dtypes(include=['number']).fillna(0).astype(np.float32)

del df  # HAPUS DATASET MENTAH DARI RAM SEBELUM SPLIT

print("--> Membagi dataset menjadi Train (80%) dan Test (20%)...")
X_train, X_test, _, y_test = train_test_split(
    X, 
    y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

del X, y  # HAPUS VARIABEL UTAMA SETELAH SPLIT

print("--> Menyimpan file ke folder 'data/'...")
X_train.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(DATA_DIR, "X_test.csv"), index=False)
y_test.to_frame(name='isFraud').to_csv(os.path.join(DATA_DIR, "y_test.csv"), index=False)

print("\n Preprocessing Sukses Tanpa MemoryError!\n")