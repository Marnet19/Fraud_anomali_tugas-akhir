# ==========================================
# File: train.py
# Tugas: Memuat data latih, melatih Isolation Forest, simpan model.pkl
# ==========================================
import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Load Data Latih (X_train)
print("Memuat data latih...")
X_train = pd.read_csv("data/X_train.csv")  # Fitur numerik hasil pre-processing

# 2. Inisialisasi dan Latih Model Isolation Forest
print("Melatih model Isolation Forest...")
model = IsolationForest(
    n_estimators=100,
    max_samples="auto",
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train)

# 3. Simpan Artefak Model
os.makedirs("models", exist_ok=True)
model_path = "models/isolation_forest.pkl"
joblib.dump(model, model_path)

print(f" Selesai! Artefak model berhasil disimpan di: {model_path}")