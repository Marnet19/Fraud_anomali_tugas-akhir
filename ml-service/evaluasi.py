import os
import sys
import json
import datetime
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    classification_report, confusion_matrix, average_precision_score,
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "rules_config.json")
REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.json")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.csv")
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.csv")   # opsional, hanya untuk pemeriksaan
RAW_PATH = os.path.join(DATA_DIR, "train_raw.csv")      # HARUS hanya data latih

TAU = 0.50

for p, hint in ((MODEL_PATH, "python train.py"), (X_TEST_PATH, "python preprocess.py"),
                (Y_TEST_PATH, "python preprocess.py")):
    if not os.path.exists(p):
        sys.exit(f"[ERROR] Tidak ditemukan: {p}\nJalankan '{hint}' terlebih dahulu.")

model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH)["isFraud"].astype(int)

assert len(X_test) == len(y_test), "X_test dan y_test tidak sejajar"
if hasattr(model, "feature_names_in_"):
    assert list(model.feature_names_in_) == list(X_test.columns), "Kolom X_test != fitur model"

print(f"--> Model: contamination={model.contamination!r}, offset_={model.offset_:.4f}, "
      f"fitur={model.n_features_in_}")
if abs(model.offset_ + 0.5) > 1e-9:
    print("[PERINGATAN] offset_ != -0.5: ambang s >= 0.50 tidak setara dengan model.predict().")

# Skor anomali (Persamaan 3) dan prediksi pada tau tetap
scores = -model.score_samples(X_test)
y_pred = (scores >= TAU).astype(int)

pr_auc = average_precision_score(y_test, scores)
roc_auc = roc_auc_score(y_test, scores)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

print("\n=======================================================")
print("           HASIL EVALUASI MODEL ISOLATION FOREST")
print("=======================================================")
print(f" Data uji      : {len(y_test):,} baris | fraud: {int(y_test.sum()):,} ({100*y_test.mean():.2f}%)")
print(f" PR-AUC        : {pr_auc:.4f}   (acuan acak = prevalensi = {y_test.mean():.4f})")
print(f" ROC-AUC       : {roc_auc:.4f}")
print(f" Accuracy      : {acc:.4f}")
print(f" Precision     : {prec:.4f}")
print(f" Recall        : {rec:.4f}")
print(f" F1-Score      : {f1:.4f}")
print(f" Ditandai      : {int(y_pred.sum()):,} ({100*y_pred.mean():.2f}%)")
print(f" TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}\n")
print(classification_report(y_test, y_pred, digits=4))

# Statistik aturan NLG (dari data LATIH saja)
print("=======================================================")
print("        STATISTIK ATURAN NARRATIVE ENGINE")
print("=======================================================")
p99_val, risky_products = None, None
if not os.path.exists(RAW_PATH):
    sys.exit(f"[ERROR] {RAW_PATH} tidak ditemukan. Nilai aturan tidak boleh diisi dengan angka bawaan.")

# train_raw.csv berisi SELURUH data (latih + uji). Baris uji dibuang lebih dulu agar
# P99 dan tingkat fraud per ProductCD tidak memakai label data uji.
# Pencocokan memakai "sidik jari" SEMUA kolom numerik yang sama antara X_test dan data mentah.
hdr = pd.read_csv(RAW_PATH, nrows=0).columns
probe = pd.read_csv(RAW_PATH, nrows=2000, low_memory=False)
key_cols = [c for c in X_test.columns if c in hdr and pd.api.types.is_numeric_dtype(probe[c])]
if len(key_cols) < 20:
    sys.exit("[ERROR] Kolom numerik bersama X_test dan train_raw terlalu sedikit untuk memisahkan baris uji.")
print(f" Memuat data mentah ({len(key_cols)} kolom kunci); ini bisa memakan beberapa menit...")
dtypes = {c: "float32" for c in key_cols if c != "TransactionAmt"}
raw = pd.read_csv(RAW_PATH, usecols=sorted(set(key_cols + ["TransactionAmt", "ProductCD", "isFraud"])),
                  dtype=dtypes, low_memory=False)


def fingerprint(df):
    vals = df[key_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    return pd.util.hash_pandas_object(vals, index=False)


raw_k = fingerprint(raw)
test_k = fingerprint(X_test)

# (a) setiap baris uji harus ditemukan di data mentah, kalau tidak sebagian baris uji tidak bisa dibuang
found = float(test_k.isin(raw_k).mean())
print(f" Baris uji yang ditemukan di data mentah: {100*found:.3f}%")
if found < 0.999:
    sys.exit("[ERROR] Sebagian baris uji tidak ditemukan di data mentah (praproses mengubah nilai kolom?). "
             "Simpan TransactionID data latih/uji saat preprocess, atau kirim preprocess.py.")

# (b) verifikasi label pada pencocokan unik: isFraud mentah harus sama dengan y_test
raw_counts = raw_k.map(raw_k.value_counts())
dup_test = test_k.duplicated(keep=False).to_numpy()
lab_by_key = pd.Series(y_test.to_numpy()[~dup_test], index=test_k.to_numpy()[~dup_test])
m = raw[(raw_counts == 1) & raw_k.isin(lab_by_key.index)]
m_lab = lab_by_key.reindex(raw_k[m.index].to_numpy()).to_numpy()
agree = float((m["isFraud"].to_numpy() == m_lab).mean()) if len(m) else 0.0
print(f" Pencocokan unik: {len(m):,} dari {len(X_test):,} baris uji "
      f"({100*len(m)/len(X_test):.1f}%); label cocok {100*agree:.3f}%")
if len(m) == 0 or agree < 1.0:
    sys.exit("[ERROR] Verifikasi label gagal; pencocokan tidak dapat dipercaya.")

# (c) buang SEMUA baris mentah yang kuncinya sama dengan baris uji mana pun (konservatif)
train_only = raw[~raw_k.isin(set(test_k))]
n_removed = len(raw) - len(train_only)
print(f" Data mentah: {len(raw):,} baris; dibuang: {n_removed:,}; sisa untuk statistik aturan: {len(train_only):,}")
if os.path.exists(X_TRAIN_PATH):
    n_train = len(pd.read_csv(X_TRAIN_PATH, usecols=[0]))
    print(f" Baris X_train: {n_train:,}; sisa/X_train = {100*len(train_only)/n_train:.2f}% "
          "(selisih = baris latih yang identik dengan baris uji, ikut terbuang)")
    if not (0.90 * n_train <= len(train_only) <= n_train):
        sys.exit("[ERROR] Jumlah baris sisa tidak sesuai X_train; periksa data.")

p99_val = float(train_only["TransactionAmt"].quantile(0.99))
rate = train_only.groupby("ProductCD")["isFraud"].mean() * 100
risky_products = rate[rate > 5.0].index.tolist()
print(f" P99 TransactionAmt (tanpa data uji): USD {p99_val:.2f}")
print(f" Tingkat fraud per ProductCD (%):\n{rate.round(2).to_string()}")
print(f" ProductCD berisiko (>5%): {risky_products}")

metrics = {
    "pr_auc": round(float(pr_auc), 4), "roc_auc": round(float(roc_auc), 4),
    "precision": round(float(prec), 4), "recall": round(float(rec), 4),
    "f1_score": round(float(f1), 4), "accuracy": round(float(acc), 4),
}

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump({"tau_threshold": TAU, "p99_threshold": round(p99_val, 2),
               "risky_products": risky_products, "metrics": metrics}, f, indent=4)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "test_data_count": int(len(y_test)),
        "threshold_tau": TAU,
        "model_offset": float(model.offset_),
        "environment": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__,
                        "numpy": np.__version__},
        "metrics": metrics,
        "matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }, f, indent=2)

print(f"\n Ditulis: {REPORT_PATH}\n Ditulis: {CONFIG_PATH}\n")