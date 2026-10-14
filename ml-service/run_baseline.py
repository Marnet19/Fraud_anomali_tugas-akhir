import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split

print("⏳ Memuat dataset 'data_uji_20_percent.csv'...")
df = pd.read_csv("data_uji_20_percent.csv")

# 1. Penentuan target kolom label (is_fraud / isFraud)
target_col = 'isFraud' if 'isFraud' in df.columns else ('is_fraud' if 'is_fraud' in df.columns else None)

if not target_col:
    raise ValueError("❌ Kolom label 'is_fraud' atau 'isFraud' tidak ditemukan pada dataset!")

# 2. Filter fitur numerik (Eksklusi ID, Waktu, & Label agar setara dengan Isolation Forest)
cols_to_exclude = [
    'isFraud', 'is_fraud', 
    'TransactionID', 'transaction_id', 
    'TransactionDT', 'transaction_dt', 
    'data_split'
]
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in cols_to_exclude]

X = df[numeric_cols].fillna(0)
y = df[target_col].values

# 3. Split 80% Train dan 20% Test (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

def print_metrics(model_name, y_true, y_pred, y_scores=None):
    print(f"\n================ HASIL {model_name.upper()} ================")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    if y_scores is not None:
        print(f"PR-AUC    : {average_precision_score(y_true, y_scores):.4f}")
    else:
        print("PR-AUC    : N/A")

# --- A. BASELINE 1: Local Outlier Factor (LOF) ---
print("\n⏳ Menjalankan Local Outlier Factor (LOF)...")
lof = LocalOutlierFactor(novelty=True, contamination=0.035, n_jobs=-1)
lof.fit(X_train)
preds_lof = lof.predict(X_test)
y_pred_lof = [1 if p == -1 else 0 for p in preds_lof]
scores_lof = -lof.decision_function(X_test)
print_metrics("Local Outlier Factor (LOF)", y_test, y_pred_lof, scores_lof)

# --- B. BASELINE 2: One-Class SVM ---
print("\n⏳ Menjalankan One-Class SVM (OCSVM)...")
ocsvm = OneClassSVM(nu=0.035)
ocsvm.fit(X_train)
preds_svm = ocsvm.predict(X_test)
y_pred_svm = [1 if p == -1 else 0 for p in preds_svm]
scores_svm = -ocsvm.decision_function(X_test)
print_metrics("One-Class SVM", y_test, y_pred_svm, scores_svm)

# --- C. BASELINE 3 SUPERVISED: XGBoost ---
print("\n⏳ Menjalankan XGBoost (Supervised)...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
scores_xgb = xgb.predict_proba(X_test)[:, 1]
print_metrics("XGBoost Classifier", y_test, y_pred_xgb, scores_xgb)