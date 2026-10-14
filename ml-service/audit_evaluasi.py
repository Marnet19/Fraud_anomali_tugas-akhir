import os
import sys
import json
import platform
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score as AP,
    roc_auc_score as AUC,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODEL_PATH = os.path.join(BASE, "models", "isolation_forest.pkl")
X_TEST = os.path.join(DATA, "X_test.csv")
Y_TEST = os.path.join(DATA, "y_test.csv")
X_TRAIN = os.path.join(DATA, "X_train.csv")      # fitur latih hasil preprocess (472.432 x 400)
RAW = os.path.join(DATA, "train_raw.csv")        # opsional

LEAK_COLS = {"isfraud", "is_fraud", "transactionid", "transactiondt"}
N_PERM = int(os.environ.get("N_PERM", 20))
N_BOOT = int(os.environ.get("N_BOOT", 200))
N_SEEDS = int(os.environ.get("N_SEEDS", 10))
RNG = np.random.default_rng(0)

report = {"warnings": []}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check(ok, msg):
    tag = "PASS" if ok else "WARN"
    print(f" [{tag}] {msg}")
    if not ok:
        report["warnings"].append(msg)


def metrics_at(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "flagged": int(pred.sum()),
        "flagged_pct": round(100 * float(pred.mean()), 3),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }


# ---------------------------------------------------------------- 0. muat
section("0. Lingkungan dan berkas")
for p in (MODEL_PATH, X_TEST, Y_TEST):
    if not os.path.exists(p):
        sys.exit(f"[ERROR] Tidak ditemukan: {p}")

env = {
    "python": platform.python_version(),
    "scikit_learn": sklearn.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
}
report["environment"] = env
print(" ", env)

model = joblib.load(MODEL_PATH)
X = pd.read_csv(X_TEST)
y = pd.read_csv(Y_TEST)["isFraud"].astype(int).to_numpy()

# ---------------------------------------------------------------- 1. model
section("1. Model tersimpan")
params = model.get_params()
print("  params   :", {k: params[k] for k in
                       ("n_estimators", "max_samples", "contamination", "max_features", "random_state")})
print("  offset_  :", model.offset_)
print("  n_features_in_ :", getattr(model, "n_features_in_", None))
report["model"] = {
    "contamination": params["contamination"],
    "max_samples": params["max_samples"],
    "n_estimators": params["n_estimators"],
    "offset_": float(model.offset_),
    "n_features_in_": int(getattr(model, "n_features_in_", -1)),
}
is_auto = params["contamination"] == "auto"
check(is_auto and abs(model.offset_ + 0.5) < 1e-9,
      "contamination='auto' dan offset_ = -0.5 (sesuai klaim naskah). "
      f"Nilai sebenarnya: contamination={params['contamination']!r}, offset_={model.offset_:.4f}")

# ---------------------------------------------------------------- 2. data
section("2. Integritas data uji dan indikasi kebocoran")
print(f"  X_test: {X.shape} | y_test: {y.shape} | fraud: {y.sum()} ({100*y.mean():.2f}%)")
check(len(X) == len(y), "Jumlah baris X_test sama dengan y_test (sejajar urutan baris)")
check(set(np.unique(y)) <= {0, 1}, "Label hanya 0/1")
leak = [c for c in X.columns if str(c).strip().lower() in LEAK_COLS]
check(not leak, f"Tidak ada kolom label/ID/waktu di X_test (ditemukan: {leak})")
if hasattr(model, "feature_names_in_"):
    same = list(model.feature_names_in_) == list(X.columns)
    check(same, "Kolom X_test identik (nama & urutan) dengan fitur saat model dilatih")
    if same is False:
        X = X[list(model.feature_names_in_)]
else:
    check(X.shape[1] == model.n_features_in_,
          f"Jumlah kolom X_test ({X.shape[1]}) = n_features_in_ ({model.n_features_in_})")
check(X.shape[1] == 377, f"Jumlah fitur = 377 seperti di naskah (aktual: {X.shape[1]})")
print("  NaN di X_test:", int(X.isna().sum().sum()))
print("  Baris duplikat di X_test:", int(X.duplicated().sum()))

X_tr = None
if os.path.exists(X_TRAIN):
    X_tr = pd.read_csv(X_TRAIN)
    h_tr = set(pd.util.hash_pandas_object(X_tr, index=False))
    h_te = pd.util.hash_pandas_object(X, index=False)
    overlap = int(h_te.isin(h_tr).sum())
    print(f"  X_train: {X_tr.shape}; baris X_test yang persis sama dengan baris di X_train: {overlap}")
    report["train_test_identical_rows"] = overlap
    print("  (catatan: baris identik bisa wajar pada data nyata; bandingkan dengan proporsi duplikat internal)")
else:
    print("  X_train.csv tidak ada -> uji tumpang tindih dan retrain dilewati")

# ---------------------------------------------------------------- 3. skor
section("3. Konsistensi skor dan ambang")
s_true = -model.score_samples(X)                 # s Persamaan (3), makalah Liu dkk.
dec = model.decision_function(X)                  # = -s - offset_
s_code = 0.5 - dec / 2                            # rumus di evaluasi.py
print("  Rumus evaluasi.py: scores = 0.5 - decision_function/2  (BUKAN s Persamaan 3)")
print("  Rumus naskah     : s = -score_samples")
print("  rentang s_true   : %.4f .. %.4f" % (s_true.min(), s_true.max()))
print("  rentang s_code   : %.4f .. %.4f" % (s_code.min(), s_code.max()))

pred_sklearn = (model.predict(X) == -1).astype(int)
pred_code = (s_code >= 0.5).astype(int)
pred_paper = (s_true >= 0.5).astype(int)          # tau = 0.5 pada s sesungguhnya
check((pred_sklearn == pred_code).all(),
      "Prediksi evaluasi.py identik dengan model.predict() (ambang mengikuti offset_ model)")
check(is_auto,
      "Jumlah peringatan pada tau=0.5 pada s asli: %d vs model.predict(): %d"
      % (pred_paper.sum(), pred_sklearn.sum()))
report["flagged"] = {
    "predict": int(pred_sklearn.sum()),
    "s_ge_0.5_true_s": int(pred_paper.sum()),
    "evaluasi_py": int(pred_code.sum()),
}

# ---------------------------------------------------------------- 4. metrik
section("4. Metrik")
res = {
    "pr_auc": round(float(AP(y, s_true)), 4),
    "roc_auc": round(float(AUC(y, s_true)), 4),
    "at_model_predict": metrics_at(y, pred_sklearn),
    "at_tau_0.5_true_s": metrics_at(y, pred_paper),
}
print(json.dumps(res, indent=2))
report["metrics"] = res

boot = []
idx_all = np.arange(len(y))
for _ in range(N_BOOT):
    b = RNG.choice(idx_all, size=len(idx_all), replace=True)
    if y[b].sum() > 0:
        boot.append(AP(y[b], s_true[b]))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"  PR-AUC bootstrap 95% CI (n={N_BOOT}): [{lo:.4f}, {hi:.4f}]")
report["pr_auc_ci95"] = [round(float(lo), 4), round(float(hi), 4)]

# ---------------------------------------------------------------- 5. sanity
section("5. Uji kewarasan (sanity check)")
perm = [AP(RNG.permutation(y), s_true) for _ in range(N_PERM)]
perm_mean = float(np.mean(perm))
print(f"  PR-AUC label diacak (rata-rata {N_PERM}x): {perm_mean:.4f}  (harus ~ {y.mean():.4f})")
check(abs(perm_mean - y.mean()) < 0.01, "PR-AUC label diacak mendekati prevalensi")
rand_ap = AP(y, RNG.random(len(y)))
print(f"  PR-AUC skor acak: {rand_ap:.4f}")

amt_col = next((c for c in X.columns if str(c).lower() in ("transactionamt", "transaction_amt")), None)
if amt_col:
    print(f"  Baseline hanya {amt_col}: PR-AUC={AP(y, X[amt_col]):.4f}  ROC-AUC={AUC(y, X[amt_col]):.4f}")
    report["baseline_amount"] = {"pr_auc": round(float(AP(y, X[amt_col])), 4),
                                 "roc_auc": round(float(AUC(y, X[amt_col])), 4)}

print("  ROC-AUC per fitur (maks(auc, 1-auc)) - 10 tertinggi:")
single = {}
for c in X.columns:
    col = X[c].to_numpy()
    if np.nanstd(col) == 0:
        continue
    a = AUC(y, col)
    single[c] = max(a, 1 - a)
top = sorted(single.items(), key=lambda kv: -kv[1])[:10]
for c, a in top:
    print(f"    {c:<20} {a:.4f}")
report["top_single_feature_auc"] = {c: round(float(a), 4) for c, a in top}
check(top[0][1] < 0.90,
      f"Tidak ada satu fitur pun yang hampir memisahkan fraud sendirian (maks {top[0][0]}={top[0][1]:.3f}); "
      "nilai >0.90 mengindikasikan kebocoran")

# ---------------------------------------------------------------- 6. retrain
if X_tr is not None and os.environ.get("SKIP_RETRAIN") != "1":
    section(f"6. Retrain {N_SEEDS} seed (contamination 'auto' vs 0.035)")
    if list(X_tr.columns) != list(X.columns):
        X_tr = X_tr[list(X.columns)]
    out = {}
    Xtr_np = X_tr.to_numpy(dtype=np.float32)
    Xte_np = X.to_numpy(dtype=np.float32)
    for cont in ("auto", 0.035):
        aps, aucs, flg, f1s = [], [], [], []
        for seed in range(N_SEEDS):
            m = IsolationForest(n_estimators=100, max_samples="auto",
                                contamination=cont, random_state=seed, n_jobs=-1)
            m.fit(Xtr_np)
            s = -m.score_samples(Xte_np)
            p = (s >= -m.offset_).astype(int)   # setara predict() == -1, tanpa menskor dua kali
            aps.append(AP(y, s)); aucs.append(AUC(y, s))
            flg.append(100 * p.mean()); f1s.append(f1_score(y, p, zero_division=0))
        out[str(cont)] = {
            "pr_auc": [round(float(np.mean(aps)), 4), round(float(np.std(aps)), 4)],
            "roc_auc": [round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4)],
            "f1": [round(float(np.mean(f1s)), 4), round(float(np.std(f1s)), 4)],
            "flagged_pct": [round(float(np.mean(flg)), 3), round(float(np.std(flg)), 3)],
        }
        print(f"  contamination={cont!r}: {json.dumps(out[str(cont)])}   (mean, SD)")
    report["retrain_multi_seed"] = out
    print("  Bandingkan dengan model tersimpan: PR-AUC=%.4f ROC-AUC=%.4f" % (res["pr_auc"], res["roc_auc"]))

# ---------------------------------------------------------------- 7. aturan NLG
if os.path.exists(RAW):
    section("7. Audit statistik aturan NLG (train_raw.csv)")
    raw = pd.read_csv(RAW, usecols=lambda c: c in ("TransactionAmt", "ProductCD", "isFraud"))
    n_total = len(raw)
    print(f"  Baris train_raw.csv: {n_total:,}")
    if X_tr is not None:
        cond = n_total == len(X_tr)
        print(f"  Baris X_train: {len(X_tr):,}; X_test: {len(X):,}; total: {len(X_tr)+len(X):,}")
        check(cond, "train_raw.csv hanya data LATIH (bukan seluruh dataset). Jika jumlahnya = total, "
                    "P99 dan tingkat fraud per ProductCD memakai label data uji (kebocoran ke NLG)")
    else:
        check(n_total != 590540, "train_raw.csv bukan 590.540 baris penuh (jika penuh, statistik NLG memakai data uji)")
    p99 = float(raw["TransactionAmt"].quantile(0.99))
    ps = raw.groupby("ProductCD")["isFraud"].agg(["mean", "count"])
    ps["mean"] = (ps["mean"] * 100).round(2)
    print(f"  P99 TransactionAmt: USD {p99:.2f}")
    print(ps.to_string())
    report["rules"] = {"p99": round(p99, 2), "product_fraud_rate_pct": ps["mean"].to_dict()}

with open(os.path.join(BASE, "audit_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

section("RINGKASAN")
if report["warnings"]:
    print(f"  {len(report['warnings'])} peringatan:")
    for w in report["warnings"]:
        print("   -", w)
else:
    print("  Semua pemeriksaan lolos.")
print("  Laporan lengkap: audit_report.json")
