import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from app.narrative_engine import NarrativeEngine

MODEL_PATH = "isolation_forest_v1.pkl"

class FraudDetectionModel:
    def __init__(self, contamination: float = 0.035, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.model_version = "v1.0-IsolationForest"

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Memastikan hanya fitur numerik murni yang digunakan (menghapus ID, Waktu, & Label)"""
        cols_to_exclude = [
            'isFraud', 'is_fraud', 
            'TransactionID', 'transaction_id', 
            'TransactionDT', 'transaction_dt', 
            'data_split'
        ]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in cols_to_exclude]

    def load_model(self, filepath: str = MODEL_PATH):
        """Memuat model Isolation Forest teruji dari file .pkl"""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"✅ Model Isolation Forest berhasil dimuat dari {filepath}")
        else:
            print(f"⚠️ File model {filepath} tidak ditemukan. Model perlu dilatih ulang.")

    def save_model(self, filepath: str = MODEL_PATH):
        """Menyimpan binary model Isolation Forest ke file .pkl"""
        joblib.dump(self.model, filepath)
        print(f"✅ Model berhasil disimpan ke {filepath}")

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Membagi dataset 80% train dan 20% test stratified berdasarkan target asli"""
        target_col = 'isFraud' if 'isFraud' in df.columns else ('is_fraud' if 'is_fraud' in df.columns else None)
        
        if not target_col:
            raise ValueError("Dataset wajib memiliki kolom 'is_fraud' atau 'isFraud' untuk membagi data.")

        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df[target_col], 
            random_state=42
        )
        train_df['data_split'] = 'train'
        test_df['data_split'] = 'test'
        return train_df, test_df

    def train(self, train_df: pd.DataFrame):
        """Melatih model Isolation Forest secara nyata pada fitur numerik dataset IEEE-CIS"""
        feature_cols = self._get_feature_cols(train_df)
        X_train = train_df[feature_cols].fillna(0)
        self.model.fit(X_train)
        self.save_model(MODEL_PATH)

    def score_batch(self, df: pd.DataFrame) -> List[dict]:
        """Menghitung prediksi dan skor anomali MURNI menggunakan Isolation Forest"""
        if os.path.exists(MODEL_PATH) and not hasattr(self.model, "estimators_"):
            self.load_model(MODEL_PATH)

        feature_cols = self._get_feature_cols(df)
        X = df[feature_cols].fillna(0) if feature_cols else pd.DataFrame()

        if X.empty:
            raise ValueError("Dataset tidak memiliki kolom fitur numerik yang valid.")

        if not hasattr(self.model, "estimators_"):
            # Jika file .pkl belum pernah dibuat, latih model langsung dari data yang masuk
            self.model.fit(X)

        # 1. Eksekusi asli algoritma Isolation Forest
        scores_raw = self.model.decision_function(X)
        predictions = self.model.predict(X)  # Output resmi sklearn: -1 = Fraud (Anomali), 1 = Normal

        # 2. Normalisasi matematis skor keputusan ke rentang 0.0 - 1.0
        denom = (scores_raw.max() - scores_raw.min() + 1e-6)
        anomaly_scores = 1 - ((scores_raw - scores_raw.min()) / denom)

        results = []
        for idx, row in df.reset_index(drop=True).iterrows():
            is_anomaly = predictions[idx] == -1
            status = "fraud" if is_anomaly else "normal"
            score = round(float(anomaly_scores[idx]), 3)

            row_dict = row.to_dict()
            narrative, rules = NarrativeEngine.generate_narrative(row_dict, score, status)

            tx_id = str(row.get('TransactionID') or row.get('transaction_id') or f'TX-{idx+1000}')
            tx_dt = str(row.get('TransactionDT') or row.get('transaction_dt') or '')
            tx_amt = float(row.get('TransactionAmt') or row.get('transaction_amt') or 0.0)
            product_cd = str(row.get('ProductCD') or row.get('product_cd') or 'W')

            results.append({
                'transaction_id': tx_id,
                'transaction_dt': tx_dt,
                'transaction_amt': tx_amt,
                'product_cd': product_cd,
                'device_type': str(row.get('DeviceType', row.get('device_type', 'unknown'))),
                'device_info': str(row.get('DeviceInfo', row.get('device_info', 'unknown'))),
                'data_split': str(row.get('data_split', 'test')),
                'anomaly_score': score,
                'risk_score': score,
                'status': status,
                'model_version': self.model_version,
                'narrative': narrative,
                'triggered_rules': rules
            })
        return results

    def evaluate_test_set(self, test_df: pd.DataFrame) -> dict:
        """Kalkulasi MURNI Confusion Matrix & Metrik Evaluasi tanpa rekayasa angka"""
        if os.path.exists(MODEL_PATH) and not hasattr(self.model, "estimators_"):
            self.load_model(MODEL_PATH)

        target_col = 'isFraud' if 'isFraud' in test_df.columns else ('is_fraud' if 'is_fraud' in test_df.columns else None)
        
        # Jika kolom label asli tidak ada, sistem menolak menghitung metrik rekayasa
        if not target_col:
            return {
                "status": "error",
                "message": "Kolom ground truth 'is_fraud' / 'isFraud' tidak ditemukan pada dataset. Evaluasi membutuhkan label asli."
            }

        feature_cols = self._get_feature_cols(test_df)
        X_test = test_df[feature_cols].fillna(0)
        y_true = test_df[target_col].values

        if not hasattr(self.model, "estimators_"):
            return {
                "status": "error",
                "message": "Model Isolation Forest belum dimuat atau belum dilatih."
            }

        # Prediksi murni dari Isolation Forest pada data uji
        preds = self.model.predict(X_test)
        y_pred = [1 if p == -1 else 0 for p in preds]  # -1 (Anomali) = Fraud (1), 1 = Normal (0)

        # Menghitung PR-AUC membutuhkan skor anomali kontinu
        scores_raw = self.model.decision_function(X_test)
        denom = (scores_raw.max() - scores_raw.min() + 1e-6)
        anomaly_scores = 1 - ((scores_raw - scores_raw.min()) / denom)

        # Kalkulasi PR-AUC
        pr_auc = average_precision_score(y_true, anomaly_scores)

        # Hitung confusion matrix asli menggunakan scikit-learn
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        return {
            "model_version": self.model_version,
            "test_data_count": len(test_df),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "pr_auc": round(float(pr_auc), 4),
            "matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            }
        }