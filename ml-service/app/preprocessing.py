import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any

class DataPreprocessor:
    """Modul praproses data, rekayasa fitur, dan pembagian dataset."""

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.numeric_medians: Dict[str, float] = {}

    def merge_datasets(self, df_transaction: pd.DataFrame, df_identity: pd.DataFrame = None) -> pd.DataFrame:
        """Menggabungkan dataset transaksi dan identitas berdasarkan TransactionID."""
        if df_identity is not None and not df_identity.empty:
            return pd.merge(df_transaction, df_identity, on="TransactionID", how="left")
        return df_transaction.copy()

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menambahkan fitur turunan rasio TransactionAmt terhadap rata-rata per ProductCD."""
        df = df.copy()
        if "TransactionAmt" in df.columns and "ProductCD" in df.columns:
            product_amt_mean = df.groupby("ProductCD")["TransactionAmt"].transform("mean")
            df["amt_to_mean_productcd"] = df["TransactionAmt"] / (product_amt_mean + 1e-5)
        else:
            df["amt_to_mean_productcd"] = 1.0
        return df

    def preprocess(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Melakukan imputasi missing value, encoding, dan rekayasa fitur.
        Mengembalikan tuple: (DataFrame Fitur terenkode, DataFrame Utuh).
        """
        df_proc = df.copy()
        df_proc = self.add_derived_features(df_proc)

        # Abaikan kolom kunci dan label dari fitur training
        ignore_cols = ["TransactionID", "isFraud", "data_split"]
        feature_cols = [c for c in df_proc.columns if c not in ignore_cols]

        num_cols = df_proc[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_proc[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

        # Imputasi nilai hilang numerik (median)
        for col in num_cols:
            if is_training:
                self.numeric_medians[col] = float(df_proc[col].median())
            df_proc[col] = df_proc[col].fillna(self.numeric_medians.get(col, 0.0))

        # Imputasi dan encoding kategorikal
        for col in cat_cols:
            df_proc[col] = df_proc[col].fillna("unknown").astype(str)
            if is_training:
                le = LabelEncoder()
                df_proc[col] = le.fit_transform(df_proc[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df_proc[col] = df_proc[col].map(
                        lambda val: le.transform([val])[0] if val in le.classes_ else -1
                    )

        return df_proc[feature_cols], df_proc

    def create_stratified_split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        """
        Membagi data 80:20 stratified berdasarkan label isFraud.
        Menambahkan kolom 'data_split' ('train' / 'test').
        """
        if "isFraud" not in df.columns:
            raise ValueError("Kolom 'isFraud' wajib ada untuk melakukan stratified split.")

        df = df.copy()
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            stratify=df["isFraud"],
            random_state=random_state
        )

        df["data_split"] = "train"
        df.loc[test_idx, "data_split"] = "test"
        return df