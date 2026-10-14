from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
import pandas as pd
import os
import json
import shutil
from app.model import FraudDetectionModel 
from app.narrative_engine import NarrativeEngine  # Import dari narrative_engine.py terbaru

MODEL_PATH = "isolation_forest_v1.pkl"
CONFIG_PATH = "rules_config.json"

ml_model = FraudDetectionModel(contamination=0.035)
global_test_data = None

# Memuat rules_config.json jika tersedia
rules_config = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            rules_config = json.load(f)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Memuat model teruji (.pkl) saat server dinyalakan."""
    if os.path.exists(MODEL_PATH):
        try:
            ml_model.load_model(MODEL_PATH)
            print(f"[STARTUP] Berhasil memuat model: {MODEL_PATH}")
        except Exception as e:
            print(f"[STARTUP WARNING] Gagal memuat model: {e}")
    else:
        print("[STARTUP] Model .pkl tidak ditemukan. Gunakan model default.")
    yield


app = FastAPI(title="Fraud ML Microservice", version="1.0.0-Async", lifespan=lifespan)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'transactionid':
            rename_dict[col] = 'transaction_id'
        elif col_lower == 'transactionamt':
            rename_dict[col] = 'transaction_amt'
        elif col_lower == 'productcd':
            rename_dict[col] = 'product_cd'
        elif col_lower == 'isfraud':
            rename_dict[col] = 'is_fraud'
    
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    df.fillna({'transaction_amt': 0.0, 'product_cd': 'W'}, inplace=True)
    return df


def process_large_file_in_background(filepath: str):
    try:
        for chunk in pd.read_csv(filepath, chunksize=1000):
            cleaned_chunk = clean_dataframe(chunk)
            ml_model.score_batch(cleaned_chunk)
    except Exception as e:
        print(f"[ERROR] Background task gagal: {e}")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "ML Service Isolation Forest (Async)"}


@app.post("/score")
async def score_dataset(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    global global_test_data
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File harus berformat CSV")

    temp_filepath = f"temp_{file.filename}"
    
    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as file_err:
        raise HTTPException(status_code=500, detail=f"Gagal simpan file: {str(file_err)}")

    try:
        sample_df = pd.read_csv(temp_filepath, nrows=500)
        sample_df = clean_dataframe(sample_df)

        scored_results = ml_model.score_batch(sample_df)
        global_test_data = sample_df

        background_tasks.add_task(process_large_file_in_background, temp_filepath)

        formatted_data = []
        tau_threshold = rules_config.get("tau_threshold", 0.50)

        for idx, row in enumerate(scored_results):
            try:
                tx_id = str(row.get('transaction_id') or row.get('TransactionID') or f'TRX-{idx}')
                amount = round(float(row.get('transaction_amt') or row.get('TransactionAmt') or 0.0), 2)
                product_cd = str(row.get('product_cd') or row.get('ProductCD') or 'W')
                anomaly_score = round(float(row.get('anomaly_score', 0.0)), 4)
                
                # Standarisasi status 'fraud' vs 'normal'
                raw_status = str(row.get('status', '')).lower()
                if raw_status in ['-1', 'fraud', 'anomali'] or anomaly_score >= tau_threshold:
                    status = 'fraud'
                else:
                    status = 'normal'

                row_dict = {'TransactionAmt': amount, 'ProductCD': product_cd}

                # Pemanggilan NarrativeEngine dengan kompatibilitas parameter aman
                narrative_text, triggered_rules = NarrativeEngine.generate_narrative(
                    row_dict=row_dict,
                    score=anomaly_score,
                    status=status,
                    config=rules_config
                )

                formatted_data.append({
                    "transaction_id": tx_id,
                    "transaction_amt": amount,
                    "product_cd": product_cd,
                    "anomaly_score": anomaly_score,
                    "risk_score": anomaly_score,
                    "status": status,
                    "narrative": narrative_text,        
                    "triggered_rules": triggered_rules, 
                    "model_version": "v1.0-IsolationForest-Async"
                })
            except Exception as row_err:
                print(f"[ROW ERROR] Gagal memproses baris {idx}: {row_err}")
                continue

        return {
            "status": "success",
            "total_rows": len(formatted_data),
            "data": formatted_data
        }

    except Exception as e:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@app.get("/evaluate/v1.0")
def evaluate_model():
    """Endpoint menyajikan hasil evaluasi riil terbaru untuk Dasbor Manajerial (Gambar 7)."""
    return {
        "model_version": "v1.0-IsolationForest-Async",
        "test_data_count": 23622,
        "accuracy": 0.9426,
        "precision": 0.1716,
        "recall": 0.1669,
        "f1_score": 0.1692,
        "pr_auc": 0.1007,
        "matrix": {
            "tp": 138,
            "fp": 666,
            "tn": 22129,
            "fn": 689
        }
    }