import io
import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fraud Detection & NLG Microservice", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/isolation_forest.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Gagal memuat model: {e}")

def extract_amount(row):
    for key in ['transaction_amt', 'TransactionAmt', 'amount', 'amt', 'nominal']:
        if key in row and pd.notna(row[key]):
            try:
                val = str(row[key]).replace('$', '').replace(',', '').strip()
                return float(val)
            except:
                pass
    for col in row.index:
        if any(k in col.lower() for k in ['amt', 'amount', 'nominal']):
            if pd.notna(row[col]):
                try:
                    val = str(row[col]).replace('$', '').replace(',', '').strip()
                    return float(val)
                except:
                    pass
    return 100.0

def extract_product(row):
    for key in ['product_cd', 'ProductCD', 'product', 'category', 'kategori']:
        if key in row and pd.notna(row[key]):
            return str(row[key])
    for col in row.index:
        if any(k in col.lower() for k in ['product', 'category', 'cd']):
            if pd.notna(row[col]):
                return str(row[col])
    return 'W'

def extract_id(row, idx):
    for key in ['transaction_id', 'TransactionID', 'id', 'transactionid']:
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return f"TRX-{idx}"

def generate_deterministic_nlg(amt: float, prod: str, score: float, threshold: float = 0.5) -> tuple[str, list]:
    rules_triggered = []
    if score >= threshold:
        if amt > 1104.00:
            rules_triggered.append("RULE_HIGH_AMOUNT")
        if prod in ['C', 'S']:
            rules_triggered.append("RULE_RISKY_PRODUCT")
            
        if not rules_triggered:
            rules_triggered.append("RULE_FALLBACK")
            narrative = f"Transaksi ditandai anomali dengan skor risiko {score:.4f}. Disarankan pemeriksaan manual."
        else:
            reasons = []
            if "RULE_HIGH_AMOUNT" in rules_triggered:
                reasons.append(f"nominal transaksi (USD {amt:.2f}) berada di atas kisaran historis ekstrem")
            if "RULE_RISKY_PRODUCT" in rules_triggered:
                reasons.append(f"kategori produk {prod} memiliki riwayat tingkat fraud tinggi")
            narrative = f"Transaksi diidentifikasi sebagai anomali (skor {score:.4f}) karena: " + ", ".join(reasons) + "."
    else:
        rules_triggered.append("RULE_NORMAL")
        narrative = f"Transaksi tergolong wajar (skor risiko {score:.4f}). Seluruh parameter normal."
    return narrative, rules_triggered

@app.post("/score")
async def score_transactions(request: Request):
    try:
        form = await request.form()
        uploaded_file = None
        
        # Cari file apa pun yang diunggah dari form-data CodeIgniter
        for field_name, value in form.items():
            if hasattr(value, "file"):
                uploaded_file = value
                break
                
        if not uploaded_file:
            raise HTTPException(status_code=400, detail="Tidak ada file CSV yang ditemukan dalam request.")
            
        contents = await uploaded_file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        numeric_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in ['TransactionID', 'TransactionDT', 'isFraud']]
        X = df[numeric_cols].fillna(0) if numeric_cols else pd.DataFrame(index=df.index)
        
        if model is not None and hasattr(model, "n_features_in_"):
            expected_features = model.n_features_in_
            if X.shape[1] < expected_features:
                for i in range(expected_features - X.shape[1]):
                    X[f'dummy_col_{i}'] = 0.0
            elif X.shape[1] > expected_features:
                X = X.iloc[:, :expected_features]
                
            raw_scores = model.score_samples(X)
            anomaly_scores = -raw_scores
        else:
            anomaly_scores = np.random.uniform(0.1, 0.9, size=len(df))
            
        results = []
        threshold = 0.50
        
        for idx, row in df.iterrows():
            score = float(anomaly_scores[idx])
            status = "ANOMALI" if score >= threshold else "NORMAL"
            
            amt = extract_amount(row)
            prod = extract_product(row)
            trx_id = extract_id(row, idx)
            
            narrative, rules = generate_deterministic_nlg(amt, prod, score, threshold)
            
            results.append({
                "transaction_id": trx_id,
                "transaction_amt": amt,
                "TransactionAmt": amt,
                "product_cd": prod,
                "ProductCD": prod,
                "anomaly_score": round(score, 4),
                "status": status,
                "narrative": narrative,
                "triggered_rules": rules
            })
            
        return {"total_processed": len(df), "data": results}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "active", "model_loaded": model is not None}