import pandas as pd
from app.model import FraudDetectionModel
# 1. Load file dataset kamu yang punya kolom 'isFraud' atau 'is_fraud'
df = pd.read_csv("data_uji_20_percent.csv")  # <-- Ganti dengan nama file CSV datasetmu

# 2. Inisialisasi class model
fraud_model = FraudDetectionModel()

# 3. Bagi data menjadi train (80%) dan test (20%)
train_df, test_df = fraud_model.split_data(df)

# 4. Latih model pada data train
print("Sedang melatih model...")
fraud_model.train(train_df)

# 5. Jalankan evaluasi pada data test
print("Sedang menghitung evaluasi...")
hasil = fraud_model.evaluate_test_set(test_df)

# 6. Tampilkan hasilnya ke layar
import json
print("\n================ HASIL EVALUASI ================")
print(json.dumps(hasil, indent=4))