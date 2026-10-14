import pandas as pd
import os

print("⏳ Memproses dataset 'train_transaction.csv' dengan metode Chunking (Aman RAM)...")

file_path = "train_transaction.csv"
output_filename_full = "data_uji_20_percent.csv"
output_filename_sample = "data_uji_sampel_500.csv"

if not os.path.exists(file_path):
    print("❌ Error: File 'train_transaction.csv' tidak ditemukan di folder ml-service!")
else:
    kolom_penting = ['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'isFraud']
    
    # Target utama sesuai naskah jurnal
    target_normal = 113974
    target_fraud = 4134
    
    # Target sampel untuk web dashboard (450 normal, 50 fraud = 500 baris)
    sample_target_normal = 450
    sample_target_fraud = 50
    
    list_normal_chunks = []
    list_fraud_chunks = []
    
    terkumpul_normal = 0
    terkumpul_fraud = 0
    
    try:
        # Membaca file per 50.000 baris agar RAM tidak jebol
        for chunk in pd.read_csv(file_path, usecols=kolom_penting, chunksize=50000):
            chunk = chunk.dropna(subset=['TransactionAmt', 'isFraud'])
            
            chunk_normal = chunk[chunk['isFraud'] == 0]
            chunk_fraud = chunk[chunk['isFraud'] == 1]
            
            if terkumpul_normal < target_normal:
                butuh_n = target_normal - terkumpul_normal
                ambil_n = chunk_normal.head(butuh_n)
                list_normal_chunks.append(ambil_n)
                terkumpul_normal += len(ambil_n)
                
            if terkumpul_fraud < target_fraud:
                butuh_f = target_fraud - terkumpul_fraud
                ambil_f = chunk_fraud.head(butuh_f)
                list_fraud_chunks.append(ambil_f)
                terkumpul_fraud += len(ambil_f)
                
            if terkumpul_normal >= target_normal and terkumpul_fraud >= target_fraud:
                break
        
        print("⏳ Menggabungkan dan menstandarisasi data uji...")
        df_normal_final = pd.concat(list_normal_chunks).head(target_normal)
        df_fraud_final = pd.concat(list_fraud_chunks).head(target_fraud)
        
        # 1. Buat Dataset Utama (118.108 baris)
        test_df = pd.concat([df_normal_final, df_fraud_final]).sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = test_df.rename(columns={
            'TransactionID': 'transaction_id',
            'TransactionDT': 'transaction_dt',
            'TransactionAmt': 'transaction_amt',
            'ProductCD': 'product_cd',
            'isFraud': 'is_fraud'
        })
        test_df.to_csv(output_filename_full, index=False)

        # 2. Buat Dataset Sampel Web Dashboard (Tepat 500 baris: 450 normal, 50 fraud)[cite: 5]
        sample_normal = df_normal_final.head(sample_target_normal)
        sample_fraud = df_fraud_final.head(sample_target_fraud)
        sample_df = pd.concat([sample_normal, sample_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
        sample_df = sample_df.rename(columns={
            'TransactionID': 'transaction_id',
            'TransactionDT': 'transaction_dt',
            'TransactionAmt': 'transaction_amt',
            'ProductCD': 'product_cd',
            'isFraud': 'is_fraud'
        })
        sample_df.to_csv(output_filename_sample, index=False)

        print("\n✅ BERHASIL MEMBUAT SELURUH DATA UJI SINKRON JURNAL!")
        print(f"📁 File Utama       : {output_filename_full} ({len(test_df):,} baris)")
        print(f"📁 File Sampel Web  : {output_filename_sample} ({len(sample_df)} baris - 450 Normal, 50 Fraud)[cite: 5]")

    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")