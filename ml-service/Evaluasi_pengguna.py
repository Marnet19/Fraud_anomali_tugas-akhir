import pandas as pd

# 1. Membaca data asli (pastikan nama file sesuai persis)
file_name = "Evaluasi Pengguna_ Sistem Pelaporan Investigasi Fraud E-Commerce (Jawaban).xlsx"
df = pd.read_excel(file_name)

# 2. Menghapus kolom yang berpotensi membocorkan identitas 
# (Sesuaikan atau tambahkan nama kolom yang ada di dalam file aslimu)
kolom_dihapus = ['Timestamp', 'Waktu', 'Nama', 'Nama Lengkap', 'Email', 'Alamat Email', 'No HP']
for col in kolom_dihapus:
    if col in df.columns:
        df = df.drop(columns=[col])

# 3. Membuat ID Responden (R-01, R-02, R-03, dst.) di posisi paling kiri
df.insert(0, 'ID Responden', ['R-{:02d}'.format(i+1) for i in range(len(df))])

# 4. Menyimpan data yang sudah dianonimkan ke file Excel baru
output_name = "Evaluasi_Anonim_Siap_Publikasi.xlsx"
df.to_excel(output_name, index=False)

print(f"Selesai! Data berhasil dianonimkan dan disimpan sebagai: {output_name}")