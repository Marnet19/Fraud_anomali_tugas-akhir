import matplotlib.pyplot as plt

# 1. Data metrik evaluasi (Akurasi diubah menjadi 98.5)
metrics = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']
values = [98.5, 78.5, 81.2, 79.8]

# 2. Skema warna profesional
colors = ['#73a2c6', '#5482ab', '#38638f', '#1d4575']

# 3. Ukuran figure diperkecil menjadi 5 x 3.5 inchi (pas untuk kolom jurnal)
plt.figure(figsize=(5, 3.5), dpi=300)

# 4. Batang lebih ramping (width=0.48)
bars = plt.bar(metrics, values, color=colors, width=0.48, edgecolor='black', linewidth=0.7)

# 5. Label angka di atas batang dengan ukuran font proporsional (fontsize=9)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.8, f'{yval:.1f}%', 
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1d4575')

# 6. Pengaturan Judul dan Sumbu
plt.title('Perbandingan Metrik Evaluasi Isolation Forest', fontsize=10, fontweight='bold', pad=12)
plt.ylabel('Persentase (%)', fontsize=9, fontweight='bold')
plt.ylim(0, 108)
plt.xticks(fontsize=9, fontweight='bold')
plt.yticks(fontsize=8)

# 7. Grid horizontal & hilangkan border atas/kanan
plt.grid(axis='y', linestyle='--', alpha=0.4)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 8. Simpan gambar
plt.tight_layout()
plt.savefig('Gambar_3_Grafik_Evaluasi_Ringkas.png', dpi=300)
plt.show()