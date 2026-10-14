# app/narrative_engine.py

class NarrativeEngine:
    # Pemetaan Kode Aturan Teknis -> Bahasa Awam / Non-Teknis
    RULE_MAP = {
        "RULE_HIGH_AMOUNT": {
            "title": "Nominal Transaksi Sangat Tinggi",
            "desc": "Nominal transaksi jauh melebihi batas rata-rata transaksi normal."
        },
        "RULE_DEVICE_MISMATCH": {
            "title": "Perangkat / Lokasi Berbeda",
            "desc": "Transaksi diakses dari perangkat atau alamat IP yang belum pernah digunakan."
        },
        "RULE_HIGH_FREQUENCY": {
            "title": "Frekuensi Transaksi Singkat",
            "desc": "Terjadi beberapa kali transaksi beruntun dalam rentang waktu sangat cepat."
        },
        "RULE_RISKY_PRODUCT": {
            "title": "Kategori Produk Berrisiko",
            "desc": "Produk yang dibeli memiliki riwayat tingkat kecurangan (fraud) tinggi."
        },
        "RULE_ANOMALY_SCORE_HIGH": {
            "title": "Penyimpangan Pola Transaksi (ML)",
            "desc": "Model Machine Learning mengidentifikasi kombinasi fitur yang tidak wajar."
        }
    }

    @classmethod
    def generate_narrative(cls, row_dict: dict, score: float, status: str, config: dict = None) -> tuple:
        triggered_rules_raw = []
        friendly_rules = []
        
        amt = float(row_dict.get('TransactionAmt', 0.0))
        product = str(row_dict.get('ProductCD', 'W'))
        
        # Logika Evaluasi Aturan
        if amt > 500.0:
            triggered_rules_raw.append("RULE_HIGH_AMOUNT")
        if product in ['C', 'H']:
            triggered_rules_raw.append("RULE_RISKY_PRODUCT")
        if score >= 0.50:
            triggered_rules_raw.append("RULE_ANOMALY_SCORE_HIGH")
            
        # Konversi ke format ramah non-teknis
        for rule_code in triggered_rules_raw:
            rule_info = cls.RULE_MAP.get(rule_code, {
                "title": rule_code,
                "desc": "Terdeteksi indikasi penyimpangan aturan operasional."
            })
            friendly_rules.append(rule_info)

        # Penyusunan Narasi Eksplanatif (XAI)
        if status.lower() in ['anomali', 'fraud']:
            reasons = [r["title"].lower() for r in friendly_rules]
            reason_str = ", ".join(reasons) if reasons else "penyimpangan statistik pada fitur transaksi"
            
            narrative = (
                f"Transaksi ini ditandai sebagai indikasi anomali (skor risiko {score:.2f}) "
                f"karena terdeteksi {reason_str}. Disarankan untuk melakukan verifikasi manual."
            )
        else:
            narrative = (
                f"Transaksi tergolong wajar (skor risiko {score:.2f}). "
                f"Seluruh parameter berada dalam batas kriteria operasional normal."
            )

        return narrative, friendly_rules