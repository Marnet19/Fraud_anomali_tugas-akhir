import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Konfigurasi terpusat untuk Layanan ML."""
    APP_NAME: str = "Fraud Detection ML Service"
    APP_VERSION: str = "1.0.0"
    
    # Hyperparameter default Isolation Forest (Proporsi fraud IEEE-CIS ~3.5%)
    DEFAULT_CONTAMINATION: float = 0.035
    DEFAULT_N_ESTIMATORS: int = 100
    DEFAULT_MAX_SAMPLES: float = 0.8
    RANDOM_STATE: int = 42
    
    # Path Direktori Penyimpanan Model (.joblib)
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models_store")

settings = Settings()
os.makedirs(settings.MODEL_DIR, exist_ok=True)