"""
ManualMind Configuration Module
Handles environment variables, application directories, model weights configuration,
and runtime settings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "ManualMind"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    RANDOM_SEED: int = 42

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 7860

    # Path Settings
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MANUALS_DIR: Path = DATA_DIR / "manuals"
    EVAL_DIR: Path = DATA_DIR / "eval"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    RESULTS_DIR: Path = REPORTS_DIR / "results"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"
    WEB_DIR: Path = BASE_DIR / "web"

    # Default Active Manual
    DEFAULT_MANUAL_NAME: str = "aurora_ax3000_manual.pdf"

    # Memory & Performance Tuning
    # Set LOW_MEMORY_MODE=True for 512MB RAM hosts (Render free tier), False for 2GB+ hosts
    LOW_MEMORY_MODE: bool = True

    # NLP Models Configuration
    SPACY_MODEL: str = "en_core_web_sm"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    QA_READER_MODEL: str = "deepset/roberta-base-squad2"
    QA_READER_FALLBACK: str = "deepset/minilm-uncased-squad2"
    NER_TRANSFORMER_MODEL: str = "dslim/bert-base-NER"
    GENERATIVE_MODEL: str = "google/flan-t5-base"
    WHISPER_MODEL_SIZE: str = "base.en"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"

    # Retrieval Thresholds & Tuning
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    RRF_K: int = 60
    TOP_K_RETRIEVAL: int = 5
    CHUNK_SIZE_WORDS: int = 150
    CHUNK_OVERLAP_WORDS: int = 30
    NO_ANSWER_PROB_THRESHOLD: float = 0.75

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure required runtime output directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MANUALS_DIR.mkdir(parents=True, exist_ok=True)
settings.EVAL_DIR.mkdir(parents=True, exist_ok=True)
settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
settings.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
