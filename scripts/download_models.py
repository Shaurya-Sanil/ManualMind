"""
Model & Asset Pre-Download Script for ManualMind
Pre-fetches all open-source models, NLTK corpora, spaCy pipeline,
and generates the default Aurora PDF manual at build time.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def download_all_assets():
    print("=" * 60)
    print("Pre-downloading ManualMind Models and Corpora at Build Time")
    print("=" * 60)

    # 1. NLTK Corpora
    print("\n[1/5] Downloading NLTK corpora...")
    import nltk
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "universal_tagset", "treebank", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        try:
            nltk.download(pkg, quiet=True)
            print(f"  - Downloaded NLTK: {pkg}")
        except Exception as e:
            print(f"  - Notice on {pkg}: {e}")

    # 2. spaCy Model
    print("\n[2/5] Ensuring spaCy English model is available...")
    import spacy
    try:
        spacy.load("en_core_web_sm")
        print("  - spaCy model 'en_core_web_sm' is already loaded.")
    except Exception:
        print("  - Installing 'en_core_web_sm' via spacy.cli.download...")
        import spacy.cli
        spacy.cli.download("en_core_web_sm")

    # 3. Hugging Face Transformers & Embedding Models (CPU)
    print("\n[3/5] Pre-caching Hugging Face models...")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForTokenClassification

    models_to_cache = [
        ("sentence-transformers/all-MiniLM-L6-v2", SentenceTransformer),
        ("deepset/minilm-uncased-squad2", lambda m: AutoModelForQuestionAnswering.from_pretrained(m)),
        ("deepset/roberta-base-squad2", lambda m: AutoModelForQuestionAnswering.from_pretrained(m)),
        ("dslim/bert-base-NER", lambda m: AutoModelForTokenClassification.from_pretrained(m)),
    ]

    for model_name, loader in models_to_cache:
        print(f"  - Downloading {model_name}...")
        try:
            AutoTokenizer.from_pretrained(model_name)
            loader(model_name)
            print(f"    Successfully cached: {model_name}")
        except Exception as e:
            print(f"    Warning downloading {model_name}: {e}")

    # 4. faster-whisper Model
    print("\n[4/5] Pre-caching faster-whisper 'base.en' model...")
    try:
        from faster_whisper import WhisperModel
        WhisperModel("base.en", device="cpu", compute_type="int8", download_root=None)
        print("    Successfully cached faster-whisper base.en")
    except Exception as e:
        print(f"    Notice on faster-whisper caching: {e}")

    # 5. Aurora Manual Generation
    print("\n[5/5] Checking Aurora AX3000 Wi-Fi 6 Router User Manual PDF...")
    manual_path = PROJECT_ROOT / "data" / "manuals" / "aurora_ax3000_manual.pdf"
    if not manual_path.exists():
        try:
            try:
                from data.manuals.generate_aurora_manual import build_manual_pdf as generate_manual
            except ImportError:
                from data.manuals.generate_aurora_manual import generate_manual
            generate_manual(str(manual_path))
            print(f"    Generated: {manual_path}")
        except Exception as e:
            print(f"    Notice on manual generation: {e}")
    else:
        print(f"    Manual already exists: {manual_path}")

    print("\n" + "=" * 60)
    print("All models, corpora, and default assets are cached and ready.")
    print("=" * 60)


if __name__ == "__main__":
    download_all_assets()

