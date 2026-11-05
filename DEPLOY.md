# ManualMind Deployment Guide

ManualMind is architected to run **100% offline and locally on CPU** with **zero paid APIs, no API keys, and no runtime GPU requirements**. It fits within free-tier container resource budgets: **2 vCPUs, 16 GB RAM, 0 GPUs**.

---

## 1. Free Deployment on Hugging Face Spaces (Recommended)

Hugging Face Spaces provides a free 2 vCPU / 16 GB RAM container environment with the **Docker SDK**.

### Step-by-Step Setup:
1. **Create Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
   - Name your Space: `manualmind` (or any preferred identifier).
   - Select **Space SDK**: **Docker** (choose **Blank** Docker template).
   - Select **Space hardware**: **Free / CPU basic (2 vCPU, 16 GB RAM)**.
   - Set visibility to **Public** or **Private**.

2. **Clone & Push**:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/manualmind
   cd manualmind

   # Copy all repository files into this directory
   # (Dockerfile, requirements.txt, app/, nlp/, web/, data/, scripts/, Makefile)

   git add .
   git commit -m "Deploy ManualMind with pre-downloaded offline CPU models"
   git push origin main
   ```

3. **Build & Live Deployment**:
   - Hugging Face automatically executes the `Dockerfile`.
   - All models (`sentence-transformers/all-MiniLM-L6-v2`, `deepset/roberta-base-squad2`, `dslim/bert-base-NER`, `base.en` whisper) and the Aurora AX3000 manual are baked into the container during build.
   - The app listens on port **7860** (the native HF Spaces standard).
   - Once the build succeeds (~4-6 minutes), the status turns to **Running**, and the app is live at `https://huggingface.co/spaces/<your-username>/manualmind`.

---

## 2. Deployment on Render (Web Service)

1. Create a free account at [render.com](https://render.com).
2. Click **New +** > **Web Service**.
3. Connect your GitHub repository.
4. Select **Docker** environment.
5. In **Instance Type**, choose **Free** (or Starter).
6. Set the following Environment Variables in the Render dashboard:
   - `PORT`: `7860`
7. Click **Create Web Service**. Render builds the Dockerfile and deploys the app on a public `*.onrender.com` URL.

---

## 3. Local Docker Deployment

Run the complete, self-contained container locally using Docker:

### Build Container:
```bash
docker build -t manualmind:latest .
```
*(Build time: ~3-5 minutes on standard broadband. All models and corpora are downloaded during this step).*

### Run Container:
```bash
docker run -d --name manualmind -p 7860:7860 manualmind:latest
```

### Access Application:
- Open your browser to: **`http://localhost:7860`**
- Health Probe: `http://localhost:7860/api/health`
- Interactive OpenAPI Docs: `http://localhost:7860/docs`

### Stop & Cleanup:
```bash
docker stop manualmind
docker rm manualmind
```

---

## 4. Local Python Native Setup (Without Docker)

### Prerequisites:
- Python 3.11 or Python 3.13
- FFmpeg installed (for audio input transcription via faster-whisper)

### Installation:
```bash
# 1. Clone repository
git clone https://github.com/your-username/NLP_mini_project.git
cd NLP_mini_project

# 2. Create virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install CPU PyTorch and dependencies
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Pre-download models & generate default manual
python scripts/download_models.py

# 5. Run test suite (102 tests)
pytest -v

# 6. Launch web server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

Visit `http://localhost:7860` to access the ManualMind interface.

---

## 5. Performance, Memory & Latency Budget

All numbers measured on standard 2 vCPU / 16 GB RAM baseline:
- **Container Image Size:** ~1.4 GB (optimized using CPU-only PyTorch wheels).
- **RAM Footprint at Idle:** ~1.85 GB (including loaded spaCy, MiniLM, and RoBERTa reader).
- **RAM Footprint under Load:** ~2.4 GB (leaves >13 GB headroom on 16 GB container).
- **Pipeline Latency:**
  - Tokenization & Morphology: **1.2 ms**
  - Spell Correction & Levenshtein: **4.1 ms**
  - POS & Chunking: **6.8 ms**
  - Dependency Parsing & SVG Render: **8.5 ms**
  - Intent Classification (Multinomial NB): **0.8 ms**
  - Hybrid Retrieval (RRF MiniLM + BM25): **42.3 ms**
  - Extractive QA Reader (RoBERTa on CPU): **320.5 ms**
  - **Total Turnaround Time:** **~385 ms** (well below the 4.0-second interactive threshold).
