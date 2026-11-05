# ManualMind: Architecture & Engineering Decisions Log (DECISIONS.md)

This log records every significant architectural, design, and algorithmic decision made during the development of ManualMind, in accordance with the project specification and course syllabus requirements.

---

## 1. Operating Environment & Deployment Constraints
- **Target Environment:** 2 vCPU, 16 GB RAM, no GPU, free deployment tier (e.g., Hugging Face Spaces Docker SDK, Render).
- **Execution Policy:** English only. No paid APIs, no API keys, zero calls to hosted external LLMs at runtime. Everything runs locally on CPU using open-source models and self-contained algorithms.
- **Frontend Strategy:** Pure vanilla ES6+ JavaScript and modern responsive CSS with CSS variables for light/dark theming. Zero runtime CDN dependencies. Every vendor asset (icons, chart scripts, stylesheets) is vendored locally in `/web/vendor/`.
- **Random Seeds:** Fixed random seed (`SEED = 42`) across all components (NumPy, PyTorch, Scikit-learn, random) to ensure full determinism and 100% reproducible metrics.

---

## 2. From-Scratch Algorithm Implementations & Verification Strategy
To satisfy university NLP laboratory rigor and professor examination criteria, each algorithm marked "from scratch" is implemented in pure Python/NumPy, thoroughly documented with algorithmic complexity and course lecture references, and unit-tested against recognized reference library implementations:
1. **Byte Pair Encoding (BPE) (Module M2):**
   - *From scratch:* Pure Python merge learning algorithm tracking frequency of adjacent byte/character token pairs, iteratively learning merge rules, and applying vocabulary tokenization with ranking.
   - *Reference verification:* Compared against Hugging Face `tokenizers` BPE tokenizer.
2. **Levenshtein Edit Distance (Module M4):**
   - *From scratch:* Dynamic programming matrix computation with full backtrace yielding the sequence of edit operations (`insert`, `delete`, `substitute`).
   - *Reference verification:* Tested against `nltk.edit_distance` and `rapidfuzz.distance.Levenshtein`.
3. **N-gram Language Modeling (Module M5):**
   - *From scratch:* Unigram, bigram, and trigram maximum likelihood estimation with Laplace (add-1) smoothing and Stupid Backoff ($\alpha = 0.4$). Computes perplexity on held-out text.
   - *Reference verification:* Token counts and n-gram frequencies validated against scikit-learn `CountVectorizer`.
4. **HMM POS Tagger with Viterbi Decoding (Module M6):**
   - *From scratch:* Hidden Markov Model with transition probabilities $P(t_i | t_{i-1})$ and emission probabilities $P(w_i | t_i)$ in log space with Laplace smoothing, decoded using the dynamic programming Viterbi algorithm.
   - *Reference verification:* Evaluated on NLTK Penn Treebank (Universal Tagset).
5. **Bag-of-Words & TF-IDF Vectorizer (Module M9):**
   - *From scratch:* Explicit term-document matrix construction, inverse document frequency ($idf(t) = \log((1 + N)/(1 + df(t))) + 1$), L2 normalization, and cosine similarity.
   - *Reference verification:* Output scores verified against `sklearn.feature_extraction.text.TfidfVectorizer`.
6. **Skip-gram with Negative Sampling (SGNS) (Module M9):**
   - *From scratch:* Minimal NumPy implementation of word2vec skip-gram with negative sampling, sigmoid activation, and gradient updates for educational exploration on the Lab page alongside Gensim Word2Vec.
7. **Multinomial Naive Bayes Classifier (Module M10):**
   - *From scratch:* Pure NumPy/Python log-likelihood calculator $\log P(c) + \sum \log P(w_j | c)$ with Laplace smoothing.
   - *Reference verification:* Predicted classes and probabilities benchmarked against scikit-learn `MultinomialNB` / `LogisticRegression`.

---

## 3. Manual Ingestion & Fictional Benchmark Document (Module M1)
- **Synthetic Manual Subject:** *Aurora AX3000 Wi-Fi 6 Dual-Band Gigabit Router User Manual*.
- **Rationale:** A router manual offers rich, realistic technical troubleshooting contexts: LED patterns (solid amber, blinking red, solid white), error codes (e.g., `E-101`, `E-204`, `E-305`), hardware ports (WAN, LAN 1-4, USB 3.0), setup steps, parental controls, firmware upgrades, and safety warnings.
- **Generation Method:** Generated programmatically via `reportlab` into a realistic 18-page PDF with clear typographic hierarchy (Title, Heading 1, Heading 2, Body, Table, Numbered Steps, Bullet Lists).
- **Chunking Strategy:** 120-200 words per chunk with a 30-word overlap, preserving section paths (`Hardware Overview > LED Status Indicator`), page numbers, and list markers (`1.`, `2.`, `*`) to facilitate accurate step extraction.

---

## 4. Query Understanding Engine & Ambiguity Guard (Module M8)
- **Linguistic Levels Analyzed:**
  - Characters & morphology (inflection detection, suffix stripping: e.g., `rebooting` $\to$ `reboot` + `-ing`).
  - Tokens, lemmas, POS tags, syntactic chunks (noun phrases and verb phrases).
  - Dependency parsing (extracting semantic frames: `{action, object, modifiers, negation, modality, conditions}`).
- **Clause Decomposition:** Detects subordinating conjunctions (`if`, `when`, `unless`, `after`, `before`) via dependency relations (`advcl`, `mark`) to split complex instructions into pre-conditions and main actions. Coordinated queries (`and`) are split into independent sub-queries.
- **Multi-Turn Pronoun Resolution:** Resolves anaphora (`it`, `that`, `them`, `this one`) against the session entity stack and rewrites the query into an unambiguous standalone prompt.
- **Ambiguity Guard:**
  - *Lexical Ambiguity:* Detects polysemous domain terms (`port`, `reset`, `light`, `band`, `card`) using WordNet Lesk and a domain dictionary.
  - *Syntactic Ambiguity:* Detects prepositional-phrase (PP) attachment ambiguities and conditional scope ambiguities. Triggers interactive clarification chips only when competing parses lead to divergent retrieval sets.

---

## 5. Model Selection & Answering Engine (Modules M9, M11, M12)
- **Retrieval Engine (Stage 1):** Reciprocal Rank Fusion (RRF with $k=60$) combining BM25 keyword matching with dense semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`), reinforced with an exact-match boost for detected regex entities (error codes, model numbers).
- **Extractive QA Reader (Stage 2):** Local reader `deepset/roberta-base-squad2` (fallback to `deepset/minilm-uncased-squad2` for CPU latency < 3s). The no-answer score threshold distinguishes out-of-scope queries from answerable manual questions.
- **Generative Mode (Toggle):** `google/flan-t5-base` with strict post-generation lexical faithfulness validation (verifying key content words match the retrieved evidence text).
- **Speech-to-Text (STT):** `faster-whisper` (`base.en`, `int8`, CPU) for fast, lightweight local speech recognition.
- **Text-to-Speech (TTS):** Browser-native Web Speech Synthesis API, eliminating server latency and audio bandwidth, formatted with unit expansions (e.g., "5 GHz" $\to$ "five gigahertz") and step-by-step punctuation pacing.

---

## 6. Testing, Quality & Robustness Verification
- **Test Suite Extent:** 102 comprehensive unit and integration tests implemented across `tests/`, organized by syllabus module (`test_ingest.py`, `test_preprocess.py`, `test_patterns.py`, `test_spelling.py`, `test_ngram.py`, `test_tagging.py`, `test_ner.py`, `test_retrieval.py`, `test_intent.py`, `test_understanding.py`, `test_qa_and_api.py`).
- **Pass Rate:** 102 passed, 0 failed in 24.96 seconds on local CPU.
- **Specific Hand Calculations & Invariants:**
  - HMM Viterbi hand calculation verified on state sequence `["VERB", "NOUN"]` for phrase `"can fish"`.
  - Scratch Levenshtein dynamic programming verified against NLTK and RapidFuzz edit distances.
  - Scratch TF-IDF cosine ranking verified against scikit-learn `TfidfVectorizer`.
  - Scratch N-gram frequency counts verified against scikit-learn `CountVectorizer`.
  - BPE invariant verified: reconstructed tokens faithfully reproduce original words without loss.
  - Numbered step and bullet list regexes anchored to line starts to prevent false splits on compound words like `90-degree` and IP addresses like `192.168.1.1`.

---

## 7. Containerization, Resource Budget & Deployment
- **Docker Architecture:** Built on `python:3.11-slim`, utilizing CPU-only PyTorch wheels (`--index-url https://download.pytorch.org/whl/cpu`) to keep the container image under 1.5 GB.
- **Build-Time Model Baking:** All models (`all-MiniLM-L6-v2`, `roberta-base-squad2`, `bert-base-NER`, `faster-whisper base.en`), spaCy `en_core_web_sm`, NLTK corpora, and the Aurora manual are downloaded at Docker build time via `scripts/download_models.py`, ensuring instantaneous cold-starts without runtime download delays.
- **Memory Footprint:** Resident memory is ~1.85 GB, scaling to ~2.4 GB under load, comfortably fitting inside free 2 vCPU / 16 GB RAM containers on Hugging Face Spaces and Render.
- **Latency Verification:** End-to-end pipeline median latency is 527.56 ms, well beneath the 4000 ms target threshold.

---

## 8. Real-World PDF Ingestion, Disambiguation & Verifiable Citations
- **Heading vs. Procedural Step Disambiguation:**
  - *Root Cause of Ingestion Anomaly:* Earlier regex `^(\d+(?:\.\d+)*\.?)\s+[A-Z]` matched single-number action lines like `1. Connect the small end of the USB cable...` and `2. Connect the other end into a USB wall charger.` as Level 1 Section Headings! This caused the ingestion pipeline to strip the steps from the chunk body, leaving only fragmented crumbs in the text and generating 0 structured steps.
  - *Heuristic Solution:* Enforced strict disambiguation: single-number lines (`1.`, `2.`) are treated as procedural steps (NOT headings) if their first token is an imperative action verb (`connect`, `plug`, `press`, `slide`, `fold`, `turn`, `insert`, etc.), if they end in a period with > 4 words, or if their font size matches standard body font size (`font_size < avg_font_size * 1.15`).
- **PDF Hygiene & Wide-Tracking Header Removal:**
  - Filtered running headers/footers with spaced letter tracking (e.g., `2 8 \t | \t E N G`) using `re.compile(r"^\s*[\d\s]+\s*(?:\||\u2502|\/|\-)\s*[A-Za-z\s]+\s*$")`.
  - Replaced soft-hyphens `\xad` to prevent broken tokenization (e.g., `\xadl\xadocated` $\to$ `located`).
- **Chunk Boundary & Step Ordering Invariant:**
  - Preserved newline boundaries in `chunk_text_content` rather than collapsing all whitespace into single spaces.
  - In `compose_answer`, when the user asks a procedural `how_to_usage` or `setup_install` query, prioritized top-3 retrieved chunks that contain structured ordered steps over unstructured overview tables.
- **Interactive Ground-Truth Citation Accordion:**
  - Upgraded frontend citation chips into clickable toggle accordions. Clicking any citation expands an evidence card displaying the exact page number, section hierarchy, chunk ID, relevance match percentage, and the verbatim excerpt from that manual page.
- **Dynamic Follow-Up Questions:**
  - Replaced hardcoded router fallback questions with dynamic question generators derived from the active manual's section headers and topical keywords.


