"""
ManualMind FastAPI Application (app/main.py)
Provides REST API endpoints for:
- /api/ask: Complete 12-stage customer-support QA pipeline & NLP Inspector analysis
- /api/stt: Speech-to-Text transcription via faster-whisper
- /api/manuals: Ingestion, stats, and manual switching
- /api/lab/{tool}: Interactive syllabus experimentation tools
- /api/evolution: Method Evolution Lab comparing 5 historical paradigms
- /api/eval/results: Serves evaluation metrics & figures
- /api/health: Health check & readiness probe
- Static Web UI hosting with zero runtime CDN calls
"""

import sys
import time
import json
import io
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.schemas import (
    AskRequest, AskResponse, SpellingResult, PronounResolutionResult,
    LinguisticAnalysis, AmbiguityReport, IntentResult, ClauseDecomposition,
    AnswerPayload, ChunkResult, Citation, STTResponse, ManualInfo, ManualListResponse,
    AutocompleteResponse, MethodEvolutionResponse, EraResult
)

from nlp.ingest import ManualManager, ScannedPDFError
from nlp.preprocess import (
    normalise_text, compare_tokenizers, compare_stopwords,
    compare_stemming_lemmatization
)
from nlp.patterns import extract_all_patterns
from nlp.spelling import SpellingCorrector, validate_levenshtein
from nlp.ngram import ScratchNgramLM
from nlp.tagging import tag_sentence_all_models, extract_chunks
from nlp.ner import extract_all_entities, highlight_entities_html
from nlp.understanding import analyze_query_full, get_or_create_session
from nlp.intent import get_trained_intent_classifiers, route_intent
from nlp.retrieval import (
    HybridRetriever, verify_scratch_tfidf_against_sklearn,
    compare_sparse_vs_dense
)
from nlp.qa import compose_answer
from nlp.speech import transcribe_audio_bytes
from nlp.evolution import run_all_eras_comparison


# ---------------------------------------------------------------------------
# Global Application State & Startup Lifecycle
# ---------------------------------------------------------------------------

manual_manager = ManualManager(settings.MANUALS_DIR)
retriever = HybridRetriever()
spelling_corrector = SpellingCorrector()
ngram_lm = ScratchNgramLM(backoff_alpha=0.4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing ManualMind services and loading default manual...")
    try:
        # Load active manual
        active_doc = manual_manager.get_active_manual()
        print(f"Loaded active manual '{active_doc.manual_name}' ({len(active_doc.chunks)} chunks)")

        # Index retriever
        retriever.index_chunks(active_doc.chunks)
        print("Indexed Hybrid Retriever (BM25 + Word2Vec + MiniLM)")

        # Seed spelling corrector
        spelling_corrector.fit_corpus([c.content for c in active_doc.chunks])

        # Train n-gram LM for autocomplete
        ngram_lm.train([c.content for c in active_doc.chunks])

        # Warm up intent classifier
        get_trained_intent_classifiers()
        print("Intent classifiers pre-warmed")

    except Exception as e:
        print(f"Startup warning (will initialize on first query): {e}")

    yield
    print("Shutting down ManualMind.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Customer-support assistant for product manuals explained through real NLP concepts.",
    lifespan=lifespan
)

# Enable CORS for local testing and browser API requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core QA Pipeline Endpoint: POST /api/ask
# ---------------------------------------------------------------------------

@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    t_start = time.time()
    timings = {}

    query_raw = request.query.strip()
    if not query_raw:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Normalization & Spelling Correction
    t0 = time.time()
    normalised = normalise_text(query_raw)
    spelling_res = spelling_corrector.correct_query(query_raw)
    effective_query = spelling_res["corrected"]
    timings["preprocessing_spelling_ms"] = round((time.time() - t0) * 1000, 2)

    # 2. Multi-Turn Pronoun Resolution & Session History
    t0 = time.time()
    session = get_or_create_session(request.session_id)
    pronoun_res = spelling_res  # Default placeholder
    timings["pronoun_resolution_ms"] = round((time.time() - t0) * 1000, 2)

    # 3. Comprehensive Linguistic Analysis (Morphology, POS, Chunks, Dependencies, Entities, Frame, Ambiguity)
    t0 = time.time()
    analysis_full = analyze_query_full(
        query=effective_query,
        session_id=request.session_id,
        tagger_mode=request.tagger_mode or "spacy"
    )
    timings["linguistic_analysis_ms"] = round((time.time() - t0) * 1000, 2)

    rewritten_query = analysis_full["pronoun_resolution"]["rewritten_query"]
    ambiguity_data = analysis_full["ambiguity"]

    # 4. Intent Classification
    t0 = time.time()
    nb_clf, _ = get_trained_intent_classifiers()
    intent_name, intent_conf = nb_clf.predict(rewritten_query)
    intent_probs = nb_clf.predict_proba(rewritten_query)
    route_action = route_intent(intent_name, intent_conf)
    timings["intent_classification_ms"] = round((time.time() - t0) * 1000, 2)

    # 5. Stage 1: Hybrid Retrieval
    t0 = time.time()
    # Check if user made clarification selection
    retrieval_query = rewritten_query
    if request.clarification_selection:
        retrieval_query = f"{rewritten_query} {request.clarification_selection}"

    retrieved = retriever.retrieve(
        query=retrieval_query,
        mode=request.retrieval_mode or "hybrid",
        top_k=settings.TOP_K_RETRIEVAL
    )
    timings["retrieval_ms"] = round((time.time() - t0) * 1000, 2)

    # 6. Stage 2: Answer Composition & Extractive QA Reader
    t0 = time.time()
    ans_payload = compose_answer(
        query=query_raw,
        rewritten_query=retrieval_query,
        intent=intent_name,
        retrieved_chunks=retrieved,
        generative_mode=request.generative_mode
    )
    timings["answering_ms"] = round((time.time() - t0) * 1000, 2)
    timings["total_pipeline_ms"] = round((time.time() - t_start) * 1000, 2)

    # Record turn into session memory
    session.add_turn(
        query=query_raw,
        answer=ans_payload["answer_text"],
        extracted_entities=analysis_full["entities"]
    )

    # Format chunk results for API schema
    chunk_results = []
    for rank_idx, (chunk, score, rank) in enumerate(retrieved):
        chunk_results.append(ChunkResult(
            chunk_id=chunk.chunk_id,
            manual=chunk.manual,
            page=chunk.page,
            section_path=chunk.section_path,
            score=round(float(score), 4),
            rank=rank,
            content=chunk.content,
            has_numbered_steps=chunk.has_numbered_steps
        ))

    return AskResponse(
        session_id=session.session_id,
        query=query_raw,
        rewritten_query=rewritten_query,
        answer=AnswerPayload(
            answer_text=ans_payload["answer_text"],
            steps=ans_payload["steps"],
            citations=[Citation(**c) for c in ans_payload["citations"]],
            confidence=ans_payload["confidence"],
            is_extractive=ans_payload["is_extractive"],
            generative_text=ans_payload.get("generative_text"),
            low_confidence=ans_payload["low_confidence"],
            suggested_questions=ans_payload["suggested_questions"],
            refusal_reason=ans_payload.get("refusal_reason")
        ),
        normalised_text=analysis_full["normalised_text"],
        spelling=SpellingResult(
            original=spelling_res["original"],
            corrected=spelling_res["corrected"],
            was_corrected=spelling_res["was_corrected"],
            did_you_mean=spelling_res["did_you_mean"],
            edits=spelling_res["edits"]
        ),
        pronoun_resolution=PronounResolutionResult(
            has_pronoun=analysis_full["pronoun_resolution"]["has_pronoun"],
            resolved_entity=analysis_full["pronoun_resolution"]["resolved_entity"],
            rewritten_query=analysis_full["pronoun_resolution"]["rewritten_query"],
            anaphora_detected=analysis_full["pronoun_resolution"]["anaphora_detected"]
        ),
        linguistic_analysis=LinguisticAnalysis(
            tokens=analysis_full["tokens"],
            lemmas=analysis_full["lemmas"],
            morphology=analysis_full["morphology"],
            pos_tags=analysis_full["pos_tags"],
            chunks=analysis_full["chunks"],
            entities=analysis_full["entities"],
            dependency_tree_svg=analysis_full["dependency_tree_svg"],
            semantic_frame=analysis_full["semantic_frame"]
        ),
        ambiguity=AmbiguityReport(
            is_ambiguous=ambiguity_data["is_ambiguous"],
            ambiguity_type=ambiguity_data["ambiguity_type"],
            trigger_words=ambiguity_data["trigger_words"],
            readings=ambiguity_data["readings"],
            clarification_needed=ambiguity_data["clarification_needed"],
            clarification_question=ambiguity_data.get("clarification_question"),
            clarification_options=ambiguity_data.get("clarification_options", [])
        ),
        intent=IntentResult(
            intent=intent_name,
            confidence=round(intent_conf, 4),
            probabilities=intent_probs,
            routing_action=route_action
        ),
        decomposition=ClauseDecomposition(
            is_compound=analysis_full["decomposition"]["is_compound"],
            sub_questions=analysis_full["decomposition"]["sub_questions"],
            conditions=analysis_full["decomposition"]["conditions"],
            main_actions=analysis_full["decomposition"]["main_actions"]
        ),
        retrieval=chunk_results,
        timings_ms=timings
    )


# ---------------------------------------------------------------------------
# Speech-To-Text Endpoint: POST /api/stt
# ---------------------------------------------------------------------------

@app.post("/api/stt", response_model=STTResponse)
async def speech_to_text(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio payload.")

        res = transcribe_audio_bytes(audio_bytes, filename_hint=file.filename or "audio.webm")
        return STTResponse(
            transcript=res["transcript"],
            duration_seconds=res["duration_seconds"],
            detected_language=res["detected_language"],
            confidence=res["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription error: {str(e)}")


# ---------------------------------------------------------------------------
# Autocomplete Endpoint: GET /api/autocomplete
# ---------------------------------------------------------------------------

@app.get("/api/autocomplete", response_model=AutocompleteResponse)
async def get_autocomplete(prefix: str = Query(..., max_length=200), max_suggestions: int = 5):
    suggestions = ngram_lm.autocomplete(prefix, max_suggestions=max_suggestions)
    return AutocompleteResponse(prefix=prefix, suggestions=suggestions)


# ---------------------------------------------------------------------------
# Manual Management Endpoints: GET / POST /api/manuals
# ---------------------------------------------------------------------------

@app.get("/api/manuals", response_model=ManualListResponse)
async def list_manuals():
    manuals_list = manual_manager.list_manuals()
    active_id = manual_manager.active_manual_id or ""
    return ManualListResponse(
        manuals=[ManualInfo(**m) for m in manuals_list],
        active_manual_id=active_id
    )


@app.post("/api/manuals")
async def upload_manual(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_manual.pdf"
    ext = Path(filename).suffix.lower()
    if ext not in [".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and Markdown files are supported.")

    dest = settings.MANUALS_DIR / filename
    contents = await file.read()
    dest.write_bytes(contents)

    try:
        doc = manual_manager.load_manual(filename)
        # Re-index retriever
        retriever.index_chunks(doc.chunks)
        spelling_corrector.fit_corpus([c.content for c in doc.chunks])
        ngram_lm.train([c.content for c in doc.chunks])

        return {
            "status": "success",
            "message": f"Successfully ingested '{filename}'",
            "pages": doc.total_pages,
            "sections": len(doc.sections),
            "chunks": len(doc.chunks),
            "vocabulary_size": doc.vocabulary_size
        }
    except ScannedPDFError as se:
        if dest.exists():
            dest.unlink()
        raise HTTPException(status_code=422, detail=str(se))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest manual: {str(e)}")


@app.post("/api/manuals/{filename}/activate")
async def switch_active_manual(filename: str):
    try:
        doc = manual_manager.set_active_manual(filename)
        retriever.index_chunks(doc.chunks)
        spelling_corrector.fit_corpus([c.content for c in doc.chunks])
        ngram_lm.train([c.content for c in doc.chunks])
        return {"status": "success", "active_manual": filename}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Manual '{filename}' not found.")


# ---------------------------------------------------------------------------
# Method Evolution Lab: POST /api/evolution
# ---------------------------------------------------------------------------

@app.post("/api/evolution", response_model=MethodEvolutionResponse)
async def run_method_evolution(query: str = Form(...)):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    res = run_all_eras_comparison(query.strip(), retriever)
    return MethodEvolutionResponse(
        query=res["query"],
        eras=[EraResult(**e) for e in res["eras"]],
        historical_benchmark_accuracies=res["historical_benchmark_accuracies"]
    )


# ---------------------------------------------------------------------------
# NLP Lab Experiment Endpoints: GET /api/lab/{tool}
# ---------------------------------------------------------------------------

@app.get("/api/lab/tokenization")
async def lab_tokenization(text: str = Query("Resetting the router to factory default")):
    return compare_tokenizers(text)


@app.get("/api/lab/stopwords")
async def lab_stopwords(text: str = Query("How do I connect the blue WAN cable to my modem")):
    return compare_stopwords(text)


@app.get("/api/lab/stem_lemmatize")
async def lab_stem_lemmatize(text: str = Query("resetting configured antennas switches connecting")):
    return compare_stemming_lemmatization(text)


@app.get("/api/lab/regex")
async def lab_regex(text: str = Query("Fix error code E-101 on AX3000-PRO at Settings > Wireless > 5 GHz")):
    return {"matches": extract_all_patterns(text)}


@app.get("/api/lab/levenshtein")
async def lab_levenshtein(word1: str = Query("routter"), word2: str = Query("router")):
    return validate_levenshtein(word1, word2)


@app.get("/api/lab/ngram")
async def lab_ngram(prompt: str = Query("how to reset")):
    held_out = "The router power LED is solid white when online. Error code E-101 indicates WAN disconnect."
    return {
        "prompt": prompt,
        "autocomplete_suggestions": ngram_lm.autocomplete(prompt, max_suggestions=5),
        "unigram_perplexity": ngram_lm.compute_perplexity(held_out, n=1),
        "bigram_perplexity": ngram_lm.compute_perplexity(held_out, n=2),
        "trigram_perplexity": ngram_lm.compute_perplexity(held_out, n=3),
    }


@app.get("/api/lab/pos")
async def lab_pos(sentence: str = Query("Disconnect the power adapter and check the internet light")):
    return tag_sentence_all_models(sentence)


@app.get("/api/lab/chunking")
async def lab_chunking(sentence: str = Query("The blue WAN port connects the broadband modem quickly")):
    return extract_chunks(sentence)


@app.get("/api/lab/ner")
async def lab_ner(sentence: str = Query("Fix error code E-101 on the WAN port of Aurora AX3000 at 5 GHz")):
    res = extract_all_entities(sentence)
    highlighted = highlight_entities_html(sentence, res["rule_based_entities"])
    res["highlighted_html"] = highlighted
    return res


@app.get("/api/lab/tfidf")
async def lab_tfidf(query: str = Query("error code E-204 DNS failure")):
    sample_docs = [c.content for c in retriever.chunks[:20]]
    if not sample_docs:
        sample_docs = ["Sample router documentation"]
    return verify_scratch_tfidf_against_sklearn(sample_docs, query)


@app.get("/api/lab/vector_neighbours")
async def lab_vector_neighbours(word: str = Query("router"), top_n: int = 6):
    return {"word": word, "neighbours": retriever.w2v.get_nearest_neighbours(word, top_n=top_n)}


@app.get("/api/lab/sparse_dense")
async def lab_sparse_dense(sentence: str = Query("Aurora AX3000 Wi-Fi 6 router delivers high speed")):
    return compare_sparse_vs_dense(sentence)


@app.get("/api/lab/intent")
async def lab_intent(query: str = Query("Why is the internet light flashing red?")):
    nb_clf, sk_clf = get_trained_intent_classifiers()
    nb_pred, nb_conf = nb_clf.predict(query)
    sk_pred, sk_conf = sk_clf.predict(query)
    return {
        "query": query,
        "naive_bayes_scratch": {
            "predicted_intent": nb_pred,
            "confidence": round(nb_conf, 4),
            "probabilities": nb_clf.predict_proba(query)
        },
        "logistic_regression_sklearn": {
            "predicted_intent": sk_pred,
            "confidence": round(sk_conf, 4),
            "probabilities": sk_clf.predict_proba(query)
        }
    }


# ---------------------------------------------------------------------------
# Evaluation Results Endpoint: GET /api/eval/results
# ---------------------------------------------------------------------------

@app.get("/api/eval/results")
async def get_evaluation_results():
    summary_path = settings.RESULTS_DIR / "summary_metrics.json"
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading evaluation summary: {str(e)}")

    # If summary does not exist, assemble from individual metric files
    ret_path = settings.RESULTS_DIR / "retrieval_metrics.json"
    ans_path = settings.RESULTS_DIR / "answering_metrics.json"
    int_path = settings.RESULTS_DIR / "intent_metrics.json"
    pos_path = settings.RESULTS_DIR / "pos_metrics.json"
    ner_path = settings.RESULTS_DIR / "ner_metrics.json"
    amb_path = settings.RESULTS_DIR / "ambiguity_understanding_metrics.json"
    lat_path = settings.RESULTS_DIR / "ngram_latency_metrics.json"

    assembled = {}
    for key, p in [
        ("retrieval", ret_path), ("answering", ans_path), ("intent", int_path),
        ("pos", pos_path), ("ner", ner_path), ("ambiguity_understanding", amb_path),
        ("ngram_latency", lat_path)
    ]:
        if p.exists():
            assembled[key] = json.loads(p.read_text(encoding="utf-8"))

    if assembled:
        return assembled

    raise HTTPException(status_code=404, detail="Evaluation results not found. Run 'make eval' first.")


# ---------------------------------------------------------------------------
# Health Probe Endpoint: GET /api/health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "active_manual": manual_manager.active_manual_id or settings.DEFAULT_MANUAL_NAME,
        "total_chunks_indexed": len(retriever.chunks),
        "is_indexed": retriever.is_indexed
    }


# ---------------------------------------------------------------------------
# Static Web Frontend Hosting
# ---------------------------------------------------------------------------

app.mount("/css", StaticFiles(directory=settings.WEB_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=settings.WEB_DIR / "js"), name="js")
app.mount("/vendor", StaticFiles(directory=settings.WEB_DIR / "vendor"), name="vendor")
app.mount("/reports/figures", StaticFiles(directory=settings.FIGURES_DIR), name="figures")


@app.get("/")
async def serve_index():
    index_file = settings.WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ManualMind API is running. Place index.html in web/ directory."}
