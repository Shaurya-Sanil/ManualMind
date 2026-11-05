"""
Unit Tests for Answering Module (M11) & FastAPI Endpoints
Tests step extraction, out-of-scope refusal, faithfulness checks,
and API route responses using FastAPI TestClient with lifespan context.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from nlp.qa import extract_numbered_steps, check_faithfulness, compose_answer
from nlp.ingest import ManualChunk


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_c:
        yield test_c


def test_extract_numbered_steps_numbered():
    """Verify regex extraction of ordered numbered steps (1. X 2. Y)."""
    text = (
        "To configure the guest network, follow these steps:\n"
        "1. Log in to the router web interface at http://192.168.1.1\n"
        "2. Navigate to Wireless > Guest Network\n"
        "3. Toggle Enable Guest Network to ON and assign a name\n"
        "4. Click Apply Changes to save."
    )
    steps = extract_numbered_steps(text)
    assert len(steps) == 4
    assert "Log in" in steps[0]
    assert "Guest Network" in steps[1]


def test_extract_numbered_steps_step_prefix():
    """Verify extraction of 'Step 1: ... Step 2: ...' patterns."""
    text = (
        "Troubleshooting internet LED:\n"
        "Step 1: Unplug the power cable from the broadband modem.\n"
        "Step 2: Wait 30 seconds before reconnecting the modem.\n"
        "Step 3: Verify the WAN LED turns solid white."
    )
    steps = extract_numbered_steps(text)
    assert len(steps) == 3
    assert "Unplug" in steps[0]
    assert "Wait 30 seconds" in steps[1]


def test_extract_numbered_steps_bullets():
    """Verify extraction of bullet point lists without splitting hyphenated compound words."""
    text = (
        "Key recommendations for placement:\n"
        "• Place router in an open central location\n"
        "• Avoid placing near microwave ovens or metal surfaces\n"
        "• Position antennas vertically at 90-degree angles"
    )
    steps = extract_numbered_steps(text)
    assert len(steps) == 3
    assert "open central location" in steps[0]
    assert "90-degree" in steps[2]


def test_check_faithfulness_faithful():
    """Verify lexical faithfulness check accepts grounded text."""
    evidence = "The Aurora AX3000 operates on 12V 2A DC power and uses a quad-core 1.5 GHz processor."
    generated = "The router uses a quad-core 1.5 GHz processor and requires 12V DC power."
    is_faithful, ratio = check_faithfulness(generated, evidence)
    assert is_faithful is True
    assert ratio >= 0.50


def test_check_faithfulness_hallucinated():
    """Verify lexical faithfulness check rejects ungrounded/hallucinated text."""
    evidence = "The Aurora AX3000 operates on 12V 2A DC power and uses a quad-core 1.5 GHz processor."
    hallucination = "To prepare delicious homemade blueberry pancakes, mix flour, sugar, and milk thoroughly."
    is_faithful, ratio = check_faithfulness(hallucination, evidence)
    assert is_faithful is False
    assert ratio < 0.20


def test_out_of_scope_intent_refusal():
    """Verify out-of-scope queries produce graceful refusal without hallucinating."""
    ans = compose_answer(
        query="What is the recipe for chocolate cake?",
        rewritten_query="What is the recipe for chocolate cake?",
        intent="out_of_scope",
        retrieved_chunks=[]
    )
    assert ans["confidence"] == 0.0
    assert "cannot assist" in ans["answer_text"].lower() or "outside" in ans["answer_text"].lower()
    assert len(ans["citations"]) == 0


def test_api_health_endpoint(client):
    """Verify GET /api/health returns operational status and loaded manual."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_manual" in data
    assert data["total_chunks_indexed"] > 0


def test_api_ask_endpoint(client):
    """Verify POST /api/ask runs complete pipeline and returns structured analysis."""
    payload = {
        "query": "What is the default IP address of the router?",
        "session_id": "test_api_session",
        "retrieval_mode": "hybrid",
        "tagger_mode": "spacy"
    }
    response = client.post("/api/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert len(data["answer"]["answer_text"]) > 10
    assert len(data["answer"]["citations"]) > 0
    assert "linguistic_analysis" in data
    assert "timings_ms" in data
    assert data["timings_ms"]["total_pipeline_ms"] > 0


def test_api_autocomplete_endpoint(client):
    """Verify GET /api/autocomplete returns n-gram predictions."""
    response = client.get("/api/autocomplete?prefix=guest&max_suggestions=5")
    assert response.status_code == 200
    data = response.json()
    assert "suggestions" in data
    assert isinstance(data["suggestions"], list)


def test_api_manuals_endpoint(client):
    """Verify GET /api/manuals lists the available documents."""
    response = client.get("/api/manuals")
    assert response.status_code == 200
    data = response.json()
    assert "manuals" in data
    assert any(m["filename"] == "aurora_ax3000_manual.pdf" for m in data["manuals"])


def test_api_lab_levenshtein_endpoint(client):
    """Verify GET /api/lab/levenshtein computes DP matrix."""
    response = client.get("/api/lab/levenshtein?word1=router&word2=roter")
    assert response.status_code == 200
    data = response.json()
    assert data["scratch_distance"] == 1
    assert "dp_matrix" in data
    assert len(data["dp_matrix"]) == 7  # len('router') + 1


def test_api_eval_results_endpoint(client):
    """Verify GET /api/eval/results returns generated evaluation data."""
    response = client.get("/api/eval/results")
    assert response.status_code == 200
    data = response.json()
    assert "retrieval" in data
    assert "answering" in data
