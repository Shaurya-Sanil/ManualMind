"""
Tests for Module M9: Retrieval and Text Representation
"""

import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp.ingest import ManualChunk
from nlp.retrieval import (
    ScratchTfidfVectorizer, ScratchBM25, TinyNumpySkipGram,
    HybridRetriever, verify_scratch_tfidf_against_sklearn,
    compare_sparse_vs_dense, expand_query_wordnet
)


def test_scratch_tfidf_against_sklearn():
    docs = [
        "the blue wan port connects to the broadband modem",
        "the yellow lan ports connect to desktop computers and consoles",
        "power on the router with the 12 V power supply adapter",
        "troubleshoot error code E-101 on the internet connection"
    ]
    query = "wan port broadband modem"

    # Scratch TF-IDF
    scratch = ScratchTfidfVectorizer()
    scratch.fit_transform(docs)
    scratch_ranks = scratch.rank(query, top_k=2)

    # Scikit-learn TF-IDF
    sk = TfidfVectorizer(token_pattern=r"\b[a-zA-Z0-9_\-]+\b", smooth_idf=True, norm='l2')
    sk_matrix = sk.fit_transform(docs)
    sk_q = sk.transform([query])
    sk_scores = (sk_matrix * sk_q.T).toarray().flatten()
    sk_top_idx = int(np.argmax(sk_scores))

    # Top-1 doc should be identical (Doc 0: blue wan port)
    assert scratch_ranks[0][0] == sk_top_idx
    assert scratch_ranks[0][0] == 0


def test_verify_scratch_tfidf_helper():
    docs = [
        "reset the admin password in web GUI",
        "check error code E-204 DNS failure"
    ]
    res = verify_scratch_tfidf_against_sklearn(docs, "admin password")
    assert res["top1_match"] is True


def test_scratch_bm25_scoring():
    docs = [
        "how to reset the router using the rear pin button",
        "how to configure the guest network and isolate devices",
        "technical specifications dual core processor and memory"
    ]
    bm25 = ScratchBM25(k1=1.5, b=0.75)
    bm25.fit(docs)

    scores = bm25.score("reset router pin")
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_tiny_numpy_skipgram_gradient_step():
    sgns = TinyNumpySkipGram(vocab_size=20, embed_dim=8, lr=0.1)
    initial_w_in = np.copy(sgns.W_in)

    loss = sgns.train_step(center_idx=2, context_idx=5, neg_indices=[8, 12, 15])
    assert loss > 0.0
    # Weights should be updated after gradient step
    assert not np.allclose(sgns.W_in, initial_w_in)


def test_hybrid_retriever_rrf_and_error_code_boost():
    chunks = [
        ManualChunk(
            chunk_id="chk_01",
            manual="aurora.pdf",
            page=1,
            section_path="LED Status Indicators",
            content="LED indicator status table: POWER LED solid white, INTERNET LED blinking red.",
            word_count=50
        ),
        ManualChunk(
            chunk_id="chk_02",
            manual="aurora.pdf",
            page=7,
            section_path="Diagnostic Error Codes",
            content="Error code E-204 indicates DNS resolution failure on the WAN link.",
            word_count=50
        ),
        ManualChunk(
            chunk_id="chk_03",
            manual="aurora.pdf",
            page=9,
            section_path="Warranty Information",
            content="Two-year limited hardware warranty covering manufacturing defects.",
            word_count=50
        )
    ]

    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    # Error code exact boost test
    results = retriever.retrieve("What should I do about error E-204?", mode="hybrid", top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "chk_02"
    assert "E-204" in results[0][0].content


def test_compare_sparse_vs_dense():
    res = compare_sparse_vs_dense("Aurora AX3000 Wi-Fi 6 router")
    assert "sparse_features" in res
    assert "dense_vector_preview" in res
    assert res["dense_dimension"] == 384
    assert len(res["dense_vector_preview"]) > 0


def test_expand_query_wordnet():
    expanded = expand_query_wordnet("start device", max_synonyms=1)
    assert "start" in expanded
    assert "device" in expanded
