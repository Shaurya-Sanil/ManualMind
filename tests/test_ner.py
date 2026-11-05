"""
Tests for Module M7: Named Entity Recognition
"""

import pytest
from nlp.ner import (
    extract_rule_entities, get_crf_ner_model, extract_all_entities,
    highlight_entities_html
)


def test_rule_based_ner_types():
    text = "Fix error code E-101 on the WAN port of Aurora AX3000 at 5 GHz using reset"
    ents = extract_rule_entities(text)
    labels = {e["label"] for e in ents}

    assert "ERROR_CODE" in labels
    assert "COMPONENT" in labels
    assert "PRODUCT" in labels
    assert "MEASUREMENT" in labels
    assert "ACTION" in labels

    # Check exact values
    err_ent = next(e for e in ents if e["label"] == "ERROR_CODE")
    assert err_ent["text"] == "E-101"

    meas_ent = next(e for e in ents if e["label"] == "MEASUREMENT")
    assert meas_ent["text"] == "5 GHz"


def test_crf_ner_predictions():
    crf_ner = get_crf_ner_model()
    text = "Error code E-204 DNS failure on the WAN port"
    ents = crf_ner.extract(text)
    assert len(ents) >= 1
    assert any(e["label"] in ["ERROR_CODE", "COMPONENT", "SETTING"] for e in ents)


def test_extract_all_entities():
    text = "Enable Guest Network on Aurora router"
    res = extract_all_entities(text)
    assert "rule_based_entities" in res
    assert "crf_entities" in res
    assert "transformer_baseline_entities" in res


def test_highlight_entities_html():
    text = "Error code E-101 on WAN port"
    ents = [
        {"text": "E-101", "label": "ERROR_CODE", "start": 11, "end": 16},
        {"text": "WAN port", "label": "COMPONENT", "start": 20, "end": 28}
    ]
    html = highlight_entities_html(text, ents)
    assert '<span class="entity-pill"' in html
    assert '[ERROR_CODE]' in html
    assert '[COMPONENT]' in html
