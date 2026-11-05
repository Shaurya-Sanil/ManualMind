"""
Named Entity Recognition Evaluation Script (Section 8)
Evaluates 3 NER Approaches:
1. Rule-Based NER (Regex & Gazetteers)
2. Statistical CRF NER (BIO sequence modeling with word shape and POS features)
3. General Transformer Baseline (dslim/bert-base-NER)

Metrics:
- Precision, Recall, F1 per entity type
  (COMPONENT, ERROR_CODE, SETTING, PRODUCT, MEASUREMENT, ACTION)
- Macro F1 across all domain types

Writes: reports/results/ner_metrics.json and reports/figures/ner_metrics.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from nlp.ner import extract_rule_entities, get_crf_ner_model, extract_transformer_baseline_entities

ENTITY_TYPES = ["COMPONENT", "ERROR_CODE", "SETTING", "PRODUCT", "MEASUREMENT", "ACTION"]


def evaluate_ner_predictions(gold_entities: List[Dict[str, Any]], pred_entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Computes exact match span precision, recall, and F1 per entity type."""
    metrics = {t: {"tp": 0, "fp": 0, "fn": 0} for t in ENTITY_TYPES}

    for t in ENTITY_TYPES:
        t_gold = [e for e in gold_entities if e["label"] == t]
        t_pred = [e for e in pred_entities if e["label"] == t]

        # Match spans
        matched_gold = set()
        matched_pred = set()

        for p_idx, p in enumerate(t_pred):
            for g_idx, g in enumerate(t_gold):
                if g_idx not in matched_gold:
                    # Check overlap or exact match
                    if (p["start"] == g["start"] and p["end"] == g["end"]) or (p["text"].lower() == g["text"].lower()):
                        metrics[t]["tp"] += 1
                        matched_gold.add(g_idx)
                        matched_pred.add(p_idx)
                        break

        metrics[t]["fp"] += len(t_pred) - len(matched_pred)
        metrics[t]["fn"] += len(t_gold) - len(matched_gold)

    results = {}
    for t in ENTITY_TYPES:
        tp = metrics[t]["tp"]
        fp = metrics[t]["fp"]
        fn = metrics[t]["fn"]
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        results[t] = {
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "support": tp + fn
        }

    return results


def run_ner_evaluation():
    print("Starting NER Evaluation...")
    ner_file = PROJECT_ROOT / "data" / "ner" / "ner_gold_40.json"
    gold_data = json.loads(ner_file.read_text(encoding="utf-8"))["sentences"]

    crf_ner = get_crf_ner_model()

    all_gold = []
    rule_preds = []
    crf_preds = []
    trans_preds = []

    for item in gold_data:
        text = item["sentence"]
        gold_ents = item["entities"]
        all_gold.extend(gold_ents)

        r_ents = extract_rule_entities(text)
        c_ents = crf_ner.extract(text)
        t_ents = extract_transformer_baseline_entities(text)

        rule_preds.extend(r_ents)
        crf_preds.extend(c_ents)
        trans_preds.extend(t_ents)

    rule_scores = evaluate_ner_predictions(all_gold, rule_preds)
    crf_scores = evaluate_ner_predictions(all_gold, crf_preds)
    trans_scores = evaluate_ner_predictions(all_gold, trans_preds)

    def compute_macro_f1(scores_dict):
        f1s = [v["f1"] for v in scores_dict.values() if v["support"] > 0]
        return round(float(np.mean(f1s)), 2) if f1s else 0.0

    output_payload = {
        "rule_based_ner": {
            "macro_f1": compute_macro_f1(rule_scores),
            "per_entity_type": rule_scores
        },
        "statistical_crf_ner": {
            "macro_f1": compute_macro_f1(crf_scores),
            "per_entity_type": crf_scores
        },
        "transformer_baseline_bert": {
            "macro_f1": compute_macro_f1(trans_scores),
            "per_entity_type": trans_scores,
            "explanation": "General BERT-base-NER trained on CoNLL (PER/ORG/LOC) misses specialized router domain entities."
        }
    }

    out_file = PROJECT_ROOT / "reports" / "results" / "ner_metrics.json"
    out_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Saved NER metrics to {out_file}")

    # Generate Figure: F1 per Entity Type Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    types = [t for t in ENTITY_TYPES if rule_scores[t]["support"] > 0]
    rb_f1s = [rule_scores[t]["f1"] for t in types]
    crf_f1s = [crf_scores[t]["f1"] for t in types]
    trans_f1s = [trans_scores[t]["f1"] for t in types]

    x = np.arange(len(types))
    width = 0.25

    ax.bar(x - width, rb_f1s, width, label='Rule-Based NER', color='#134074')
    ax.bar(x, crf_f1s, width, label='CRF Statistical NER', color='#2A9D8F')
    ax.bar(x + width, trans_f1s, width, label='BERT General Baseline', color='#E76F51')

    ax.set_ylabel('F1 Score (%)')
    ax.set_title('Named Entity Recognition F1 by Entity Type')
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=15)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_path = PROJECT_ROOT / "reports" / "figures" / "ner_metrics.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Generated figure: {fig_path}")

    return output_payload


if __name__ == "__main__":
    run_ner_evaluation()
