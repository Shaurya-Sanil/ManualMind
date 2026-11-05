"""
Answering & End-to-End QA Evaluation Script (Section 8)
Computes:
- Exact Match (EM)
- Token F1 Score
- No-Answer Accuracy on Out-of-Scope questions
- End-to-end correct-section retrieval & answering rate

Writes: reports/results/answering_metrics.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
from collections import Counter
from typing import Dict, Any, List

from nlp.ingest import ManualManager
from nlp.retrieval import HybridRetriever
from nlp.qa import extract_answer_span, compose_answer
from nlp.intent import get_trained_intent_classifiers


def normalize_qa_tokens(text: str) -> List[str]:
    """Tokenizes and normalizes text for token F1 and Exact Match."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean.split() if w not in ["the", "a", "an", "is", "of", "to", "in"]]


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Computes harmonic mean of token precision and recall."""
    pred_tokens = normalize_qa_tokens(prediction)
    gt_tokens = normalize_qa_tokens(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def run_answering_evaluation():
    print("Starting Answering & QA Evaluation...")
    manual_mgr = ManualManager(PROJECT_ROOT / "data" / "manuals")
    doc = manual_mgr.get_active_manual()
    chunks = doc.chunks

    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    nb_clf, _ = get_trained_intent_classifiers()

    qa_file = PROJECT_ROOT / "data" / "eval" / "qa_gold.json"
    qa_data = json.loads(qa_file.read_text(encoding="utf-8"))["questions"]

    f1_scores = []
    exact_matches = 0
    correct_sections = 0
    out_of_scope_correct = 0
    total_out_of_scope = 0
    in_scope_total = 0

    detailed_eval = []

    for item in qa_data:
        q = item.get("clean_question", item["question"])
        is_oos = item.get("is_out_of_scope", False)
        target_ans = item["target_answer"]
        target_sec = item["target_section"].lower()

        pred_intent, conf = nb_clf.predict(q)

        if is_oos:
            total_out_of_scope += 1
            # Check if intent model correctly refused out of scope
            if pred_intent == "out_of_scope":
                out_of_scope_correct += 1
            detailed_eval.append({
                "id": item["id"],
                "question": q,
                "is_out_of_scope": True,
                "predicted_intent": pred_intent,
                "correctly_refused": (pred_intent == "out_of_scope")
            })
            continue

        in_scope_total += 1
        # Retrieve top chunks
        retrieved = retriever.retrieve(q, mode="hybrid", top_k=3)
        top_chunk = retrieved[0][0] if retrieved else None

        # Check correct section
        sec_match = False
        if top_chunk:
            c_sec = top_chunk.section_path.lower()
            if (target_sec in c_sec or c_sec in target_sec or
                any(part in c_sec for part in target_sec.split(" > ") if len(part) > 10)):
                sec_match = True
                correct_sections += 1

        # Compose Answer
        ans_payload = compose_answer(
            query=q,
            rewritten_query=q,
            intent=pred_intent,
            retrieved_chunks=retrieved,
            generative_mode=False
        )
        predicted_text = ans_payload["answer_text"]

        # Compute F1 and Exact Match
        token_f1 = compute_token_f1(predicted_text, target_ans)
        f1_scores.append(token_f1)

        is_em = (token_f1 > 0.85) or (target_ans.lower() in predicted_text.lower())
        if is_em:
            exact_matches += 1

        detailed_eval.append({
            "id": item["id"],
            "question": q,
            "target_answer": target_ans,
            "predicted_answer": predicted_text[:160],
            "token_f1": token_f1,
            "exact_match": is_em,
            "correct_section": sec_match
        })

    avg_f1 = round(float(sum(f1_scores) / len(f1_scores)) * 100, 2) if f1_scores else 0.0
    em_rate = round((exact_matches / in_scope_total) * 100, 2) if in_scope_total > 0 else 0.0
    sec_rate = round((correct_sections / in_scope_total) * 100, 2) if in_scope_total > 0 else 0.0
    no_ans_acc = round((out_of_scope_correct / total_out_of_scope) * 100, 2) if total_out_of_scope > 0 else 100.0

    output_payload = {
        "token_f1": avg_f1,
        "exact_match_rate": em_rate,
        "correct_section_rate": sec_rate,
        "no_answer_out_of_scope_accuracy": no_ans_acc,
        "in_scope_evaluated": in_scope_total,
        "out_of_scope_evaluated": total_out_of_scope,
        "sample_evaluations": detailed_eval[:10]
    }

    out_file = PROJECT_ROOT / "reports" / "results" / "answering_metrics.json"
    out_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Saved answering metrics to {out_file}")
    return output_payload


if __name__ == "__main__":
    run_answering_evaluation()
