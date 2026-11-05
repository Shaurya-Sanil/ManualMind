"""
Ambiguity Guard & Understanding Layer Ablation Evaluation Script (Section 8)
Computes:
1. Ambiguity Guard: Precision, Recall, F1 on 20 ambiguous & control queries.
2. Understanding Layer Ablation:
   Evaluates multi-turn pronoun queries and compound conditional queries:
   - WITH Understanding Layer (Pronoun Resolution + Clause Decomposition + Guard)
   - WITHOUT Understanding Layer (Ablated: raw queries passed directly)
   Demonstrates quantitatively that the NLP understanding layer improves answers.

Writes: reports/results/ambiguity_understanding_metrics.json and reports/figures/understanding_ablation.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import matplotlib.pyplot as plt

from nlp.ingest import ManualManager
from nlp.retrieval import HybridRetriever
from nlp.understanding import evaluate_ambiguity_guard, resolve_pronouns, ConversationSession, decompose_clauses
from nlp.tagging import get_spacy_nlp
from nlp.qa import extract_answer_span
from eval.eval_answering import compute_token_f1


def run_ambiguity_understanding_evaluation():
    print("Starting Ambiguity Guard & Understanding Layer Ablation Evaluation...")

    # 1. Ambiguity Guard Detection Evaluation
    ambiguity_metrics = evaluate_ambiguity_guard()

    # 2. Understanding Layer Ablation on QA Gold Set
    manual_mgr = ManualManager(PROJECT_ROOT / "data" / "manuals")
    doc = manual_mgr.get_active_manual()
    retriever = HybridRetriever()
    retriever.index_chunks(doc.chunks)
    nlp = get_spacy_nlp()

    qa_file = PROJECT_ROOT / "data" / "eval" / "qa_gold.json"
    qa_data = json.loads(qa_file.read_text(encoding="utf-8"))["questions"]

    # Filter to follow-up pronoun queries and compound queries
    multi_turn_pairs = []
    # Identify pairs: turn 1 followed by turn 2
    for i in range(len(qa_data) - 1):
        if "turn1" in qa_data[i]["id"] and "turn2" in qa_data[i + 1]["id"]:
            multi_turn_pairs.append((qa_data[i], qa_data[i + 1]))

    decomp_queries = [q for q in qa_data if "decomposition" in q.get("type", "")]

    # Evaluate WITH vs WITHOUT Understanding Layer
    with_layer_f1 = []
    without_layer_f1 = []
    with_layer_r1 = 0
    without_layer_r1 = 0

    ablation_items = []

    # A. Multi-turn pronoun evaluations
    for t1_item, t2_item in multi_turn_pairs:
        # Turn 1 sets the session context
        session = ConversationSession(session_id="eval_session")
        session.add_turn(
            query=t1_item["question"],
            answer=t1_item["target_answer"],
            extracted_entities=[{"text": t1_item["question"].split()[-2], "label": "COMPONENT"}]
        )

        q2_raw = t2_item["question"]
        target_ans = t2_item["target_answer"]
        target_sec = t2_item["target_section"].lower()

        # WITH Resolution:
        res = resolve_pronouns(q2_raw, session)
        q2_rewritten = res["rewritten_query"]

        ret_with = retriever.retrieve(q2_rewritten, mode="hybrid", top_k=3)
        chunk_with = ret_with[0][0] if ret_with else None
        ans_with = extract_answer_span(q2_rewritten, chunk_with.content)["answer"] if chunk_with else ""
        f1_with = compute_token_f1(ans_with, target_ans)
        with_layer_f1.append(f1_with)

        if chunk_with and (target_sec in chunk_with.section_path.lower() or chunk_with.section_path.lower() in target_sec):
            with_layer_r1 += 1

        # WITHOUT Resolution (Ablated: raw 'it' passed directly):
        ret_without = retriever.retrieve(q2_raw, mode="hybrid", top_k=3)
        chunk_without = ret_without[0][0] if ret_without else None
        ans_without = extract_answer_span(q2_raw, chunk_without.content)["answer"] if chunk_without else ""
        f1_without = compute_token_f1(ans_without, target_ans)
        without_layer_f1.append(f1_without)

        if chunk_without and (target_sec in chunk_without.section_path.lower() or chunk_without.section_path.lower() in target_sec):
            without_layer_r1 += 1

        ablation_items.append({
            "id": t2_item["id"],
            "raw_query": q2_raw,
            "rewritten_query": q2_rewritten,
            "f1_with_understanding": f1_with,
            "f1_without_understanding": f1_without
        })

    # B. Decomposition queries
    for d_item in decomp_queries:
        q_raw = d_item["question"]
        target_ans = d_item["target_answer"]
        target_sec = d_item["target_section"].lower()

        # WITH Decomposition:
        doc_spacy = nlp(q_raw)
        decomp = decompose_clauses(q_raw, doc_spacy)
        sub_qs = decomp["sub_questions"]
        # Retrieve for sub questions
        ret_sub = retriever.retrieve(sub_qs[0], mode="hybrid", top_k=3)
        chunk_with = ret_sub[0][0] if ret_sub else None
        ans_with = extract_answer_span(sub_qs[0], chunk_with.content)["answer"] if chunk_with else ""
        f1_with = compute_token_f1(ans_with, target_ans)
        with_layer_f1.append(f1_with)

        if chunk_with and (target_sec in chunk_with.section_path.lower() or chunk_with.section_path.lower() in target_sec):
            with_layer_r1 += 1

        # WITHOUT Decomposition (compound question passed as one):
        ret_without = retriever.retrieve(q_raw, mode="hybrid", top_k=3)
        chunk_without = ret_without[0][0] if ret_without else None
        ans_without = extract_answer_span(q_raw, chunk_without.content)["answer"] if chunk_without else ""
        f1_without = compute_token_f1(ans_without, target_ans)
        without_layer_f1.append(f1_without)

        if chunk_without and (target_sec in chunk_without.section_path.lower() or chunk_without.section_path.lower() in target_sec):
            without_layer_r1 += 1

    total_eval = len(with_layer_f1) or 1
    avg_f1_with = round(float(np.mean(with_layer_f1)) * 100, 2)
    avg_f1_without = round(float(np.mean(without_layer_f1)) * 100, 2)
    r1_with = round((with_layer_r1 / total_eval) * 100, 2)
    r1_without = round((without_layer_r1 / total_eval) * 100, 2)

    output_payload = {
        "ambiguity_guard_metrics": {
            "precision": round(ambiguity_metrics["precision"] * 100, 2),
            "recall": round(ambiguity_metrics["recall"] * 100, 2),
            "f1": round(ambiguity_metrics["f1"] * 100, 2),
            "accuracy": round(ambiguity_metrics["accuracy"] * 100, 2),
            "total_benchmark_queries": ambiguity_metrics["total_queries"]
        },
        "understanding_layer_ablation": {
            "queries_evaluated": total_eval,
            "with_understanding_layer": {
                "token_f1": avg_f1_with,
                "recall_at_1": r1_with
            },
            "without_understanding_layer_ablated": {
                "token_f1": avg_f1_without,
                "recall_at_1": r1_without
            },
            "absolute_f1_improvement": round(avg_f1_with - avg_f1_without, 2),
            "relative_f1_gain_percent": round(((avg_f1_with - avg_f1_without) / max(0.1, avg_f1_without)) * 100, 2)
        },
        "sample_ablations": ablation_items[:6]
    }

    out_file = PROJECT_ROOT / "reports" / "results" / "ambiguity_understanding_metrics.json"
    out_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Saved ambiguity & understanding metrics to {out_file}")

    # Generate Figure: Ablation Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Subplot 1: Ambiguity Guard Metrics
    metrics_names = ["Precision", "Recall", "F1 Score", "Accuracy"]
    guard_vals = [
        output_payload["ambiguity_guard_metrics"]["precision"],
        output_payload["ambiguity_guard_metrics"]["recall"],
        output_payload["ambiguity_guard_metrics"]["f1"],
        output_payload["ambiguity_guard_metrics"]["accuracy"]
    ]
    ax1.bar(metrics_names, guard_vals, color='#134074', width=0.45)
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Ambiguity Guard Detection Performance')
    ax1.set_ylim(0, 105)
    for i, v in enumerate(guard_vals):
        ax1.text(i, v + 2, f"{v}%", ha='center', fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Subplot 2: Understanding Layer Ablation (With vs Without)
    abl_cats = ["Token F1", "Recall@1"]
    vals_with = [avg_f1_with, r1_with]
    vals_without = [avg_f1_without, r1_without]

    x = np.arange(len(abl_cats))
    width = 0.35

    ax2.bar(x - width/2, vals_with, width, label='With Understanding Layer', color='#2A9D8F')
    ax2.bar(x + width/2, vals_without, width, label='Without Understanding (Ablated)', color='#E76F51')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Ablation: Understanding Layer Impact')
    ax2.set_xticks(x)
    ax2.set_xticklabels(abl_cats)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_path = PROJECT_ROOT / "reports" / "figures" / "understanding_ablation.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Generated figure: {fig_path}")

    return output_payload


if __name__ == "__main__":
    run_ambiguity_understanding_evaluation()

