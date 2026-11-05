"""
Master Evaluation Suite Orchestrator (eval/run_all_evals.py)
Invoked by 'make eval'.
Executes every evaluation script sequentially with fixed random seed 42,
writes all individual JSON results and PNG figures, and builds the
consolidated summary_metrics.json for the frontend Evaluation page.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np

from eval.eval_retrieval import run_retrieval_evaluation
from eval.eval_answering import run_answering_evaluation
from eval.eval_intent import run_intent_evaluation
from eval.eval_pos import run_pos_evaluation
from eval.eval_ner import run_ner_evaluation
from eval.eval_ambiguity_understanding import run_ambiguity_understanding_evaluation
from eval.eval_ngram_latency import run_ngram_latency_evaluation


def run_full_suite():
    t_start = time.time()
    print("=" * 70)
    print("MANUALMIND NLP LABORATORY COMPREHENSIVE EVALUATION SUITE")
    print("=" * 70)

    # 1. Retrieval
    retrieval_metrics = run_retrieval_evaluation()
    print("-" * 50)

    # 2. Answering
    answering_metrics = run_answering_evaluation()
    print("-" * 50)

    # 3. Intent
    intent_metrics = run_intent_evaluation()
    print("-" * 50)

    # 4. POS
    pos_metrics = run_pos_evaluation()
    print("-" * 50)

    # 5. NER
    ner_metrics = run_ner_evaluation()
    print("-" * 50)

    # 6. Ambiguity & Understanding Ablation
    ambiguity_metrics = run_ambiguity_understanding_evaluation()
    print("-" * 50)

    # 7. N-Gram & Latency
    ngram_latency_metrics = run_ngram_latency_evaluation()
    print("-" * 50)

    # Consolidated Summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluation_time_seconds": round(time.time() - t_start, 2),
        "random_seed": 42,
        "retrieval": retrieval_metrics,
        "answering": answering_metrics,
        "intent": intent_metrics,
        "pos": pos_metrics,
        "ner": ner_metrics,
        "ambiguity_understanding": ambiguity_metrics,
        "ngram_latency": ngram_latency_metrics
    }

    summary_file = PROJECT_ROOT / "reports" / "results" / "summary_metrics.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nALL EVALUATIONS COMPLETED SUCCESSFULLY in {summary['total_evaluation_time_seconds']}s!")
    print(f"Summary metrics written to: {summary_file}")
    print("All visual figures written to: reports/figures/")
    print("=" * 70)


if __name__ == "__main__":
    run_full_suite()

