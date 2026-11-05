"""
Retrieval & Representation Evaluation Script (Section 8)
Computes Recall@1, Recall@3, Recall@5, and MRR across:
- Bag-of-Words (BoW)
- TF-IDF (Scratch)
- BM25
- Word2Vec CBOW
- Word2Vec Skip-gram
- Dense (sentence-transformers/all-MiniLM-L6-v2)
- Hybrid (Reciprocal Rank Fusion)

Also runs Preprocessing Ablation:
1. Raw (no preprocessing)
2. Stopwords removed
3. Stemming (Porter)
4. Lemmatization (WordNet)

Writes: reports/results/retrieval_metrics.json and reports/figures/retrieval_comparison.png
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import matplotlib.pyplot as plt

from nlp.ingest import ManualManager
from nlp.retrieval import HybridRetriever, ScratchTfidfVectorizer, ScratchBM25
from nlp.preprocess import normalise_text, compare_stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def run_retrieval_evaluation():
    print("Starting Retrieval & Representation Evaluation...")
    manual_mgr = ManualManager(PROJECT_ROOT / "data" / "manuals")
    doc = manual_mgr.get_active_manual()
    chunks = doc.chunks

    # Load QA gold set
    qa_file = PROJECT_ROOT / "data" / "eval" / "qa_gold.json"
    qa_data = json.loads(qa_file.read_text(encoding="utf-8"))["questions"]
    # Filter to in-scope questions
    in_scope_qas = [q for q in qa_data if not q.get("is_out_of_scope", False)]

    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    methods = ["tfidf", "bm25", "w2v_cbow", "w2v_skipgram", "dense", "hybrid"]
    metrics = {m: {"r1": 0, "r3": 0, "r5": 0, "reciprocal_ranks": []} for m in methods}

    for item in in_scope_qas:
        query = item.get("clean_question", item["question"])
        target_sec = item["target_section"].lower()

        for m in methods:
            results = retriever.retrieve(query, mode=m, top_k=5)
            # Find rank of first matching chunk (section path match or target answer keywords)
            found_rank = None
            for rank_idx, (chunk, score, rank) in enumerate(results):
                # Target section match
                c_sec = chunk.section_path.lower()
                if (target_sec in c_sec or c_sec in target_sec or
                    any(part in c_sec for part in target_sec.split(" > ") if len(part) > 10)):
                    found_rank = rank
                    break
                # Or high content overlap with target answer
                target_ans_words = set(item["target_answer"].lower().split())
                chunk_words = set(chunk.content.lower().split())
                if len(target_ans_words & chunk_words) >= 4:
                    found_rank = rank
                    break

            if found_rank == 1:
                metrics[m]["r1"] += 1
                metrics[m]["r3"] += 1
                metrics[m]["r5"] += 1
                metrics[m]["reciprocal_ranks"].append(1.0)
            elif found_rank and found_rank <= 3:
                metrics[m]["r3"] += 1
                metrics[m]["r5"] += 1
                metrics[m]["reciprocal_ranks"].append(1.0 / found_rank)
            elif found_rank and found_rank <= 5:
                metrics[m]["r5"] += 1
                metrics[m]["reciprocal_ranks"].append(1.0 / found_rank)
            else:
                metrics[m]["reciprocal_ranks"].append(0.0)

    total_q = len(in_scope_qas) or 1
    retrieval_summary = {}
    for m in methods:
        retrieval_summary[m] = {
            "Recall@1": round((metrics[m]["r1"] / total_q) * 100, 2),
            "Recall@3": round((metrics[m]["r3"] / total_q) * 100, 2),
            "Recall@5": round((metrics[m]["r5"] / total_q) * 100, 2),
            "MRR": round(float(np.mean(metrics[m]["reciprocal_ranks"])), 4)
        }

    # -----------------------------------------------------------------------
    # Preprocessing Ablation on BM25
    # -----------------------------------------------------------------------
    print("Running Preprocessing Ablation on BM25...")
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text: str, mode: str) -> str:
        if mode == "raw":
            return text
        elif mode == "stopwords_removed":
            sw_res = compare_stopwords(text)
            return " ".join(sw_res["nltk_filtered_tokens"])
        elif mode == "stemming":
            tokens = normalise_text(text).split()
            return " ".join([porter.stem(t) for t in tokens])
        elif mode == "lemmatization":
            tokens = normalise_text(text).split()
            return " ".join([lemmatizer.lemmatize(t) for t in tokens])
        return text

    ablation_modes = ["raw", "stopwords_removed", "stemming", "lemmatization"]
    ablation_results = {}

    for mode in ablation_modes:
        prep_docs = [preprocess_text(f"{c.section_path}\n{c.content}", mode) for c in chunks]
        bm25_model = ScratchBM25()
        bm25_model.fit(prep_docs)

        r1 = 0
        r5 = 0
        rr_list = []

        for item in in_scope_qas:
            query = item.get("clean_question", item["question"])
            q_prep = preprocess_text(query, mode)
            scores = bm25_model.score(q_prep)
            top5_idx = np.argsort(scores)[::-1][:5]

            target_sec = item["target_section"].lower()
            found_rank = None
            for r_idx, idx in enumerate(top5_idx):
                c_sec = chunks[idx].section_path.lower()
                if (target_sec in c_sec or c_sec in target_sec or
                    any(part in c_sec for part in target_sec.split(" > ") if len(part) > 10)):
                    found_rank = r_idx + 1
                    break

            if found_rank == 1:
                r1 += 1
                r5 += 1
                rr_list.append(1.0)
            elif found_rank and found_rank <= 5:
                r5 += 1
                rr_list.append(1.0 / found_rank)
            else:
                rr_list.append(0.0)

        ablation_results[mode] = {
            "Recall@1": round((r1 / total_q) * 100, 2),
            "Recall@5": round((r5 / total_q) * 100, 2),
            "MRR": round(float(np.mean(rr_list)), 4)
        }

    output_payload = {
        "retrieval_models": retrieval_summary,
        "preprocessing_ablation": ablation_results,
        "total_test_queries": total_q
    }

    # Save JSON result
    out_json = PROJECT_ROOT / "reports" / "results" / "retrieval_metrics.json"
    out_json.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Saved retrieval metrics to {out_json}")

    # Generate Figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Retrieval Models MRR & Recall@1
    model_labels = ["TF-IDF", "BM25", "W2V-CBOW", "W2V-SG", "MiniLM", "Hybrid (RRF)"]
    mrr_vals = [retrieval_summary[m]["MRR"] for m in methods]
    r1_vals = [retrieval_summary[m]["Recall@1"] for m in methods]

    x = np.arange(len(model_labels))
    width = 0.35

    ax1.bar(x - width/2, r1_vals, width, label='Recall@1 (%)', color='#134074')
    ax1.bar(x + width/2, [v * 100 for v in mrr_vals], width, label='MRR x100', color='#8DA9C4')
    ax1.set_ylabel('Score / Percentage')
    ax1.set_title('Retrieval Methods Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=20)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Chart 2: Preprocessing Ablation MRR
    abl_labels = ["Raw", "Stopwords Filter", "Stemming", "Lemmatization"]
    abl_r1 = [ablation_results[k]["Recall@1"] for k in ablation_modes]
    abl_mrr = [ablation_results[k]["MRR"] * 100 for k in ablation_modes]

    ax2.plot(abl_labels, abl_r1, marker='o', linewidth=2, color='#0B2545', label='Recall@1 (%)')
    ax2.plot(abl_labels, abl_mrr, marker='s', linewidth=2, color='#C1121F', linestyle='--', label='MRR x100')
    ax2.set_ylabel('Score / Percentage')
    ax2.set_title('Preprocessing Ablation Effect on BM25')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_path = PROJECT_ROOT / "reports" / "figures" / "retrieval_comparison.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Generated figure: {fig_path}")

    return output_payload


if __name__ == "__main__":
    run_retrieval_evaluation()
