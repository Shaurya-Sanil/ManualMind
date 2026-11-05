"""
Intent Classification Evaluation Script (Section 8)
Compares:
1. Scratch Multinomial Naive Bayes (log-probabilities & Laplace smoothing)
2. Scikit-learn Logistic Regression on TF-IDF n-grams

Metrics:
- Overall Accuracy
- Macro Precision, Recall, F1
- Per-Class Precision, Recall, F1
- Confusion Matrices

Writes: reports/results/intent_metrics.json and reports/figures/intent_confusion_matrices.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from nlp.intent import ScratchMultinomialNB, SklearnIntentClassifier, INTENT_CLASSES


def run_intent_evaluation():
    print("Starting Intent Classification Evaluation...")
    train_file = PROJECT_ROOT / "data" / "intent" / "intent_train.json"
    test_file = PROJECT_ROOT / "data" / "intent" / "intent_test.json"

    train_data = json.loads(train_file.read_text(encoding="utf-8"))["data"]
    test_data = json.loads(test_file.read_text(encoding="utf-8"))["data"]

    train_texts = [d["query"] for d in train_data]
    train_labels = [d["intent"] for d in train_data]

    test_texts = [d["query"] for d in test_data]
    test_labels = [d["intent"] for d in test_data]

    # 1. Train Scratch Naive Bayes
    nb = ScratchMultinomialNB(alpha=0.8)
    nb.fit(train_texts, train_labels)

    # 2. Train Sklearn Logistic Regression
    lr = SklearnIntentClassifier()
    lr.fit(train_texts, train_labels)

    # 3. Evaluate Predictions
    nb_preds = [nb.predict(t)[0] for t in test_texts]
    lr_preds = [lr.predict(t)[0] for t in test_texts]

    # Reports
    nb_report = classification_report(test_labels, nb_preds, labels=INTENT_CLASSES, output_dict=True, zero_division=0)
    lr_report = classification_report(test_labels, lr_preds, labels=INTENT_CLASSES, output_dict=True, zero_division=0)

    # Confusion matrices
    cm_nb = confusion_matrix(test_labels, nb_preds, labels=INTENT_CLASSES)
    cm_lr = confusion_matrix(test_labels, lr_preds, labels=INTENT_CLASSES)

    output_payload = {
        "classes": INTENT_CLASSES,
        "total_test_samples": len(test_labels),
        "naive_bayes_scratch": {
            "accuracy": round(nb_report["accuracy"] * 100, 2),
            "macro_precision": round(nb_report["macro avg"]["precision"] * 100, 2),
            "macro_recall": round(nb_report["macro avg"]["recall"] * 100, 2),
            "macro_f1": round(nb_report["macro avg"]["f1-score"] * 100, 2),
            "per_class": {
                c: {
                    "precision": round(nb_report[c]["precision"] * 100, 2),
                    "recall": round(nb_report[c]["recall"] * 100, 2),
                    "f1": round(nb_report[c]["f1-score"] * 100, 2)
                } for c in INTENT_CLASSES
            },
            "confusion_matrix": cm_nb.tolist()
        },
        "logistic_regression_sklearn": {
            "accuracy": round(lr_report["accuracy"] * 100, 2),
            "macro_precision": round(lr_report["macro avg"]["precision"] * 100, 2),
            "macro_recall": round(lr_report["macro avg"]["recall"] * 100, 2),
            "macro_f1": round(lr_report["macro avg"]["f1-score"] * 100, 2),
            "per_class": {
                c: {
                    "precision": round(lr_report[c]["precision"] * 100, 2),
                    "recall": round(lr_report[c]["recall"] * 100, 2),
                    "f1": round(lr_report[c]["f1-score"] * 100, 2)
                } for c in INTENT_CLASSES
            },
            "confusion_matrix": cm_lr.tolist()
        }
    }

    out_file = PROJECT_ROOT / "reports" / "results" / "intent_metrics.json"
    out_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Saved intent metrics to {out_file}")

    # Plot Confusion Matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(cm_nb, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title(f'Scratch Naive Bayes (Acc: {nb_report["accuracy"]*100:.1f}%)')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(INTENT_CLASSES))
    short_labels = [c[:8] for c in INTENT_CLASSES]
    ax1.set_xticks(tick_marks)
    ax1.set_xticklabels(short_labels, rotation=45)
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(short_labels)
    ax1.set_ylabel('True Intent')
    ax1.set_xlabel('Predicted Intent')

    im2 = ax2.imshow(cm_lr, interpolation='nearest', cmap=plt.cm.Greens)
    ax2.set_title(f'Sklearn Logistic Regression (Acc: {lr_report["accuracy"]*100:.1f}%)')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(short_labels, rotation=45)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(short_labels)
    ax2.set_ylabel('True Intent')
    ax2.set_xlabel('Predicted Intent')

    plt.tight_layout()
    fig_path = PROJECT_ROOT / "reports" / "figures" / "intent_confusion_matrices.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Generated figure: {fig_path}")

    return output_payload


if __name__ == "__main__":
    run_intent_evaluation()
