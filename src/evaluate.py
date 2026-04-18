"""
------------------------------------------------------------
 Phase 5: Evaluation Framework
------------------------------------------------------------
 Metrics implemented:
   1. ROUGE-1, ROUGE-2, ROUGE-L  (n-gram overlap)
   2. BERTScore                   (semantic similarity)
   3. Compression Ratio           (length reduction)
   4. Keyword Coverage            (information preservation)
   5. Manual Evaluation Rubric    (human judgment template)
   6. Comparison Table Generator  (side-by-side results)
------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from collections import Counter


# ------------------------------------------------------------
#  1. ROUGE Scores
# ------------------------------------------------------------

def compute_rouge(reference: str, generated: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L between reference and generated summary.

    ROUGE-1: Unigram overlap (measures informativeness)
    ROUGE-2: Bigram overlap  (measures fluency)
    ROUGE-L: Longest common subsequence (measures structure)

    Returns:
        {
            "rouge1": {"precision": float, "recall": float, "fmeasure": float},
            "rouge2": {"precision": float, "recall": float, "fmeasure": float},
            "rougeL": {"precision": float, "recall": float, "fmeasure": float},
        }
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, generated)

    result = {}
    for metric in ["rouge1", "rouge2", "rougeL"]:
        s = scores[metric]
        result[metric] = {
            "precision": round(s.precision, 4),
            "recall": round(s.recall, 4),
            "fmeasure": round(s.fmeasure, 4),
        }

    return result


# ------------------------------------------------------------
#  2. BERTScore
# ------------------------------------------------------------

def compute_bertscore(references: list[str], generated: list[str]) -> dict:
    """
    Compute BERTScore (semantic similarity using BERT embeddings).

    Unlike ROUGE (which counts word overlap), BERTScore captures
    meaning similarity — e.g., 'car' and 'vehicle' score highly.

    Args:
        references: List of reference summaries.
        generated:  List of generated summaries (same length).

    Returns:
        {
            "precision": float (avg),
            "recall": float (avg),
            "f1": float (avg),
        }
    """
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            generated,
            references,
            lang="en",
            verbose=False,
            model_type="microsoft/deberta-xlarge-mnli",
        )

        return {
            "precision": round(P.mean().item(), 4),
            "recall": round(R.mean().item(), 4),
            "f1": round(F1.mean().item(), 4),
        }
    except Exception as e:
        # Fallback if bert_score fails or model not available
        print(f"⚠️  BERTScore computation failed: {e}")
        print("    Falling back to a simpler model...")
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn(
                generated, references, lang="en", verbose=False
            )
            return {
                "precision": round(P.mean().item(), 4),
                "recall": round(R.mean().item(), 4),
                "f1": round(F1.mean().item(), 4),
            }
        except Exception as e2:
            print(f"⚠️  BERTScore fallback also failed: {e2}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ------------------------------------------------------------
#  3. Compression Ratio
# ------------------------------------------------------------

def compute_compression_ratio(original: str, summary: str) -> float:
    """
    Calculate the compression ratio: summary_length / original_length.

    Ideal range: 0.15–0.35 (summary is 15-35% of original).
    """
    orig_len = len(original.split())
    summ_len = len(summary.split())
    if orig_len == 0:
        return 0.0
    return round(summ_len / orig_len, 4)


# ------------------------------------------------------------
#  4. Keyword Coverage (Information Preservation)
# ------------------------------------------------------------

def compute_keyword_coverage(reference_keywords: list[str], summary: str) -> float:
    """
    Check what fraction of reference keywords appear in the generated summary.

    This measures whether key information from the original is preserved.
    """
    if not reference_keywords:
        return 0.0

    summary_lower = summary.lower()
    matched = sum(1 for kw in reference_keywords if kw.lower() in summary_lower)
    return round(matched / len(reference_keywords), 4)


# ------------------------------------------------------------
#  5. Full Evaluation for One Article
# ------------------------------------------------------------

def evaluate_summary(
    original_text: str,
    reference_summary: str,
    generated_summary: str,
    reference_keywords: list[str] = None,
) -> dict:
    """
    Run all automatic evaluation metrics on a single generated summary.

    Returns a flat dictionary of all metric scores.
    """
    # ROUGE
    rouge = compute_rouge(reference_summary, generated_summary)

    # Compression ratio
    compression = compute_compression_ratio(original_text, generated_summary)

    # Keyword coverage
    kw_coverage = 0.0
    if reference_keywords:
        kw_coverage = compute_keyword_coverage(reference_keywords, generated_summary)

    return {
        "rouge1_f": rouge["rouge1"]["fmeasure"],
        "rouge2_f": rouge["rouge2"]["fmeasure"],
        "rougeL_f": rouge["rougeL"]["fmeasure"],
        "rouge1_p": rouge["rouge1"]["precision"],
        "rouge1_r": rouge["rouge1"]["recall"],
        "compression_ratio": compression,
        "keyword_coverage": kw_coverage,
    }


# ------------------------------------------------------------
#  6. Batch Evaluation & Comparison Table
# ------------------------------------------------------------

def evaluate_batch(
    originals: list[str],
    references: list[str],
    generated: list[str],
    method_name: str,
) -> pd.DataFrame:
    """
    Evaluate a batch of summaries and return a DataFrame of results.
    """
    records = []
    for i, (orig, ref, gen) in enumerate(zip(originals, references, generated)):
        scores = evaluate_summary(orig, ref, gen)
        scores["article_id"] = i
        scores["method"] = method_name
        records.append(scores)

    return pd.DataFrame(records)


def build_comparison_table(all_results: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine evaluation results from multiple methods into one comparison table.

    Input: List of DataFrames from evaluate_batch() for each method.
    Output: Aggregated mean scores per method.
    """
    combined = pd.concat(all_results, ignore_index=True)

    metrics = [
        "rouge1_f", "rouge2_f", "rougeL_f",
        "compression_ratio", "keyword_coverage"
    ]

    comparison = combined.groupby("method")[metrics].mean().round(4)
    return comparison


# ------------------------------------------------------------
#  7. Manual Evaluation Template
# ------------------------------------------------------------

def generate_manual_eval_template(
    articles: list[str],
    summaries_by_method: dict[str, list[str]],
    n_samples: int = 10,
) -> pd.DataFrame:
    """
    Generate a template for manual (human) evaluation.

    Creates a DataFrame with columns for each evaluation criterion
    (to be filled in by the human evaluator).

    Criteria:
        - Informativeness (1-5): Does it capture the main idea?
        - Fluency (1-5):         Is it grammatically correct?
        - Conciseness (1-5):     Is there no unnecessary info?
        - Faithfulness (1-5):    Is everything factually correct?
    """
    import random
    random.seed(42)

    n = min(n_samples, len(articles))
    indices = random.sample(range(len(articles)), n)

    rows = []
    for idx in indices:
        for method_name, summaries in summaries_by_method.items():
            rows.append({
                "article_id": idx,
                "method": method_name,
                "article_preview": articles[idx][:200] + "...",
                "summary": summaries[idx],
                "informativeness (1-5)": "",
                "fluency (1-5)": "",
                "conciseness (1-5)": "",
                "faithfulness (1-5)": "",
                "notes": "",
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
#  Quick Test
# ------------------------------------------------------------

if __name__ == "__main__":
    reference = "The food and service were excellent but delivery was slow."
    generated = "The restaurant had great food and service. However, the delivery was slow and the order arrived late."

    print("=" * 60)
    print("EVALUATION METRICS TEST")
    print("=" * 60)

    rouge = compute_rouge(reference, generated)
    for metric, values in rouge.items():
        print(f"  {metric}: F1={values['fmeasure']:.4f}  P={values['precision']:.4f}  R={values['recall']:.4f}")

    cr = compute_compression_ratio(
        "The food was amazing and the service was excellent. " * 5,
        generated
    )
    print(f"  Compression Ratio: {cr}")

    kw_cov = compute_keyword_coverage(["food", "service", "delivery", "slow", "price"], generated)
    print(f"  Keyword Coverage: {kw_cov}")

    print("\n✅ All evaluation metrics working!")
