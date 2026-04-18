"""
------------------------------------------------------------
 Text Summarization System (Accuracy Test Process)
------------------------------------------------------------
 This script runs a real test using samples from the CNN/DailyMail dataset
 to calculate ROUGE accuracy scores and displays the text for each method.
------------------------------------------------------------
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our custom modules
from src.preprocessing import preprocess_article
from src.baseline import summarize_tfidf, summarize_hybrid
from src.advanced import T5Summarizer, BERTExtractiveSummarizer, BARTSummarizer
from src.evaluate import evaluate_summary

# -- Dataset Loading --------------------------------------------------------
DATA_FILE = "data/test_samples.csv"
OUTPUT_DIR = "outputs"
RATIO = 0.4  

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/reports", exist_ok=True)

def load_data():
    """Load the test dataset from CSV."""
    if not os.path.exists(DATA_FILE):
        print(f"Error: Dataset not found at {DATA_FILE}")
        return []
    df = pd.read_csv(DATA_FILE)
    return df.to_dict('records')

def run_accuracy_test():
    print("Starting REAL TEST & PROCESS DEMO (Accuracy Evaluation)...")

    # 1. Initialize Models
    print("Initializing Models (Upgraded to BART Distil for HIGHER Accuracy)...")
    print("Please wait, loading models might take a moment if downloading for the first time...")
    try:
        # We upgraded to distilbart for much better abstractive accuracy than T5-small
        advanced_model = BARTSummarizer(model_name="sshleifer/distilbart-cnn-12-6")
        bert_ext_model = BERTExtractiveSummarizer()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 2. Process Articles
    comparison_data = []
    test_entries = load_data()
    
    if not test_entries:
        print("No data to process. System exiting.")
        return

    for i, entry in enumerate(test_entries):
        article = entry["article"]
        ref = entry["highlights"]
        art_id = entry.get("id", f"sample_{i+1}")

        print(f"\n{'='*70}")
        print(f"Processing Sample {i+1}...")
        print(f"{'='*70}")
        
        print(f"\n[1] ORIGINAL TEXT:\n{article}")
        print(f"\n[2] HUMAN HIGHLIGHTS (Target):\n{ref}")
        print(f"\n{'-'*70}")
        print(" GENERATED SUMMARIES BY OUR METHODS:")
        print(f"{'-'*70}")

        # --- Baseline: TF-IDF ---
        res_tfidf = summarize_tfidf(article, ratio=RATIO)
        print(f"TF-IDF (Baseline) :\n{res_tfidf['summary']}")
        scores_tfidf = evaluate_summary(article, ref, res_tfidf["summary"])
        scores_tfidf.update({"method": "TF-IDF (Baseline)", "article_id": art_id})
        comparison_data.append(scores_tfidf)

        # --- Hybrid: TF-IDF + Embeddings ---
        res_hybrid = summarize_hybrid(article, ratio=RATIO, alpha=0.5)
        print(f"\nHybrid (TF-IDF + SBERT) :\n{res_hybrid['summary']}")
        scores_hybrid = evaluate_summary(article, ref, res_hybrid["summary"])
        scores_hybrid.update({"method": "Hybrid (SBERT + TF-IDF)", "article_id": art_id})
        comparison_data.append(scores_hybrid)

        # --- BERT Extractive ---
        res_bert_ext = bert_ext_model.summarize(article, ratio=RATIO)
        print(f"\nBERT Extractive :\n{res_bert_ext['summary']}")
        scores_bert_ext = evaluate_summary(article, ref, res_bert_ext["summary"])
        scores_bert_ext.update({"method": "BERT Extractive", "article_id": art_id})
        comparison_data.append(scores_bert_ext)

        # --- BART Abstractive ---
        res_bart = advanced_model.summarize(article)
        print(f"\nBART Abstractive :\n{res_bart['summary']}")
        scores_bart = evaluate_summary(article, ref, res_bart["summary"])
        scores_bart.update({"method": "BART Abstractive", "article_id": art_id})
        comparison_data.append(scores_bart)

    # 3. Generate Table
    df_all = pd.DataFrame(comparison_data)
    metrics = ["rouge1_f", "rouge2_f", "rougeL_f", "compression_ratio"]
    comparison_table = df_all.groupby("method")[metrics].mean().round(4).reset_index()
    
    print("\n" + "=" * 70)
    print("COMPOSITE COMPARISON TABLE (ACCURACY RESULTS):")
    print("=" * 70)
    print(comparison_table.to_string(index=False))
    print("=" * 70)
    print("\nNote: Accuracy is measured by ROUGE overlap with Human Highlights.")

if __name__ == "__main__":
    run_accuracy_test()
