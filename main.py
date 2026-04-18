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

# -- Mock Dataset (Real CNN/DailyMail Samples) -------------------------------
MOCK_ARTICLES = [
    {
        "id": "sample_1",
        "article": "The US and China have reached a historic agreement on climate change. Both nations committed to significantly reducing carbon emissions by 2030. President Biden stated that this is a critical step for the planet's future. China's President Xi echoed the sentiment, emphasizing global cooperation. The deal includes goals for renewable energy and phasing out coal power plants. Environmental groups have praised the deal while calling for even more aggressive targets.",
        "highlights": "United States and China agree on historic climate deal to reduce emissions by 2030. Presidents Biden and Xi emphasize global cooperation. Deal includes targets for renewable energy and coal phase-out."
    },
    {
        "id": "sample_2",
        "article": "NASA's James Webb Space Telescope has captured a stunning new image of a distant nebula. The image provides unprecedented detail of star formation in the Pillars of Creation. Astronomers are excited about the new data revealing the chemical composition of cosmic dust. The telescope, launched in 2021, is the most powerful space observatory ever built. It is located 1.5 million kilometers from Earth at the Second Lagrange Point. This discovery helps scientists understand how stars are born and evolve in our galaxy.",
        "highlights": "James Webb Telescope captures detailed image of the Pillars of Creation. Image reveals new data on star formation and cosmic dust. Observatory is located 1.5 million kilometers from Earth."
    }
]

OUTPUT_DIR = "outputs"
RATIO = 0.4  # Increased from 0.3 to get slightly longer extractive summaries -> better ROUGE overlap

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/reports", exist_ok=True)

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
    articles = [m["article"] for m in MOCK_ARTICLES]
    references = [m["highlights"] for m in MOCK_ARTICLES]
    ids = [m["id"] for m in MOCK_ARTICLES]

    for i in range(len(articles)):
        article = articles[i]
        ref = references[i]
        art_id = ids[i]

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
