"""
------------------------------------------------------------
 Phase 4: Advanced Transformer-Based Summarization
------------------------------------------------------------
 Methods implemented:
   1. BART Abstractive Summarizer  (sshleifer/distilbart-cnn-12-6 - Optimized)
   2. T5 Abstractive Summarizer    (t5-small - lightweight fallback)
   3. BERT Extractive Summarizer   (Sentence-BERT + TextRank)
------------------------------------------------------------
"""

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.preprocessing import preprocess_article


# ------------------------------------------------------------
#  1. BART Abstractive Summarizer (Primary Advanced Method)
# ------------------------------------------------------------

class BARTSummarizer:
    """
    Abstractive summarization using BART.
    Uses Direct Model Loading instead of Pipeline for better compatibility.
    """

    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """
        Initialize the BART model and tokenizer.
        Defaulting to distilbart for faster performance in university projects.
        """
        print(f"Loading BART model: {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("BART model loaded.")

    def summarize(
        self,
        article_text: str,
        max_length: int = None,
        min_length: int = None,
        num_beams: int = 4,
    ) -> dict:
        """Generate an abstractive summary using BART."""
        if not article_text or article_text.strip() == "":
            return {"summary": "", "model": self.model_name, "params": {}}
            
        # Dynamic length calculation to ensure ~15-35% compression ratio
        # Only apply dynamic calc if min/max length were not explicitly provided
        word_count = len(article_text.split())
        
        if max_length is None:
            max_length = max(30, min(250, int(word_count * 0.35)))
        
        if min_length is None:
            min_length = max(10, int(word_count * 0.15))

        # Ensure max_length > min_length
        if max_length <= min_length:
            max_length = min_length + 10

        # Prepare input
        inputs = self.tokenizer(
            article_text, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        # Generate
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            early_stopping=True,
        )

        # Decode
        summary = self.tokenizer.decode(
            summary_ids[0], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )

        return {
            "summary": summary,
            "model": self.model_name,
            "params": {
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": num_beams,
            },
        }


# ------------------------------------------------------------
#  2. T5 Abstractive Summarizer (Lightweight Alternative)
# ------------------------------------------------------------

class T5Summarizer:
    """
    Abstractive summarization using T5-small.
    Uses Direct Model Loading.
    """

    def __init__(self, model_name: str = "t5-small"):
        print(f"Loading T5 model: {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("T5 model loaded.")

    def summarize(
        self,
        article_text: str,
        max_length: int = 150,
        min_length: int = 30,
    ) -> dict:
        """Generate an abstractive summary using T5."""
        if not article_text or article_text.strip() == "":
            return {"summary": "", "model": self.model_name, "params": {}}

        input_text = "summarize: " + article_text
        inputs = self.tokenizer(
            input_text, 
            max_length=512, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
        )

        summary = self.tokenizer.decode(
            summary_ids[0], 
            skip_special_tokens=True
        )

        return {
            "summary": summary,
            "model": self.model_name,
            "params": {"max_length": max_length, "min_length": min_length},
        }


# ------------------------------------------------------------
#  3. BERT Extractive Summarizer (TextRank on Sentence Embeddings)
# ------------------------------------------------------------

class BERTExtractiveSummarizer:
    """
    Extractive summarization using Sentence-BERT embeddings
    with a TextRank-inspired graph algorithm.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading Sentence-BERT for extractive: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Sentence-BERT loaded.")

    def summarize(
        self,
        article_text: str,
        ratio: float = 0.3,
        num_iterations: int = 50,
        damping: float = 0.85,
    ) -> dict:
        """Generate an extractive summary using TextRank on BERT embeddings."""
        data = preprocess_article(article_text)
        original = data["original_sentences"]

        if not original:
            return {"summary": "", "key_sentences": [], "scores": np.array([])}

        # Step 1: Encode all sentences
        embeddings = self.model.encode(original, show_progress_bar=False)

        # Step 2: Build similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)

        # Step 3: Normalize
        row_sums = sim_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm_matrix = sim_matrix / row_sums

        # Step 4: TextRank
        n = len(original)
        scores = np.ones(n) / n
        for _ in range(num_iterations):
            new_scores = (1 - damping) / n + damping * norm_matrix.T @ scores
            if np.abs(new_scores - scores).sum() < 1e-6:
                break
            scores = new_scores

        # Step 5: Select
        n_select = max(1, int(n * ratio))
        top_indices = np.argsort(scores)[::-1][:n_select]
        top_indices_sorted = sorted(top_indices)

        key_sentences = [original[i] for i in top_indices_sorted]
        summary = " ".join(key_sentences)

        return {
            "summary": summary,
            "key_sentences": key_sentences,
            "scores": scores,
        }
