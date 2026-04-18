"""
------------------------------------------------------------
 Phase 2 & 3: Feature Extraction + Baseline Extractive Summarizer
------------------------------------------------------------
 Methods implemented:
   1. TF-IDF Extractive  (pure TF-IDF sentence scoring)
   2. Embedding-based     (Sentence-BERT centroid similarity)
   3. Hybrid              (α·TF-IDF + (1-α)·Embedding — best approach)
   4. Keyword extraction  (from TF-IDF weights)
   5. Main Idea detection (via K-Means clustering of embeddings)
------------------------------------------------------------
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from src.preprocessing import preprocess_article, clean_sentence

# ── Load Sentence-BERT model (compact but powerful) ─────────────────────────
# This is loaded once and reused across all function calls.
print("Loading Sentence-BERT model (first time may download ~90MB)...")
SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Sentence-BERT model loaded.")


# ------------------------------------------------------------
#  1. TF-IDF Sentence Scoring
# ------------------------------------------------------------

def compute_tfidf_scores(cleaned_sentences: list[str]) -> np.ndarray:
    """
    Compute a TF-IDF importance score for each sentence.

    Strategy:
        - Fit a TfidfVectorizer on all sentences within one article.
        - Score each sentence = mean of its non-zero TF-IDF values.
        - Higher score ⇒ sentence contains rarer, more informative terms.

    Returns: 1-D array of shape (n_sentences,) with scores in [0, 1].
    """
    if not cleaned_sentences or all(s.strip() == "" for s in cleaned_sentences):
        return np.zeros(len(cleaned_sentences))

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=1,
        max_df=0.85,
        sublinear_tf=True,      # log-scaled TF
    )

    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)  # sparse matrix
    # Mean of non-zero TF-IDF values per sentence
    scores = np.array(tfidf_matrix.mean(axis=1)).flatten()
    return scores


def get_tfidf_keywords(cleaned_sentences: list[str], top_n: int = 20) -> list[str]:
    """
    Extract the top-N keywords from an article based on TF-IDF weights.
    Fulfills the 'Important Keywords' output requirement.
    """
    if not cleaned_sentences or all(s.strip() == "" for s in cleaned_sentences):
        return []

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
    feature_names = vectorizer.get_feature_names_out()

    # Sum TF-IDF across all sentences for each term
    term_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    top_indices = term_scores.argsort()[::-1][:top_n]

    return [feature_names[i] for i in top_indices]


# ------------------------------------------------------------
#  2. Sentence-BERT Embedding Scoring (Centroid-Based)
# ------------------------------------------------------------

def compute_embedding_scores(original_sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute semantic importance score using Sentence-BERT.

    Strategy (Centroid-Based):
        1. Encode every sentence → 384-dim vector.
        2. Compute document centroid = mean of all sentence vectors.
        3. Score each sentence = cosine_similarity(sentence, centroid).
        4. Sentences closest to centroid = most representative of the document.

    Returns:
        - scores: 1-D array of cosine similarities (n_sentences,)
        - embeddings: 2-D array (n_sentences, 384) for reuse
    """
    if not original_sentences:
        return np.array([]), np.array([])

    embeddings = SBERT_MODEL.encode(original_sentences, show_progress_bar=False)
    centroid = embeddings.mean(axis=0, keepdims=True)
    scores = cosine_similarity(embeddings, centroid).flatten()

    return scores, embeddings


# ------------------------------------------------------------
#  3. Hybrid Scoring (TF-IDF + Embeddings)
# ------------------------------------------------------------

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1] range."""
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def compute_hybrid_scores(
    tfidf_scores: np.ndarray,
    embedding_scores: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Combine TF-IDF and embedding scores using a weighted sum.

    Formula:
        final_score = α × norm(tfidf) + (1 - α) × norm(embedding)

    This directly fulfills the project requirement:
        "Important sentences are selected based on their score (using TF-IDF)
         and their meaning (using embeddings)."
    """
    tfidf_norm = normalize_scores(tfidf_scores)
    embed_norm = normalize_scores(embedding_scores)

    return alpha * tfidf_norm + (1 - alpha) * embed_norm


# ------------------------------------------------------------
#  4. MMR — Maximal Marginal Relevance (Reduces Redundancy)
# ------------------------------------------------------------

def mmr_selection(
    scores: np.ndarray,
    embeddings: np.ndarray,
    n_select: int,
    lambda_param: float = 0.7,
) -> list[int]:
    """
    Select sentences using Maximal Marginal Relevance (MMR).

    MMR balances relevance (high score) with diversity (dissimilarity to
    already-selected sentences), preventing redundant sentences in summaries.

    Formula:
        MMR = λ × score(s) - (1 - λ) × max_sim(s, selected)

    Args:
        scores:       1-D array of sentence importance scores.
        embeddings:   2-D array of sentence embeddings (n_sentences, dim).
        n_select:     Number of sentences to select for the summary.
        lambda_param: Trade-off (0.7 = favor relevance, 0.3 = favor diversity).

    Returns:
        List of selected sentence indices (in selection order).
    """
    n = len(scores)
    if n <= n_select:
        return list(range(n))

    selected = []
    remaining = list(range(n))

    # Start with the highest-scoring sentence
    first = int(np.argmax(scores))
    selected.append(first)
    remaining.remove(first)

    for _ in range(n_select - 1):
        best_idx = None
        best_mmr = -np.inf

        for idx in remaining:
            relevance = scores[idx]

            # Max similarity to any already-selected sentence
            if embeddings is not None and len(embeddings) > 0:
                sims = cosine_similarity(
                    embeddings[idx].reshape(1, -1),
                    embeddings[selected],
                ).flatten()
                max_similarity = sims.max()
            else:
                max_similarity = 0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


# ------------------------------------------------------------
#  5. Main Ideas via Clustering
# ------------------------------------------------------------

def extract_main_ideas(
    original_sentences: list[str],
    embeddings: np.ndarray,
    n_ideas: int = 3,
) -> list[str]:
    """
    Identify main ideas by clustering sentence embeddings.

    Strategy:
        1. Cluster all sentence embeddings into k clusters using K-Means.
        2. For each cluster, find the sentence nearest to the cluster centroid.
        3. That sentence represents the 'main idea' of its cluster.

    Returns: List of main-idea sentences.
    """
    n_sentences = len(original_sentences)
    if n_sentences <= n_ideas:
        return original_sentences

    k = min(n_ideas, n_sentences)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    main_ideas = []
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)

        sims = cosine_similarity(cluster_embeddings, centroid).flatten()
        best_local_idx = sims.argmax()
        best_global_idx = cluster_indices[best_local_idx]

        main_ideas.append(original_sentences[best_global_idx])

    return main_ideas


# -----------------------------------------------------------------------------
#  6. High-Level Summarizer Functions
# -----------------------------------------------------------------------------

def summarize_tfidf(article_text: str, ratio: float = 0.3) -> dict:
    """
    Baseline extractive summarization using ONLY TF-IDF scores.

    Returns:
        {
            "summary": str,
            "key_sentences": list[str],
            "keywords": list[str],
            "scores": np.ndarray,
        }
    """
    data = preprocess_article(article_text)
    original = data["original_sentences"]
    cleaned = data["cleaned_sentences"]

    if not original:
        return {"summary": "", "key_sentences": [], "keywords": [], "scores": np.array([])}

    tfidf_scores = compute_tfidf_scores(cleaned)
    n_select = max(1, int(len(original) * ratio))
    top_indices = np.argsort(tfidf_scores)[::-1][:n_select]

    # Sort by original order for coherent reading
    top_indices_sorted = sorted(top_indices)
    key_sentences = [original[i] for i in top_indices_sorted]
    summary = " ".join(key_sentences)
    keywords = get_tfidf_keywords(cleaned)

    return {
        "summary": summary,
        "key_sentences": key_sentences,
        "keywords": keywords,
        "scores": tfidf_scores,
    }


def summarize_hybrid(article_text: str, ratio: float = 0.3, alpha: float = 0.5) -> dict:
    """
    Advanced extractive summarization combining TF-IDF + Sentence-BERT
    with MMR for diversity.

    This is the strongest extractive method — directly fulfills the project
    requirement of using "TF-IDF scores AND embeddings".

    Returns:
        {
            "summary": str,
            "key_sentences": list[str],
            "main_ideas": list[str],
            "keywords": list[str],
            "scores": np.ndarray,
        }
    """
    data = preprocess_article(article_text)
    original = data["original_sentences"]
    cleaned = data["cleaned_sentences"]

    if not original:
        return {
            "summary": "", "key_sentences": [], "main_ideas": [],
            "keywords": [], "scores": np.array([])
        }

    # Compute both scoring methods
    tfidf_scores = compute_tfidf_scores(cleaned)
    embedding_scores, embeddings = compute_embedding_scores(original)

    # Combine into hybrid score
    hybrid_scores = compute_hybrid_scores(tfidf_scores, embedding_scores, alpha)

    # Select sentences using MMR (relevance + diversity)
    n_select = max(1, int(len(original) * ratio))
    selected_indices = mmr_selection(hybrid_scores, embeddings, n_select, lambda_param=0.7)

    # Sort by original order for coherent reading
    selected_indices_sorted = sorted(selected_indices)
    key_sentences = [original[i] for i in selected_indices_sorted]
    summary = " ".join(key_sentences)

    # Extract main ideas via clustering
    main_ideas = extract_main_ideas(original, embeddings, n_ideas=3)

    # Extract keywords
    keywords = get_tfidf_keywords(cleaned)

    return {
        "summary": summary,
        "key_sentences": key_sentences,
        "main_ideas": main_ideas,
        "keywords": keywords,
        "scores": hybrid_scores,
    }


# ------------------------------------------------------------
#  Quick Test
# ------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "The food was amazing and the service was excellent. "
        "The restaurant was clean and the staff were friendly. "
        "However, the delivery was very slow and the order arrived late. "
        "Overall, it was a good experience despite the delivery issue. "
        "The menu had a wide variety of options for all dietary needs. "
        "Prices were reasonable compared to other restaurants in the area."
    )

    print("\n" + "=" * 70)
    print("BASELINE — TF-IDF EXTRACTIVE")
    print("=" * 70)
    result_tfidf = summarize_tfidf(sample, ratio=0.4)
    print(f"  Summary  : {result_tfidf['summary']}")
    print(f"  Keywords : {result_tfidf['keywords'][:10]}")

    print("\n" + "=" * 70)
    print("HYBRID — TF-IDF + SENTENCE-BERT + MMR")
    print("=" * 70)
    result_hybrid = summarize_hybrid(sample, ratio=0.4, alpha=0.5)
    print(f"  Summary    : {result_hybrid['summary']}")
    print(f"  Main Ideas : {result_hybrid['main_ideas']}")
    print(f"  Keywords   : {result_hybrid['keywords'][:10]}")
