"""
------------------------------------------------------------
 Phase 1: Text Preprocessing Pipeline
------------------------------------------------------------
 Handles all text cleaning and preparation steps:
   - Sentence segmentation (preserving originals)
   - Lowercasing
   - Punctuation removal
   - Tokenization
   - Stopword removal
   - Lemmatization
------------------------------------------------------------
"""

import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Download required NLTK data (one-time) ──────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Module-level constants ──────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
# Extra stopwords common in news articles that add no summarization value
EXTRA_STOP = {"said", "also", "would", "could", "one", "two", "new", "like", "may"}
STOP_WORDS = STOP_WORDS.union(EXTRA_STOP)

LEMMATIZER = WordNetLemmatizer()


# ------------------------------------------------------------
#  Core Functions
# ------------------------------------------------------------

def segment_sentences(text: str) -> list[str]:
    """
    Split raw text into a list of sentences.
    Returns the ORIGINAL (uncleaned) sentences so they can be used
    as-is in the final extractive summary output.
    """
    sentences = sent_tokenize(text)
    # Filter out very short fragments (< 5 words) that are likely noise
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
    return sentences


def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove punctuation while preserving spaces and word characters."""
    # Keep apostrophes within contractions (e.g., don't, it's)
    text = re.sub(r"[^\w\s']", " ", text)
    # Remove standalone apostrophes
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize text into a list of word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove English stopwords from token list."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize tokens to their root form (e.g., 'running' → 'run')."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


# ------------------------------------------------------------
#  Full Pipeline
# ------------------------------------------------------------

def clean_sentence(sentence: str) -> str:
    """
    Apply the full cleaning pipeline to a SINGLE sentence and return
    the cleaned text as a string (for TF-IDF vectorizer input).

    Pipeline: lowercase → remove punctuation → tokenize →
              remove stopwords → lemmatize → rejoin.
    """
    text = lowercase(sentence)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_article(article_text: str) -> dict:
    """
    Full preprocessing pipeline for one article.

    Returns a dictionary:
    {
        "original_sentences": [...],  # list of raw sentences (for display)
        "cleaned_sentences":  [...],  # list of cleaned sentence strings (for TF-IDF)
        "cleaned_tokens":     [...],  # list of token-lists per sentence (for Word2Vec)
    }
    """
    original_sentences = segment_sentences(article_text)

    cleaned_sentences = []
    cleaned_tokens = []

    for sent in original_sentences:
        text = lowercase(sent)
        text = remove_punctuation(text)
        tokens = tokenize(text)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize(tokens)

        cleaned_sentences.append(" ".join(tokens))
        cleaned_tokens.append(tokens)

    return {
        "original_sentences": original_sentences,
        "cleaned_sentences": cleaned_sentences,
        "cleaned_tokens": cleaned_tokens,
    }


# ------------------------------------------------------------
#  Keyword Extraction Helper
# ------------------------------------------------------------

def extract_keywords_from_tokens(all_tokens: list[list[str]], top_n: int = 20) -> list[str]:
    """
    Extract the most frequent meaningful keywords from token lists.
    Used for the 'Important Keywords' output requirement.
    """
    from collections import Counter
    flat = [t for tokens in all_tokens for t in tokens]
    counts = Counter(flat)
    return [word for word, _ in counts.most_common(top_n)]


# ------------------------------------------------------------
#  Quick Test
# ------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "The food was amazing and the service was excellent. "
        "The restaurant was clean and the staff were friendly. "
        "However, the delivery was very slow and the order arrived late. "
        "Overall, it was a good experience despite the delivery issue."
    )

    result = preprocess_article(sample)

    print("=" * 60)
    print("PREPROCESSING PIPELINE TEST")
    print("=" * 60)
    for i, (orig, clean) in enumerate(
        zip(result["original_sentences"], result["cleaned_sentences"])
    ):
        print(f"\n--- Sentence {i + 1} ---")
        print(f"  Original : {orig}")
        print(f"  Cleaned  : {clean}")

    keywords = extract_keywords_from_tokens(result["cleaned_tokens"])
    print(f"\n🔑 Top Keywords: {keywords}")
