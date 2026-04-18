# Text Summarization System

A comprehensive NLP project to summarize long texts using classical (TF-IDF) and advanced (Transformer-based) methods.

## Features
- **Preprocessing**: Lowercasing, punctuation removal, stopword removal, tokenization, and lemmatization.
- **Feature Extraction**: TF-IDF and Sentence-BERT embeddings.
- **Extractive Summarization**: Baseline method using TF-IDF and semantic centroids.
- **Abstractive Summarization**: Advanced method using BART/T5 models.
- **Evaluation**: ROUGE, BERTScore, and manual quality metrics.

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Structure
- `src/`: Source code modules.
- `data/`: Dataset storage.
- `outputs/`: Generated summaries and evaluation results.
