# Text Summarization System 🚀
**Faculty of Computing & Artificial Intelligence | NLP Course Project**

A professional-grade text summarization system that implements multiple methodologies to condense long articles into concise summaries while preserving core meaning.

---

## 🌟 Features
- **Multi-Method Approach**: Covers Baseline (TF-IDF), Hybrid (SBERT + ML), and Advanced (BART Abstractive) summarization.
- **Academic UI**: A full-featured Streamlit dashboard for real-time interactive summarization.
- **Accuracy Metrics**: Automated ROUGE evaluation against human-annotated references.
- **NLP Pipeline**: Complete text cleaning (Lemmatization, Stopwords, etc.) using NLTK and SpaCy.

---

## 📂 Project Structure
- `app.py`: The Main Graphical User Interface (GUI).
- `main.py`: CLI Entry point for running accuracy tests on the entire dataset.
- `src/`: 
  - `preprocessing.py`: Text cleaning and normalization.
  - `baseline.py`: TF-IDF and Hybrid extraction logic.
  - `advanced.py`: Transformer-based summarization (BART/BERT).
  - `evaluate.py`: ROUGE and compression calculation.
- `data/`: Contains `test_samples.csv` (Sample dataset from CNN/DailyMail).
- `report/`: Generated artifacts and project documentation.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tal3at10/Text-Summarization-System.git
   cd Text-Summarization-System
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### 1. Interactive User Interface (Recommended for Demo)
To launch the professional web-based dashboard:
```bash
python -m streamlit run app.py
```
*It will automatically open in your browser at `http://localhost:8501`.*

### 2. Accuracy Evaluation Test
To run the automated test on all 10 dataset samples and generate ROUGE scores:
```bash
python main.py
```

---

## 📚 Project Resources
For team members preparing for the project defense, please refer to:
- **[Project Discussion Guide](Project_Discussion_Guide.md)**: Contains expected exam/defense questions and detailed technical explanations in both Arabic & English.
- **[Evaluation Report](report/report_template.md)**: Final results and model comparison.

---
**Developed for the Spring 2026 NLP Course.**
