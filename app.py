import streamlit as st
import time
import sys
import os

# Ensure the 'src' directory is in the path
sys.path.append(os.getcwd())

from src.baseline import summarize_tfidf, summarize_hybrid
from src.advanced import BERTExtractiveSummarizer, BARTSummarizer

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Text Summarization Assistant", page_icon="📝", layout="wide")

# --- CACHE THE MODELS ---
# Using cache_resource to load heavy models like BART once when the app starts
@st.cache_resource(show_spinner="Loading BART Model (Takes a few seconds)...")
def load_bart():
    return BARTSummarizer(model_name="sshleifer/distilbart-cnn-12-6")

@st.cache_resource(show_spinner="Loading Sentence-BERT Model...")
def load_bert_extractive():
    return BERTExtractiveSummarizer()

# --- HEADER SECTION ---
st.title("✨ AI Text Summarization System ✨")
st.markdown("Welcome! Paste any long article or text below, and our AI models will summarize it for you. Perfect for quickly understanding long news or research papers.")

# --- SIDEBAR: MODEL SETTINGS ---
st.sidebar.header("⚙️ Summarization Settings")
model_choice = st.sidebar.selectbox(
    "Select Model (Algorithm):",
    [
        "BART (Advanced Abstractive)",
        "Hybrid (TF-IDF + SBERT)",
        "BERT Extractive",
        "TF-IDF (Baseline Extract)"
    ]
)

if "BART" not in model_choice:
    ratio = st.sidebar.slider(
        "Extraction Ratio", 
        min_value=0.1, max_value=0.8, value=0.4, step=0.1, 
        help="How much of the text to keep? 0.4 means the summary will be roughly 40% of the original size."
    )
else:
    ratio = 0.4 # Placeholder for abstractive which relies on token lengths internally

st.sidebar.markdown("---")
st.sidebar.markdown("""
**🧠 About Our Models:**
- **BART:** Writes a completely new summary by understanding context (like ChatGPT).
- **Hybrid/BERT:** Smartly highlights and extracts the most important sentences.
- **TF-IDF:** A fast classic algorithm relying on word frequency importance.
""")

# --- MAIN INTERFACE: INPUT ---
text_input = st.text_area("📄 Paste your article here:", height=300, placeholder="Enter a very long text here...")

# --- MAIN INTERFACE: ACTION ---
col1, col2 = st.columns([1, 4])
with col1:
    summarize_btn = st.button("🚀 Summarize Text", type="primary", use_container_width=True)

if summarize_btn:
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to summarize!")
    else:
        with st.spinner(f"Generating summary using **{model_choice}**... Please wait."):
            start_time = time.time()
            summary_text = ""
            
            try:
                # Choose algorithm
                if model_choice == "TF-IDF (Baseline Extract)":
                    result = summarize_tfidf(text_input, ratio=ratio)
                    summary_text = result["summary"]
                    
                elif model_choice == "Hybrid (TF-IDF + SBERT)":
                    result = summarize_hybrid(text_input, ratio=ratio)
                    summary_text = result["summary"]
                    
                elif model_choice == "BERT Extractive":
                    extractive_model = load_bert_extractive()
                    result = extractive_model.summarize(text_input, ratio=ratio)
                    summary_text = result["summary"]
                    
                elif model_choice == "BART (Advanced Abstractive)":
                    bart_model = load_bart()
                    result = bart_model.summarize(text_input)
                    summary_text = result["summary"]
                    
                end_time = time.time()
                
                # Check if output is empty
                if not summary_text.strip():
                    st.error("The model couldn't generate a summary. The text might be too short.")
                else:
                    # Output the result
                    st.success(f"✅ Summarization Complete! (Took {end_time - start_time:.2f} seconds)")
                    st.subheader("📋 Your Summary:")
                    st.info(summary_text)
                    
                    # Show performance stats
                    orig_len = len(text_input.split())
                    sum_len = len(summary_text.split())
                    if orig_len > 0:
                        st.caption(f"**Original Length:** {orig_len} words | **Summary Length:** {sum_len} words | **Compression:** {(sum_len/orig_len*100):.1f}%")

            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")
