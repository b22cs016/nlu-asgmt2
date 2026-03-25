"""
DATASET PREPARATION FOR WORD2VEC (IIT JODHPUR DATA)

This script:
1. Scrapes textual data from IIT Jodhpur website
2. Extracts text from PDF (academic regulations, annual reports, newsletters)
3. Cleans and preprocesses text
4. Tokenizes and builds corpus
5. Computes dataset statistics
6. Generates a word cloud

NOTE:
-functions are  modular to improve readability and originality

"""

import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from pdfminer.high_level import extract_text
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('punkt_tab')
# -------------------------------
# STEP 1: SCRAPE WEBSITE TEXT
# -------------------------------

def save_corpus_to_file(corpus, filename="cleaned_corpus.txt"):
    # Open a file in write mode
    with open(filename, 'w', encoding='utf-8') as f:
        for document in corpus:
            # Join the tokens back into a string and write to file
            f.write(" ".join(document) + "\n")
    print(f"Corpus saved to {filename}")


def scrape_website(url):
    """
    Fetches and extracts visible text from a webpage.

    Note:
    Websites contain HTML tags, scripts, and styles.
    We remove them to keep only meaningful text.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts and styles (boilerplate)
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()

        # Extract clean text
        text = soup.get_text(separator=' ')

        return text

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


# -------------------------------
# STEP 2: EXTRACT PDF TEXT
# -------------------------------

def extract_pdf_text(pdf_path):
    """
    Extracts raw text from PDF using pdfminer.

    Why pdfminer?:
    Academic regulations are usually in PDF format,
    so we need a parser instead of scraping.
    """
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


# -------------------------------
# STEP 3: CLEAN TEXT
# -------------------------------

def clean_text(text):
    """
    Cleans raw text by:
    - Lowercasing
    - Removing special characters
    - Removing extra whitespace

    Note:
    Word2Vec works better on normalized text. 
    Normalization helps reduce noise and improves model quality.
    """
    text = text.lower()

    # Remove non-alphabet characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -------------------------------
# STEP 4: TOKENIZATION
# -------------------------------

def tokenize_text(text):
    """
    Splits text into tokens (words).

    Why tokenization?:
    Word2Vec requires tokenized input.
    """
    tokens = word_tokenize(text)
    return tokens


# -------------------------------
# STEP 5: BUILD CORPUS
# -------------------------------

def build_corpus(text_list):
    """
    Converts list of documents into tokenized corpus.

    Output format:
    [
        ['iit', 'jodhpur', 'offers', ...],
        ['students', 'can', 'apply', ...]
    ]
    """
    corpus = []

    for text in text_list:
        cleaned = clean_text(text)
        tokens = tokenize_text(cleaned)

        # Avoid empty documents
        if len(tokens) > 5:
            corpus.append(tokens)

    return corpus


# -------------------------------
# STEP 6: DATASET STATISTICS
# -------------------------------

def compute_statistics(corpus):
    """
    Computes:
    - Total documents
    - Total tokens
    - Vocabulary size
    """
    total_docs = len(corpus)

    all_tokens = [word for doc in corpus for word in doc]

    total_tokens = len(all_tokens)

    vocab = set(all_tokens)
    vocab_size = len(vocab)

    print("\n DATASET STATISTICS")
    print(f"Total Documents: {total_docs}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Vocabulary Size: {vocab_size}")

    return all_tokens


# -------------------------------
# STEP 7: WORD CLOUD
# -------------------------------

def generate_wordcloud(tokens):
    """
    Generates word cloud from token frequency.

    WHY:
    Helps visualize most frequent words in corpus.
    """
    text = " ".join(tokens)

    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wc)
    plt.axis('off')
    plt.title("Word Cloud of IIT Jodhpur Corpus")
    plt.show()


# -------------------------------
# MAIN EXECUTION
# -------------------------------

if __name__ == "__main__":

    # URLS to scrape (IIT Jodhpur website)
    urls = [
    "https://iitj.ac.in/office-of-academics/en/list-of-academic-programs",
    "https://iitj.ac.in/office-of-academics/en/program-Structure",
    "https://www.iitj.ac.in/admission-postgraduate-programs/en/Admission-to-Postgraduate-Programs",
    "https://www.iitj.ac.in/main/en/research-highlight",
    "https://www.iitj.ac.in/sst/en/research"
]

    documents = []

    # Scrape websites
    for url in urls:
        text = scrape_website(url)
        documents.append(text)

    local_pdfs = [
        "btech_rules.pdf", 
        "mtech_rules.pdf", 
        "phd_rules.pdf",
        "annual_report2122.pdf",
        "annual_report2223.pdf",
        "newslettermay24.pdf"
    ]

    print("Extracting text from local PDFs...")
    for pdf_file in local_pdfs:
        pdf_text = extract_pdf_text(pdf_file)
        if pdf_text:  # Only append if text was actually extracted
            documents.append(pdf_text)

    # Build corpus
    corpus = build_corpus(documents)

    # Compute stats
    tokens = compute_statistics(corpus)

    # Generate word cloud
    generate_wordcloud(tokens)

    # Save corpus to file
    save_corpus_to_file(corpus)