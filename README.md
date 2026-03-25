# NLP Assignment 2  
## Word Embeddings & Character-Level Name Generation

This repository contains solutions for two NLP tasks:

- **Problem 1:** Learning Word Embeddings using Word2Vec on IIT Jodhpur data  
- **Problem 2:** Character-level Name Generation using RNN-based models  

---

#  Project Structure

```
.
├── problem1/
│   ├── data_extract.py
│   ├── model_train.py
│   ├── semantic_analysis.py
│   ├── visualization.py
│   ├── print_vector.py
│   ├── 300d_model.py
│   
│
├── problem2/
│   ├── clean_names.py
│   ├── main_notebook.ipynb
│   └── TrainingNames.txt (generated)
```

---

#  Requirements

Install dependencies using:

```bash
pip install gensim nltk matplotlib scikit-learn wordcloud pdfminer.six beautifulsoup4 requests torch
```

---

#  Problem 1: Word2Vec on IIT Jodhpur Corpus

## Step 1: Dataset Preparation

Run:

```bash
python problem1/data_extract.py
```

This script:
- Scrapes IIT Jodhpur website data  
- Extracts text from PDFs  
- Cleans and tokenizes text  
- Generates dataset statistics and word cloud  
- Saves processed corpus as `cleaned_corpus.txt`

---

## Step 2: Train Word2Vec Models

```bash
python problem1/model_train.py
```

This will:
- Load corpus and create pseudo-sentences  
- Train CBOW and Skip-gram models  
- Save trained models  
- Print training summary  

---

## Step 3: Semantic Analysis

```bash
python problem1/semantic_analysis.py
```

This script:
- Finds nearest neighbors  
- Performs analogy tasks  
- Uses cosine similarity  

---

## Step 4: Visualization

```bash
python problem1/visualization.py
```

- Applies PCA to embeddings  
- Plots word clusters  

---

## Step 5: Print Word Embedding (300D)

```bash
python problem1/300d_model.py
python problem1/print_vector.py
```

- Trains a 300-dimensional Word2Vec model  
- Prints embedding vector  

---

#  Problem 2: Character-Level Name Generation

## Step 1: Prepare Dataset

```bash
python problem2/clean_names.py
```

- Cleans raw names  
- Saves as TrainingNames.txt  

---

## Step 2: Train Models & Evaluate

Open and run:

```bash
problem2/main_notebook.ipynb
```

This notebook:
- Implements RNN, BiLSTM, Attention models  
- Trains models  
- Generates names  
- Computes novelty and diversity  

---

#  Evaluation Metrics

- Novelty Rate: Percentage of generated names not in training data  
- Diversity: Unique generated names divided by total generated names  

---

#  Key Observations

- RNN produced the most realistic names  
- BiLSTM showed high novelty but poor realism  
- Attention model suffered from repetition  

---

#  Notes

- Custom preprocessing used due to lack of sentence boundaries  
- Sliding window approach used for pseudo-sentences  
- Temperature-based sampling used for generation  

---

#  Author

Siddhesh Ayyathan
B22CS016
