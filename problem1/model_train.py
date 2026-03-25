"""
TASK-2: WORD2VEC MODEL TRAINING 

This version is adapted for corpus WITHOUT punctuation.
Instead of sentence tokenization, we:
- Split long lines into fixed-size chunks
- Tokenize each chunk
- Train CBOW and Skip-gram models

This ensures proper context learning.
"""

from gensim.models import Word2Vec
import time


# -------------------------------
# STEP 1: LOAD + FIX CORPUS
# -------------------------------

def load_corpus(file_path, chunk_size=20):
    """
    Custom corpus loader for long continuous text.

    WHY THIS WORKS:
    - Your data has no sentence boundaries
    - So we manually create "pseudo-sentences"
    - Each chunk acts like a sentence for Word2Vec
    """

    corpus = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()

            if not line:
                continue

            # Split into words
            words = line.split()

            # Create chunks (IMPORTANT FIX)
            for i in range(0, len(words), chunk_size):
                chunk = words[i:i + chunk_size]

                # Keep meaningful chunks only
                if len(chunk) >= 5:
                    corpus.append(chunk)

    return corpus


# Load corpus
corpus = load_corpus("cleaned_corpus.txt", chunk_size=20)

print(f"Corpus Loaded: {len(corpus)} sentences")

# Debug: show few samples
print("\nSample sentences:")
for i in range(min(5, len(corpus))):
    print(corpus[i])


# -------------------------------
# STEP 2: HYPERPARAMETERS
# -------------------------------

dimensions = [50, 100]
windows = [2, 5, 10]
neg_samples = [5, 10]

trained_models = []


# -------------------------------
# STEP 3: CBOW TRAINING
# -------------------------------

print("\n Training CBOW models...")

for dim in dimensions:
    for win in windows:
        for neg in neg_samples:

            model_name = f"cbow_dim{dim}_win{win}_neg{neg}"
            print(f"Training: {model_name}")

            start_time = time.time()

            model = Word2Vec(
                sentences=corpus,
                vector_size=dim,
                window=win,
                sg=0,              # CBOW
                negative=neg,
                min_count=2,
                workers=4
            )

            end_time = time.time()

            model.save(f"{model_name}.model")

            trained_models.append({
                "Architecture": "CBOW",
                "Dimension": dim,
                "Window": win,
                "Negative Samples": neg,
                "Vocab Size": len(model.wv),
                "Time (s)": round(end_time - start_time, 2)
            })


# -------------------------------
# STEP 4: SKIP-GRAM TRAINING
# -------------------------------

print("\nTraining Skip-gram models...")

for dim in dimensions:
    for win in windows:
        for neg in neg_samples:

            model_name = f"skipgram_dim{dim}_win{win}_neg{neg}"
            print(f"Training: {model_name}")

            start_time = time.time()

            model = Word2Vec(
                sentences=corpus,
                vector_size=dim,
                window=win,
                sg=1,              # Skip-gram
                negative=neg,
                min_count=2,
                workers=4
            )

            end_time = time.time()

            model.save(f"{model_name}.model")

            trained_models.append({
                "Architecture": "Skip-gram",
                "Dimension": dim,
                "Window": win,
                "Negative Samples": neg,
                "Vocab Size": len(model.wv),
                "Time (s)": round(end_time - start_time, 2)
            })


# -------------------------------
# STEP 5: PRINT RESULTS
# -------------------------------

print("\n TRAINING SUMMARY\n")

print("{:<12} {:<8} {:<8} {:<18} {:<10} {:<8}".format(
    "Model", "Dim", "Window", "NegSamples", "Vocab", "Time"
))

print("-" * 70)

for m in trained_models:
    print("{:<12} {:<8} {:<8} {:<18} {:<10} {:<8}".format(
        m["Architecture"],
        m["Dimension"],
        m["Window"],
        m["Negative Samples"],
        m["Vocab Size"],
        m["Time (s)"]
    ))


# -------------------------------
# STEP 6: SAVE RESULTS
# -------------------------------

with open("training_results.txt", "w") as f:
    for m in trained_models:
        f.write(str(m) + "\n")

print("\n All models trained successfully!")