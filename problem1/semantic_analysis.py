"""
TASK-3: SEMANTIC ANALYSIS USING WORD2VEC

This script:
1. Loads a trained Word2Vec model
2. Finds nearest neighbors using cosine similarity
3. Performs analogy reasoning
"""

from gensim.models import Word2Vec


# -------------------------------
# STEP 1: LOAD BEST MODEL
# -------------------------------

# this was the best performing model 
model_path = "skipgram_dim100_win5_neg5.model"

model = Word2Vec.load(model_path)

print(f"\nLoaded Model: {model_path}")
print(f"Vocabulary Size: {len(model.wv)}")


# -------------------------------
# STEP 2: NEAREST NEIGHBORS
# -------------------------------

def get_neighbors(word, topn=5):
    """
    Returns top-N most similar words using cosine similarity.
    Handles missing words safely.
    """
    print(f"\n🔍 Nearest neighbors for: '{word}'")

    if word not in model.wv:
        print("Word not found in vocabulary")
        return

    neighbors = model.wv.most_similar(word, topn=topn)

    for w, score in neighbors:
        print(f"{w:15}  similarity = {round(score, 4)}")


# Words given in question
words_to_check = ["research", "student", "phd", "exam"]

for w in words_to_check:
    get_neighbors(w)


# -------------------------------
# STEP 3: ANALOGY EXPERIMENTS
# -------------------------------

def solve_analogy(pos_words, neg_words):
    """
    Solves analogy: A : B :: C : ?
    using vector arithmetic
    """

    print(f"\n Analogy: {pos_words} - {neg_words}")

    try:
        result = model.wv.most_similar(
            positive=pos_words,
            negative=neg_words,
            topn=3
        )

        for word, score in result:
            print(f"{word:15}  score = {round(score, 4)}")

    except KeyError as e:
        print(f"Word missing: {e}")


# -------------------------------
# ANALOGY TESTS
# -------------------------------

# Example 1: UG : BTech :: PG : ?
solve_analogy(pos_words=["pg", "btech"], neg_words=["ug"])

# Example 2: student : exam :: teacher : ?
solve_analogy(pos_words=["teacher", "exam"], neg_words=["student"])

# Example 3: research : phd :: study : ?
solve_analogy(pos_words=["study", "phd"], neg_words=["research"])

# Example 4: ai : engineering :: science : ?
solve_analogy(pos_words=["ai", "engineering"], neg_words=["science"])


# -------------------------------
# STEP 4: INTERPRETATION HELPER
# -------------------------------

print("\n NOTE:")
print("Higher similarity/score → stronger semantic relationship")