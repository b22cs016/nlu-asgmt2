"""
TASK-4: VISUALIZATION USING PCA

This script:
1. Loads trained Word2Vec models (CBOW + Skip-gram)
2. Selects important words
3. Projects embeddings into 2D using PCA
4. Visualizes clusters
"""

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# -------------------------------
# STEP 1: LOAD MODELS
# -------------------------------

cbow_model = Word2Vec.load("cbow_dim100_win5_neg5.model")
sg_model = Word2Vec.load("skipgram_dim100_win5_neg5.model")


# -------------------------------
# STEP 2: SELECT WORDS
# -------------------------------

# Choose meaningful academic/domain words
words = [
    "student", "research", "phd", "exam",
    "course", "faculty", "degree", "engineering",
    "science", "ai", "data", "learning",
    "program", "university", "study"
]


def filter_words(model, words):
    """Keep only words present in vocabulary"""
    return [w for w in words if w in model.wv]


cbow_words = filter_words(cbow_model, words)
sg_words = filter_words(sg_model, words)


# -------------------------------
# STEP 3: PCA FUNCTION
# -------------------------------

def plot_pca(model, words, title):
    """
    Projects word vectors into 2D and plots them
    """

    vectors = [model.wv[word] for word in words]

    # Apply PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    x = reduced[:, 0]
    y = reduced[:, 1]

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y)

    # Annotate each point
    for i, word in enumerate(words):
        plt.annotate(word, (x[i], y[i]))

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()

    plt.show()


# -------------------------------
# STEP 4: VISUALIZE BOTH MODELS
# -------------------------------

print("\n PCA Visualization: CBOW")
plot_pca(cbow_model, cbow_words, "CBOW Word Embeddings (PCA)")

print("\n PCA Visualization: Skip-gram")
plot_pca(sg_model, sg_words, "Skip-gram Word Embeddings (PCA)")