from gensim.models import Word2Vec

model = Word2Vec.load("word2vec_300.model")

# choose a word to print its vector representation
word = "student"   

if word in model.wv:
    vector = model.wv[word]

    # format as comma-separated list
    vector_str = ", ".join([f"{v:.4f}" for v in vector])

    print(f"{word} - {vector_str}")
else:
    print("Word not found in vocabulary")