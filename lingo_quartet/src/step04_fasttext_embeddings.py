"""
Create FastText embeddings in preparation for the hierarchical clustering
process.

Authored by Megan
"""

from gensim.models import FastText
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_data(
    path="./data/metadata_w_2020articles_cleaned.csv",
):
    """
    Read and check dataframe.

    Returns:
        df(pd.DataFrame): dataframe of 2020 articles
    """
    df = pd.read_csv(path, index_col=0)
    train, test = train_test_split(df, test_size=0.4, random_state=42)
    return train


def averaged_word2vec_vectorizer(corpus, model, num_features):
    """
    Create embeddings for each element in the corpus.

    Inputs:
        corpus: corpus of words to create embeddings for
        model(FastText): FastText model to create embeddings from
        num_features(int): number of features aka number of words/tokens

    Returns:
        (np.array) of embedded tokens
    """
    vocabulary = set(model.wv.index_to_key)

    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.0

        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.0
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [
        average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
        for tokenized_sentence in corpus
    ]
    return np.array(features)


def embed(df: pd.DataFrame):
    """
    Create the FastText embeddings for each article in the dataframe.

    Inputs:
        df(pd.DataFrame): dataframe of articles to create embeddings for
    """
    corpus = df.loc[:, "title"]  # first create embeddings for the titles

    # tokenizing title docs
    tokenized_docs = [doc.split() for doc in corpus]
    ft_model = FastText(
        tokenized_docs,
        vector_size=512,
        window=20,
        min_count=2,
        workers=1,
        sg=1,
        seed=42,
        epochs=10,
    )

    doc_vecs_ft = averaged_word2vec_vectorizer(tokenized_docs, ft_model, 512)
    print(f"Embedding shape: {doc_vecs_ft.shape}")

    return doc_vecs_ft


def write_embeddings(
    arr: np.array,
    filepath: str = "data/fasttext.npy",
):
    """
    Write out embeddings to numpy filepath since the embedding process
    can take a very long time.

    Inputs:
        arr(np.array): numpy array of embeddings to write out
        filepath(str): pickle filepath to write to
    """
    np.save(filepath, arr)


if __name__ == "__main__":
    df = read_data()
    df = embed(df)
    write_embeddings(df)
