#!/usr/bin/env python3
""" TF-IDF Embedding """
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding
    sentences: A list of sentences to analyze
    vocab: A list of the vocabulary words to use for the analysis
    Returns: embeddings, features
        embeddings: A numpy.ndarray of shape(s, f) containing the embeddings
            s: Number of sentences in sentences list
            f: Number of features analyzed
        features: A list of the features used for embeddings
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
