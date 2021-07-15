#!/usr/bin/env python3
""" Bag of Words Embedding """
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
    sentences: list of sentences to analyze
    vocab: list of the vocabulary words
    Returns embeddings, features
        embeddings: A numpy.ndarray of shape(s, f) containing the embeddings
            s: Number of sentences in sentences list
            f: Number of features analyzed
        features: A list of the features used for embeddings
    """
    vectorizer = CountVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
