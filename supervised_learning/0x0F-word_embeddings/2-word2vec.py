#!/usr/bin/env python3
""" Word2Vec """
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a Word2Vec model
    Returns: The trained model
    """
    model = Word2Vec(
        sentences=sentences,
        size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=1 if cbow is False else 0,
        iter=iterations,
        seed=seed,
        workers=workers
    )
    return model
