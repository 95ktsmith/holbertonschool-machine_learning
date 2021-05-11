#!/usr/bin/env python3
""" FastText """
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a fasttext model
    Returns: The trained model
    """
    model = FastText(
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
