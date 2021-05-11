#!/usr/bin/env python3
""" GenSim model to Keras embedding layer """


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer
    model: The gensim model to convert
    Returns: the trainable keras Embedding
    """
    return model.wv.get_keras_embedding()
