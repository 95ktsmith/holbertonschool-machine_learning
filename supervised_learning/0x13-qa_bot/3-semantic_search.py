#!/usr/bin/env python3
""" Semantic Search """
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents
    corpus_path is the path to the corpus of reference documents on which to
        perform semantic search
    sentence is the sentence from which to perform semantic search
    Returns: the reference text of the document most simlilar to sentence
    """

    # Load all reference documents
    # sentence is the first item in the list because I couldn't get an
    # embedding with it in its own list
    docs = [sentence]
    for doc in os.listdir(corpus_path):
        if doc[-3:] == ".md":
            with open(corpus_path + "/" + doc, "r") as f:
                docs.append(f.read())

    # Load embedder
    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    )

    # Get embeddings for query sentence and reference documents
    embeddings = embed(docs)

    # Calculate cosine similarites between query embedding and each reference
    # document embedding
    cos_sims = []
    query = embeddings[0]

    for doc in embeddings[1:]:  # Skip similarity between query and itself
        # All embedding vectors have a vector length of 1, meaning just the
        # dot product will be the cosine similarity
        cos_sims.append(np.dot(query, doc))

    # Get index of highest score (most similar)
    most_similar = np.argmax(cos_sims)

    # Return that document (+1 for skipping self-similarity in loop)
    return docs[most_similar + 1]
