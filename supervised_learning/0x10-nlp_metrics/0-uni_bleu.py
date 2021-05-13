#!/usr/bin/env python3
""" Unigram BLEU Score """
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    references: List of reference translations
        Each reference translation is a list of the words in the translation
    sentence: List containing the model proposed sentence
    Returns: The unigram BLEU score
    """
    counts = []
    for word in sentence:
        max_credits = 0
        counts.append(0)
        for reference in references:
            credits = reference.count(word)
            if credits > max_credits:
                max_credits = credits
            counts[-1] = min(max_credits, max(credits, counts[-1]))

    # Length of shortest reference sentence
    r = min([len(ref) for ref in references])
    # Length of translation sentence
    c = len(sentence)
    if c <= r:
        brevity_penalty = np.exp(1 - (r / c))
    else:
        brevity_penalty = 1
    return sum(counts) * brevity_penalty / c
