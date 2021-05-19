#!/usr/bin/env python3
""" n-gram BLEU Score """
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence
    references: List of reference translations
        Each reference translation is a list of the words in the translation
    sentence: List containing the model proposed sentence
    n: size of the n-gram to use for evaluation
    Returns: The unigram BLEU score
    """
    counts = {}
    sen_grams = []
    for i in range(len(sentence) - (n - 1)):
        sen_grams.append(sentence[i:i+n])

    ref_grams = []
    for reference in references:
        ref_grams.append([])
        for i in range(len(reference) - (n - 1)):
            ref_grams[-1].append(reference[i:i+n])

    for gram in sen_grams:
        max_credits = 0
        counts[str(gram)] = 0
        for reference in ref_grams:
            credits = reference.count(gram)
            if credits > max_credits:
                max_credits = credits
            counts[str(gram)] = min(max_credits,
                                    max(credits, counts[str(gram)]))

    # Length of shortest reference sentence
    r = min([len(ref) for ref in references])
    # Length of translation sentence
    c = len(sentence)
    if c <= r:
        brevity_penalty = np.exp(1 - (r / c))
    else:
        brevity_penalty = 1

    return sum(counts.values()) * brevity_penalty / len(sen_grams)
