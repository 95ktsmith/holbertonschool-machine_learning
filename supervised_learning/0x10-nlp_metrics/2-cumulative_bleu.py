#!/usr/bin/env python3
""" Cumulative BLEU Score """
ngram_bleu = __import__('1-ngram_bleu').ngram_bleu


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence
    references: List of reference translations
        Each reference translation is a list of the words in the translation
    sentence: List containing the model proposed sentence
    n: size of the largest n-gram to use for evaluation
    Returns: The unigram BLEU score
    """
    individual_scores = []
    for n_gram in range(1, n + 1):
        individual_scores.append(ngram_bleu(references, sentence, n_gram))

    geo_mean = 1
    for score in individual_scores:
        geo_mean *= score
    geo_mean = geo_mean ** (1 / len(individual_scores))

    return geo_mean
