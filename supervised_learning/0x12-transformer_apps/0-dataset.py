#!/usr/bin/env python3
""" Dataset """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ Class to load an prepare a dataset for machine translation """
    def __init__(self):
        """ Init """
        data = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            as_supervised=True
        )
        self.data_train = data['train']
        self.data_valid = data['validation']

        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for a dataset
        data: tf.data.Dataset whose examples are formatted as a table (pt, en)
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the English sentence
        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        encoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = encoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=2 ** 15
        )
        tokenizer_en = encoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=2 ** 15
        )
        return tokenizer_pt, tokenizer_en
