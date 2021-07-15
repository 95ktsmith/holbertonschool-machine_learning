#!/usr/bin/env python3
""" Dataset """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ Class to load an prepare a dataset for machine translation """
    def __init__(self, batch_size, max_len):
        """
        Init
        batch_size: Batch size for training/validation
        max_len: maximum number of tokens allowed per example sentence
        """
        # Load dataset, split into training and validation sets
        data = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            as_supervised=True
        )
        self.data_train = data['train']
        self.data_valid = data['validation']

        # Create tokenizers for encoding
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        def filter_by_max_len(x, y, max_len=max_len):
            """
            Function to filter datasets, removing examples where either input
            or target sentences have more tokens than max_len
            """
            return tf.logical_and(
                tf.size(x) <= max_len,
                tf.size(y) <= max_len
            )

        # Encode, filter, cache, shuffle, batch, and prefetch training set
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(
            filter_by_max_len
        )
        self.data_train = self.data_train.cache()

        # Get number of examples in set to use for shuffling
        num_examples = 0
        for example in self.data_train:
            num_examples += 1

        self.data_train = self.data_train.shuffle(num_examples)
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        # Encode, filter, and batch validation set
        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(
            filter_by_max_len
        )
        self.data_valid = self.data_valid.padded_batch(batch_size)

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

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        pt: tf.Tensor containing the Portuguese sentence
        en: tf.Tensor containing the corresponding English sentence
        Returns: pt_tokens, en_tokens
            pt_tokens: np.ndarray containing Portuguese tokens
            en_tokens: np.ndarray containing the English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens += self.tokenizer_pt.encode(pt.numpy())
        pt_tokens += [pt_tokens[0] + 1]

        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens += self.tokenizer_en.encode(en.numpy())
        en_tokens += [en_tokens[0] + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Tensorflow wrapper for encode instance method
        Returns: pt, en
        """
        pt, en = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        pt.set_shape([None])
        en.set_shape([None])
        return pt, en
