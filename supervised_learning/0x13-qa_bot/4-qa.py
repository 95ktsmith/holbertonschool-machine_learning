#!/usr/bin/env python3
""" Question and answering with semantic search """
import tensorflow as tf
import os
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np


def question_answer(corpus_path):
    """
    Interactive questions answering from multiple reference texts
    corpus_path: Path to the corpus of reference documents
    """

    # Load pre-trained tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Load corpus of reference documents
    references = []
    for doc in os.listdir(corpus_path):
        if doc[-3:] == ".md":
            with open(corpus_path + "/" + doc, "r") as f:
                references.append(f.read())

    # Load embedder
    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    )
    # Get embeddings of reference documents, transposed for use later
    ref_embeddings = tf.transpose(embed(references))

    @tf.function(experimental_relax_shapes=True)
    def get_reference(question):
        """
        Returns the reference document index most similar to the question
        """
        query = embed([question, "tmp string to fit shape requirements"])[0]
        cos_sims = tf.tensordot(query, ref_embeddings, 1)
        return tf.math.argmax(cos_sims)

    @tf.function(experimental_relax_shapes=True)
    def get_answer(question, reference):
        """
        Finds a snippet of text within a reference document to answer question
        question: String containing the question to answer
        reference: String containing the reference document from which to find
            the answer
        Returns: A string containing the answer
        """
        # Build list of tokens
        question_tokens = tokenizer.tokenize(question)
        reference_tokens = tokenizer.tokenize(reference)
        tokens = ['[CLS]'] + \
            question_tokens + \
            ['[SEP]'] + \
            reference_tokens + \
            ['[SEP]']

        # Create token IDs, mask, and type IDs
        input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_word_ids)
        input_type_ids = [0] * (len(question_tokens) + 2)  # +2: CLS/SEP tokens
        input_type_ids += [1] * (len(reference_tokens) + 1)  # +1: SEP token

        # Convert to tensors and shape for input
        input_word_ids, input_mask, input_type_ids = map(
            lambda t: tf.expand_dims(
                tf.convert_to_tensor(t, dtype=tf.int32),
                0
            ),
            (input_word_ids, input_mask, input_type_ids)
        )

        # Run inputs through model
        outputs = model([input_word_ids, input_mask, input_type_ids])

        # Get mostly like start and end tokens
        start = tf.argmax(outputs[0][0][1:]) + 1
        end = tf.argmax(outputs[1][0][1:]) + 1

        return tokens, start, end

    # Start Q&A Loop
    while True:
        question = input("Q: ").lower()
        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        # Get most similar reference document
        reference = references[get_reference(question)]

        # Find answer within reference document
        tokens, start, end = get_answer(question, reference)
        answer_tokens = tokens[start:end + 1]
        # This is gross but it works
        answer_tokens = list(map(
            lambda t: str(t.numpy())[2:-1],
            answer_tokens)
        )
        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        # Print result
        if type(answer) is not str or \
                len(answer) < 1 or\
                answer in question or\
                '[SEP]' in answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: " + answer)
