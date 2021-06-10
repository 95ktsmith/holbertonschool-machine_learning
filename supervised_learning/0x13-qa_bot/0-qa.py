#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question
    question: String containing the question to answer
    reference: String containing the reference document from which to find
        the answer
    Returns: A string containing the answer
    """

    # Load pre-trained tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

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
    input_type_ids = [0] * (len(question_tokens) + 2)  # +2: CLS and SEP tokens
    input_type_ids += [1] * (len(reference_tokens) + 1)  # +1: SEP token

    # Convert to tensors and shape for input
    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids)
    )

    # Run inputs through model
    outputs = model([input_word_ids, input_mask, input_type_ids])

    # Get mostly like start and end tokens
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    # Slice and convert back to string
    answer_tokens = tokens[short_start:short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer
