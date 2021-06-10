#!/usr/bin/env python3
""" Question and answering with semantic search """
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np
import os


def answer_question(question, reference):
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

def answer_loop(corpus_path):
    """
    Answers questions from a reference text
    Responds with 'Sorry, I do not understand your question.' if an answer
        could not be found.
    Entering 'exit', 'quit', 'goodbye', or 'bye' exits the loop
    """

    while True:
        question = input("Q: ").lower()
        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        # Get most similar reference document
        reference = semantic_search(corpus_path, question)

        # Find answer within reference document
        answer = answer_question(question, reference)

        if type(answer) is not str or len(answer) < 1 or answer in question:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: " + answer)

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

def question_answer(corpus_path):
    """
    Interactive questions answering from multiple reference texts
    """
    answer_loop(corpus_path)
