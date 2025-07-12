"""
Functions to tokenize prompts and perform padding
"""

import pickle


def tokenize(tokenizer, data_path: str, path_to_save: str):
    # Load prompts
    with open(path_to_save, "rb") as file:
        prompts = pickle.load(file)

    pass
