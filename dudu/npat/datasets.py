"""
Define dataset sources here.
"""

import pickle

import torch


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, padding_token_id: int):
        with open(path, "rb") as file:
            self.data = pickle.load(file=file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        """
        Won't perform padding here.
        """
        prompt, label = self.data[index]

        return prompt, label
