from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

TokenizerType = Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AbstractDataset(ABC, Dataset):
    def __init__(self, df: pd.DataFrame, tok: TokenizerType) -> None:
        super().__init__()
        self._data = df
        self._tok = tok

    def _tokenize_sample(self, sample: str) -> Dict[str, torch.Tensor]:
        tokenized = self._tok(
            sample,
            padding="max_length",
            truncation=False,
            return_tensors="pt",
        )
        tokenized = {key: val.view(-1).to(DEVICE) for key, val in tokenized.items()}

        return tokenized

    def __len__(self) -> int:
        return len(self._data)

    @abstractmethod
    def __getitem__(self, index: int):
        return NotImplementedError


class TrainValDataset(AbstractDataset):
    def __init__(self, df: pd.DataFrame, tok: TokenizerType) -> None:
        super().__init__(df, tok)

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        sample = self._data.iloc[index]
        tokenized = self._tokenize_sample(sample["tweet"])
        labels = {
            "gender": sample["gender"],
            "i_bin": sample["ideology_binary"],
            "i_mul": sample["ideology_multiclass"],
            "author": sample["author"],
            "num_tokens": sample["num_tokens"],
        }

        return labels, tokenized


class TestDataset(AbstractDataset):
    def __init__(self, df: pd.DataFrame, tok: TokenizerType) -> None:
        super().__init__(df, tok)

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        sample = self._data.iloc[index]
        tokenized = self._tokenize_sample(sample["tweet"])
        labels = {"author": sample["author"], "num_tokens": sample["num_tokens"]}

        return labels, tokenized
