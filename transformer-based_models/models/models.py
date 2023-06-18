from typing import Dict

import torch
from torch import nn
from transformers import AutoModel


class MultiTaskModel(nn.Module):
    def __init__(self, drop_rate: float, lm: str) -> None:
        super().__init__()

        self._language_model = AutoModel.from_pretrained(lm)
        self._common_block = nn.RNN(
            input_size=768,
            hidden_size=768,
            num_layers=2,
            bias=True,
            batch_first=True,
            bidirectional=True,
            nonlinearity="relu",
            dropout=drop_rate,
        )
        self._gender_layer = nn.Sequential(nn.Linear(1536, 1))
        self._i_bin_layer = nn.Sequential(nn.Linear(1536, 1))
        self._i_mul_layer = nn.Sequential(nn.Linear(1536, 4))

    def forward(self, tokenized: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cls_repr = self._language_model(**tokenized)[0][:, 0, :]
        out, _ = self._common_block(cls_repr)

        return {
            "gender": self._gender_layer(out),
            "ideology_binary": self._i_bin_layer(out),
            "ideology_multiclass": self._i_mul_layer(out),
        }
