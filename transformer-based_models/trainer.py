import math
from pathlib import Path
from typing import Any, Dict

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from models.models import MultiTaskModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        train_data: Dataset,
        epochs: int,
        tuned_params: Dict[str, Any],
        gender_weights: torch.Tensor,
        i_bin_weights: torch.Tensor,
        i_mul_weights: torch.Tensor,
        lm: str,
    ) -> None:
        self._hyperparams = tuned_params
        self._epochs = epochs
        self._batch_size = self._get_batch_size()
        self._model = MultiTaskModel(self._hyperparams["drop_rate"], lm).to(DEVICE)
        self._train_loader = DataLoader(train_data, self._batch_size, shuffle=True)
        self._optimizer = self._get_optimizer()
        self._scheduler = get_cosine_schedule_with_warmup(
            optimizer=self._optimizer,
            num_warmup_steps=5,
            num_training_steps=len(self._train_loader) * epochs,
        )
        self._gender_weights = gender_weights
        self._i_bin_weights = i_bin_weights
        self._i_mul_weights = i_mul_weights

    def _get_batch_size(self) -> int:
        return 2 ** (math.ceil(math.log(self._hyperparams["batch_size"], 2)))

    def _get_optimizer(self) -> Optimizer:
        return AdamW(
            params=self._model.parameters(),
            lr=self._hyperparams["optimizer"]["AdamW"]["lr"],
            weight_decay=self._hyperparams["optimizer"]["AdamW"]["weight_decay"],
            betas=(
                self._hyperparams["optimizer"]["AdamW"]["beta1"],
                self._hyperparams["optimizer"]["AdamW"]["beta2"],
            ),
        )

    def train(self) -> None:
        self._model.train()

        bce_loss_gender = BCEWithLogitsLoss(pos_weight=self._gender_weights)
        bce_loss_i_bin = BCEWithLogitsLoss(pos_weight=self._i_bin_weights)
        cre_loss = CrossEntropyLoss(weight=self._i_mul_weights)

        for epoch in range(self._epochs):
            # sum of all batch losses for a given epoch i.e total epoch loss
            train_loss = 0

            for labels, tokenized in self._train_loader:
                # golden labels
                gender = labels.get("gender").to(DEVICE, torch.float)
                i_bin = labels.get("i_bin").to(DEVICE, torch.float)
                i_mul = labels.get("i_mul").view(-1).to(DEVICE)

                # clear the gradients of all optimized variables
                self._optimizer.zero_grad()

                # forward pass: predict gender and ideology
                out = self._model(tokenized)
                pred_gender = out["gender"].view(-1).to(DEVICE, torch.float)
                pred_i_bin = out["ideology_binary"].view(-1).to(DEVICE, torch.float)
                pred_i_mul = (
                    out["ideology_multiclass"].view(-1, 4).to(DEVICE, torch.float)
                )

                # compute losses and wrap them into function h (https://arxiv.org/abs/2002.04792)
                h = lambda loss: torch.exp(torch.div(loss, self._hyperparams["T"]))

                gender_bce_loss = h(bce_loss_gender(pred_gender, gender))
                i_bin_bce_loss = h(bce_loss_i_bin(pred_i_bin, i_bin))
                i_mul_cre_loss = h(cre_loss(pred_i_mul, i_mul))

                train_batch_loss = gender_bce_loss + i_bin_bce_loss + i_mul_cre_loss

                # backward pass: compute batch loss gradient with respect to model parameters
                train_batch_loss.backward()

                # perform a single optimization step (parameter update)
                self._optimizer.step()

                self._scheduler.step()

                # total loss of the current batch (not averaged)
                train_loss += train_batch_loss.item() * len(labels.get("gender"))

            # averaged loss across all samples for the current epoch [ 1 ]
            avg_train_loss = train_loss / len(list(self._train_loader.sampler))
            print(f"Taining | epoch: {epoch} | loss: {avg_train_loss}")

        torch.save(self._model.state_dict(), Path(__file__).parent / "checkpoint.pt")


""" NOTES
[ 1 ]   https://discuss.pytorch.org/t/on-running-loss-and-average-loss/107890
        https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#training-the-model
        https://stackoverflow.com/a/61094330
"""
