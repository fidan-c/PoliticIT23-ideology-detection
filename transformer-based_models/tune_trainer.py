import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from ray.tune import Trainable
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from models.models import MultiTaskModel
from validate import validate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(Trainable):
    def setup(
        self,
        config: Dict[str, Any],
        train_data: Dataset,
        val_data: Dataset,
        gender_weights: torch.Tensor,
        i_bin_weights: torch.Tensor,
        i_mul_weights: torch.Tensor,
        lm: str,
    ):
        self._config = config
        self._batch_size = self._get_batch_size(self._config)
        self._train_loader = DataLoader(train_data, self._batch_size, shuffle=True)
        self._val_loader = DataLoader(val_data, self._batch_size, shuffle=True)
        self._model = MultiTaskModel(self._config["drop_rate"], lm).to(DEVICE)
        self._optimizer = self._get_optimizer(self._config)
        self._scheduler = get_cosine_schedule_with_warmup(
            optimizer=self._optimizer,
            num_warmup_steps=5,
            num_training_steps=len(self._train_loader) * config["epochs"],
        )
        self._bce_loss_gender = BCEWithLogitsLoss(pos_weight=gender_weights)
        self._bce_loss_i_bin = BCEWithLogitsLoss(pos_weight=i_bin_weights)
        self._cre_loss = CrossEntropyLoss(weight=i_mul_weights)

    def _get_batch_size(self, config: Dict[str, Any]) -> int:
        return 2 ** (math.ceil(math.log(config["batch_size"], 2)))

    def _get_optimizer(self, config: Dict[str, Any]) -> Optimizer:
        return AdamW(
            params=self._model.parameters(),
            lr=config["optimizer"]["AdamW"]["lr"],
            weight_decay=config["optimizer"]["AdamW"]["weight_decay"],
            betas=(
                config["optimizer"]["AdamW"]["beta1"],
                config["optimizer"]["AdamW"]["beta2"],
            ),
        )

    def step(self) -> Dict[str, float]:
        return {**self._train(), **self._validate()}

    def _train(self) -> Dict[str, float]:
        self._model.train()

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
            pred_i_mul = out["ideology_multiclass"].view(-1, 4).to(DEVICE, torch.float)

            # compute losses and wrap them into function h (https://arxiv.org/abs/2002.04792)
            h = lambda loss: torch.exp(torch.div(loss, self._config["T"]))

            gender_bce_loss = h(self._bce_loss_gender(pred_gender, gender))
            i_bin_bce_loss = h(self._bce_loss_i_bin(pred_i_bin, i_bin))
            i_mul_cre_loss = h(self._cre_loss(pred_i_mul, i_mul))

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
        print(f"Tune-Train | epoch: {self.iteration} | loss: {avg_train_loss}")

        return {"train_loss": avg_train_loss}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        f1_score = validate(self._val_loader, self._model, self.iteration)

        return {"f1_score": f1_score}

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        checkpoint_path = str(Path(checkpoint_dir) / "model.pt")
        torch.save(self._model.state_dict(), checkpoint_path)

        return checkpoint_dir
