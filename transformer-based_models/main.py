import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
import typer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from compute_scores import compute_scores
from data.data_preprocessor import DataPreprocessor
from data.dataset_classes import TestDataset, TrainValDataset
from inference import inference
from models.models import MultiTaskModel
from trainer import Trainer
from tune_parameters import tune_parameters

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_weights(train_data: pd.DataFrame) -> Tuple[Tensor, Tensor, Tensor]:
    # compute pos_weights [ 1 ] for gender label
    gender_counter = Counter(train_data["gender"])
    gender_w = torch.Tensor([gender_counter[0] / gender_counter[1]]).to(DEVICE)

    # compute pos_weights [ 1 ] for ideology_binary
    i_bin_counter = Counter(train_data["ideology_binary"])
    i_bin_w = torch.Tensor([i_bin_counter[0] / i_bin_counter[1]]).to(DEVICE)

    # compute weights for ideology_multiclass
    class_labels = train_data["ideology_multiclass"].to_list()
    i_mul_w = torch.FloatTensor(
        compute_class_weight(
            class_weight="balanced",
            classes=list(set(class_labels)),
            y=class_labels,
        )
    ).to(DEVICE)

    return gender_w, i_bin_w, i_mul_w


def compute_results(tuned_params: Dict[str, Any], lm: str) -> None:
    model = MultiTaskModel(tuned_params["drop_rate"], lm)
    model.load_state_dict(torch.load("checkpoint.pt"))
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(lm)

    raw_test_data = pd.read_csv(
        Path(__file__).parent / "datasets/politicIT_test_set.csv"
    ).loc[:, ["label", "tweet"]]

    preprocessor = DataPreprocessor(raw_test_data, tokenizer, "it")
    processed_data = preprocessor.prepare_data()

    test_data = TestDataset(df=processed_data, tok=tokenizer)

    batch_size = 8
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    inference(test_dataloader, model)


def main(
    data_es: str,
    data_it: str,
    lm: str,
    lang: Optional[str] = typer.Argument(default=None),
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(lm)
    processed_es, processed_it = None, None
    preprocessor_es, preprocessor_it = None, None

    # process and group tweets from the same authors
    if lang and lang == "es":
        raw_data_es = pd.read_csv(data_es, sep=",").dropna()
        preprocessor_es = DataPreprocessor(raw_data_es, tokenizer, "es")
        processed_es = preprocessor_es.prepare_data()
    elif lang and lang == "it":
        raw_data_it = pd.read_csv(data_it, sep=",").dropna()
        preprocessor_it = DataPreprocessor(raw_data_it, tokenizer, "it")
        processed_it = preprocessor_it.prepare_data()
    else:
        raw_data_es = pd.read_csv(data_es, sep=",").dropna()
        raw_data_it = pd.read_csv(data_it, sep=",").dropna()
        preprocessor_es = DataPreprocessor(raw_data_es, tokenizer, "es")
        processed_es = preprocessor_es.prepare_data()
        preprocessor_it = DataPreprocessor(raw_data_it, tokenizer, "it")
        processed_it = preprocessor_it.prepare_data()

    # TUNE model to get best hyper-parameters
    #########################################

    # split data into train- and validation-set
    if lang and lang == "es" and processed_es is not None:
        train_data_es, val_data = train_test_split(
            processed_es, train_size=0.8, shuffle=True, random_state=42
        )
        train_tune_set = TrainValDataset(df=train_data_es, tok=tokenizer)
        val_set = TrainValDataset(df=val_data, tok=tokenizer)
    elif lang and lang == "it" and processed_it is not None:
        train_data_it, val_data = train_test_split(
            processed_it, train_size=0.8, shuffle=True, random_state=42
        )
        train_tune_set = TrainValDataset(df=train_data_it, tok=tokenizer)
        val_set = TrainValDataset(df=val_data, tok=tokenizer)
    elif processed_es is not None and processed_it is not None:
        train_data_it, val_data = train_test_split(
            processed_it, train_size=0.8, shuffle=True, random_state=42
        )
        train_tune_data = pd.concat([processed_es, train_data_it])
        train_tune_set = TrainValDataset(df=train_tune_data, tok=tokenizer)
        val_set = TrainValDataset(df=val_data, tok=tokenizer)
    else:
        raise ValueError()

    # compute weights to balance data
    gender_w, i_bin_w, i_mul_w = compute_weights(train_tune_set._data)

    # tune hyper-parameters
    tune_parameters(train_tune_set, val_set, gender_w, i_bin_w, i_mul_w, lm)

    # retrieve best hyper-parameters
    tuned_params = json.loads(
        (Path(__file__).parent / "tuning_results.jsonl").read_text()
    )

    # TRAIN model with tuned hyper-parameters
    #########################################

    # prepare training-set according to the given input
    if lang and lang == "es" and processed_es is not None:
        train_set = TrainValDataset(df=processed_es, tok=tokenizer)
    elif lang and lang == "it" and processed_it is not None:
        train_set = TrainValDataset(df=processed_it, tok=tokenizer)
    elif processed_es is not None and processed_it is not None:
        train_data = pd.concat([processed_es, processed_it])
        train_set = TrainValDataset(df=train_data, tok=tokenizer)
    else:
        raise ValueError()

    # compute weights on data used for training
    gender_w, i_bin_w, i_mul_w = compute_weights(train_set._data)

    Trainer(
        epochs=15,
        train_data=train_set,
        gender_weights=gender_w,
        i_bin_weights=i_bin_w,
        i_mul_weights=i_mul_w,
        tuned_params=tuned_params,
        lm=lm,
    ).train()

    compute_results(tuned_params, lm)
    results = pd.read_csv("results.csv")

    # decode predicted gender, i-bin and i-mul labels
    if lang and lang == "es" and preprocessor_es:
        results = preprocessor_es.decode_labels(results)
    elif preprocessor_it:
        results = preprocessor_it.decode_labels(results)
    else:
        raise ValueError()

    results.to_csv("decoded_results.csv", index=False)
    compute_scores()


if __name__ == "__main__":
    typer.run(main)


"""NOTES
[ 1 ]   https://stackoverflow.com/q/57021620
        https://discuss.pytorch.org/t/weights-in-bcewithlogitsloss/27452/10
"""
