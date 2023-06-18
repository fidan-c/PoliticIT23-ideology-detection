from pathlib import Path

import numpy as np
import pandas as pd
from torch import nn, sigmoid, softmax
from torch.utils.data import DataLoader


def inference(dataloader: DataLoader, model: nn.Module) -> None:
    model.eval()

    intermediate_results = {}
    output = {
        "label": [],
        "gender": [],
        "ideology_binary": [],
        "ideology_multiclass": [],
    }

    for batch_labels, tokenized in dataloader:
        gender_logits, i_bin_logits, i_mul_logits = model(tokenized).values()
        zipped = zip(*batch_labels.values(), gender_logits, i_bin_logits, i_mul_logits)

        for label, num_tokens, gender_logit, i_bin_logit, i_mul_logits in zipped:
            pred_gender = sigmoid(gender_logit).item()
            pred_i_bin = sigmoid(i_bin_logit).item()
            pred_i_mul = softmax(i_mul_logits, dim=0).tolist()

            if label not in intermediate_results.keys():
                intermediate_results[label] = {
                    "num_tokens": [num_tokens.tolist()],
                    "gender": [pred_gender],
                    "ideology_binary": [pred_i_bin],
                    "ideology_multiclass": [pred_i_mul],
                }
            else:
                intermediate_results[label]["num_tokens"].append(num_tokens.tolist())
                intermediate_results[label]["gender"].append(pred_gender)
                intermediate_results[label]["ideology_binary"].append(pred_i_bin)
                intermediate_results[label]["ideology_multiclass"].append(pred_i_mul)

    for label, values in intermediate_results.items():
        output["label"].append(label)

        if len(values["num_tokens"]) > 1:
            w = np.array(values["num_tokens"]) / sum(values["num_tokens"])

            pred_gender_per_block = np.array(values["gender"])
            weighted_pred_gender = pred_gender_per_block * w
            pred_gender = int(np.where(weighted_pred_gender.sum(axis=0) > 0.5, 1, 0))
            output["gender"].append(pred_gender)

            pred_ibin_per_block = np.array(values["ideology_binary"])
            weighted_pred_ibin = pred_ibin_per_block * w
            pred_i_bin = int(np.where(weighted_pred_ibin.sum(axis=0) > 0.5, 1, 0))
            output["ideology_binary"].append(pred_i_bin)

            pred_imul_per_block = np.array(values["ideology_multiclass"]).reshape(-1, 4)
            reshaped_w = np.expand_dims(w, axis=0).T
            sum_weighted_pred_imul = (pred_imul_per_block * reshaped_w).sum(axis=0)
            pred_i_mul = int(np.argmax(sum_weighted_pred_imul, axis=0))
            output["ideology_multiclass"].append(pred_i_mul)

        else:
            pred_gender = int(np.where(values["gender"][0] > 0.5, 1, 0))
            output["gender"].append(pred_gender)

            pred_i_bin = int(np.where(values["ideology_binary"][0] > 0.5, 1, 0))
            output["ideology_binary"].append(pred_i_bin)

            pred_i_mul = np.array(values["ideology_multiclass"]).reshape(-1, 4)
            pred_i_mul = int(np.argmax(pred_i_mul, axis=1).item())
            output["ideology_multiclass"].append(pred_i_mul)

    df = pd.DataFrame.from_dict(output)

    # Adjust multi-class results according to binary ones
    #####################################################

    # if ibin==left and imul==moderate_right set imul to moderate_left
    df.loc[
        (df["ideology_binary"] == "left")
        & (df["ideology_multiclass"] == "moderate_right"),
        "ideology_multiclass",
    ] = "moderate_left"

    # if ibin==left and imul==right set imul to left
    df.loc[
        (df["ideology_binary"] == "left") & (df["ideology_multiclass"] == "right"),
        "ideology_multiclass",
    ] = "left"

    # if ibin==right and imul==moderate_left set imul to moderate_right
    df.loc[
        (df["ideology_binary"] == "right")
        & (df["ideology_multiclass"] == "moderate_left"),
        "ideology_multiclass",
    ] = "moderate_right"

    # if ibin==right and imul==left set imul to right
    df.loc[
        (df["ideology_binary"] == "right") & (df["ideology_multiclass"] == "left"),
        "ideology_multiclass",
    ] = "right"

    df.to_csv(Path(__file__).parent / "results.csv", index=False)
