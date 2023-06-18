import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from torch import nn, sigmoid, softmax
from torch.utils.data import DataLoader


def validate(dataloader: DataLoader, model: nn.Module, iteration: int) -> float:
    model.eval()

    intermediate_results = {}
    output = {
        "author": [],
        "true_gender": [],
        "true_i_bin": [],
        "true_i_mul": [],
        "pred_gender": [],
        "pred_i_bin": [],
        "pred_i_mul": [],
    }

    for batch_labels, tokenized in dataloader:
        gender_logits, i_bin_logits, i_mul_logits = model(tokenized).values()
        zipped = zip(*batch_labels.values(), gender_logits, i_bin_logits, i_mul_logits)

        for (
            gender,
            i_bin,
            i_mul,
            author,
            num_tokens,
            gender_logit,
            i_bin_logit,
            i_mul_logits,
        ) in zipped:
            pred_gender = sigmoid(gender_logit).item()
            pred_i_bin = sigmoid(i_bin_logit).item()
            pred_i_mul = softmax(i_mul_logits, dim=0).tolist()

            if author not in intermediate_results.keys():
                intermediate_results[author] = {
                    "target_gender": gender.item(),
                    "target_i_bin": i_bin.item(),
                    "target_i_mul": i_mul.item(),
                    "num_tokens": [num_tokens.tolist()],
                    "pred_gender": [pred_gender],
                    "pred_i_bin": [pred_i_bin],
                    "pred_i_mul": [pred_i_mul],
                }
            else:
                intermediate_results[author]["num_tokens"].append(num_tokens.tolist())
                intermediate_results[author]["pred_gender"].append(pred_gender)
                intermediate_results[author]["pred_i_bin"].append(pred_i_bin)
                intermediate_results[author]["pred_i_mul"].append(pred_i_mul)

    for author, values in intermediate_results.items():
        output["author"].append(author)
        output["true_gender"].append(values["target_gender"])
        output["true_i_bin"].append(values["target_i_bin"])
        output["true_i_mul"].append(values["target_i_mul"])

        if len(values["num_tokens"]) > 1:
            w = np.array(values["num_tokens"]) / sum(values["num_tokens"])

            pred_gender_per_block = np.array(values["pred_gender"])
            weighted_pred_gender = pred_gender_per_block * w
            pred_gender = int(np.where(weighted_pred_gender.sum(axis=0) > 0.5, 1, 0))
            output["pred_gender"].append(pred_gender)

            pred_ibin_per_block = np.array(values["pred_i_bin"])
            weighted_pred_ibin = pred_ibin_per_block * w
            pred_i_bin = int(np.where(weighted_pred_ibin.sum(axis=0) > 0.5, 1, 0))
            output["pred_i_bin"].append(pred_i_bin)

            pred_imul_per_block = np.array(values["pred_i_mul"]).reshape(-1, 4)
            reshaped_w = np.expand_dims(w, axis=0).T
            sum_weighted_pred_imul = (pred_imul_per_block * reshaped_w).sum(axis=0)
            pred_i_mul = int(np.argmax(sum_weighted_pred_imul, axis=0))
            output["pred_i_mul"].append(pred_i_mul)
        else:
            pred_gender = int(np.where(values["pred_gender"][0] > 0.5, 1, 0))
            output["pred_gender"].append(pred_gender)

            pred_i_bin = int(np.where(values["pred_i_bin"][0] > 0.5, 1, 0))
            output["pred_i_bin"].append(pred_i_bin)

            pred_i_mul = np.array(values["pred_i_mul"]).reshape(-1, 4)
            pred_i_mul = int(np.argmax(pred_i_mul, axis=1).item())
            output["pred_i_mul"].append(pred_i_mul)

    df = pd.DataFrame.from_dict(output)

    f1_gender = f1_score(
        y_true=df["true_gender"].to_numpy(),
        y_pred=df["pred_gender"].to_numpy(),
        average="macro",
    )
    f1_i_bin = f1_score(
        y_true=df["true_i_bin"].to_numpy(),
        y_pred=df["pred_i_bin"].to_numpy(),
        average="macro",
    )
    f1_i_mul = f1_score(
        y_true=df["true_i_mul"].to_numpy(),
        y_pred=df["pred_i_mul"].to_numpy(),
        average="macro",
    )

    tot = f1_gender + f1_i_bin + f1_i_mul

    print(
        (
            f"Validation | epoch: {iteration} | "
            f"gender: {f1_gender} | "
            f"i_bin: {f1_i_bin} | "
            f"i_mul: {f1_i_mul} | "
            f"tot. {tot}\n"
        )
    )
    total_f1 = float(f1_gender + f1_i_bin + f1_i_mul)

    return total_f1
