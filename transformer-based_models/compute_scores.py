import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_scores() -> None:
    res = pd.read_csv("decoded_results.csv", sep=",")
    res.sort_values("label")

    gold = pd.read_csv("datasets/politicIT_test_set_with_labels.csv", sep=",")
    gold.drop_duplicates(subset=["label"], inplace=True)
    gold.sort_values("label")

    # F1
    f1_gender = f1_score(
        y_pred=res.loc[:, "gender"].to_numpy(),
        y_true=gold.loc[:, "gender"].to_numpy(),
        average="macro",
    )
    f1_ibin = f1_score(
        y_pred=res.loc[:, "ideology_binary"].to_numpy(),
        y_true=gold.loc[:, "ideology_binary"].to_numpy(),
        average="macro",
    )
    f1_imul = f1_score(
        y_pred=res.loc[:, "ideology_multiclass"].to_numpy(),
        y_true=gold.loc[:, "ideology_multiclass"].to_numpy(),
        average="macro",
    )

    # Recall
    recall_gender = recall_score(
        y_pred=res.loc[:, "gender"].to_numpy(),
        y_true=gold.loc[:, "gender"].to_numpy(),
        average="macro",
    )
    recall_ibin = recall_score(
        y_pred=res.loc[:, "ideology_binary"].to_numpy(),
        y_true=gold.loc[:, "ideology_binary"].to_numpy(),
        average="macro",
    )
    recall_imul = recall_score(
        y_pred=res.loc[:, "ideology_multiclass"].to_numpy(),
        y_true=gold.loc[:, "ideology_multiclass"].to_numpy(),
        average="macro",
    )

    # Precision
    prec_gender = precision_score(
        y_pred=res.loc[:, "gender"].to_numpy(),
        y_true=gold.loc[:, "gender"].to_numpy(),
        average="macro",
    )
    prec_ibin = precision_score(
        y_pred=res.loc[:, "ideology_binary"].to_numpy(),
        y_true=gold.loc[:, "ideology_binary"].to_numpy(),
        average="macro",
    )
    prec_imul = precision_score(
        y_pred=res.loc[:, "ideology_multiclass"].to_numpy(),
        y_true=gold.loc[:, "ideology_multiclass"].to_numpy(),
        average="macro",
    )

    print(
        f"f1 gender: {f1_gender}\n"
        f"f1 ibin: {f1_ibin}\n"
        f"f1 imul: {f1_imul}\n\n"
        f"recall gender: {recall_gender}\n"
        f"recall ibin: {recall_ibin}\n"
        f"recall imul: {recall_imul}\n\n"
        f"precision gender: {prec_gender}\n"
        f"precision ibin: {prec_ibin}\n"
        f"precision imul: {prec_imul}\n"
    )
