import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def main() -> None:
    
    res = pd.read_csv("decoded_results.csv", sep=",")
    res.sort_values("label")

    gold = pd.read_csv("test_with_labels.csv", sep=",")
    gold.drop_duplicates(subset=["label"], inplace=True)
    gold.sort_values("label")

    # F1
    f1_gender = f1_score(
        y_pred=res.loc[:, "gender"].to_numpy(),
        y_true=gold.loc[:, "gender"].to_numpy(),
        average="macro"
    )
    f1_ibin = f1_score(
        y_pred=res.loc[:, "ideology_binary"].to_numpy(),
        y_true=gold.loc[:, "ideology_binary"].to_numpy(),
        average="macro"
    )
    f1_imul = f1_score(
        y_pred=res.loc[:, "ideology_multiclass"].to_numpy(),
        y_true=gold.loc[:, "ideology_multiclass"].to_numpy(),
        average="macro"
    )

    # Recall
    recall_gender = recall_score(
        y_pred=res.loc[:, "gender"].to_numpy(),
        y_true=gold.loc[:, "gender"].to_numpy(),
        average="macro"
    )
    recall_ibin = recall_score(
        y_pred=res.loc[:, "ideology_binary"].to_numpy(),
        y_true=gold.loc[:, "ideology_binary"].to_numpy(),
        average="macro"
    )
    recall_imul = recall_score(
        y_pred=res.loc[:, "ideology_multiclass"].to_numpy(),
        y_true=gold.loc[:, "ideology_multiclass"].to_numpy(),
        average="macro"
    )

    # Precision
    prec_gender = precision_score(
        y_pred=res.loc[:, "gender"].to_numpy(),
        y_true=gold.loc[:, "gender"].to_numpy(),
        average="macro"
    )
    prec_ibin = precision_score(
        y_pred=res.loc[:, "ideology_binary"].to_numpy(),
        y_true=gold.loc[:, "ideology_binary"].to_numpy(),
        average="macro"
    )
    prec_imul = precision_score(
        y_pred=res.loc[:, "ideology_multiclass"].to_numpy(),
        y_true=gold.loc[:, "ideology_multiclass"].to_numpy(),
        average="macro"
    )



    print(
        f"f1 gender: {f1_gender * 100}\n"
        f"f1 ibin: {f1_ibin * 100}\n"
        f"f1 imul: {f1_imul * 100}\n\n"
        f"recall gender: {recall_gender * 100}\n"
        f"recall ibin: {recall_ibin * 100}\n"
        f"recall imul: {recall_imul * 100}\n\n"
        f"precision gender: {prec_gender * 100}\n"
        f"precision ibin: {prec_ibin * 100}\n"
        f"precision imul: {prec_imul * 100}\n"
    )
if __name__ == "__main__":
    main()
