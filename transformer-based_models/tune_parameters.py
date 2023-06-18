import json
from pathlib import Path

import torch
from ray.air import CheckpointConfig, RunConfig
from ray.tune import TuneConfig, Tuner, with_parameters, with_resources
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from torch.utils.data import Dataset

from config_parameters import params
from tune_trainer import Trainer


def tune_parameters(
    train_data: Dataset,
    val_data: Dataset,
    gender_w: torch.Tensor,
    i_bin_w: torch.Tensor,
    i_mul_w: torch.Tensor,
    lm: str,
) -> None:
    config = params

    # set up and run tuning for current model and config
    my_checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="f1_score",
        checkpoint_score_order="max",
    )

    my_run_config = RunConfig(
        stop={"training_iteration": config["epochs"]},
        checkpoint_config=my_checkpoint_config,
        name="tweet_classification",
        local_dir=str(Path(__file__).parent / f"ray_checkpoints"),
        verbose=0,
    )

    # [ 1 ]
    bayesopt = BayesOptSearch(
        patience=3,
        skip_duplicate=True,
        metric="f1_score",
        mode="max",
        # https://github.com/fmfn/BayesianOptimization/blob/cf13889bdcb8a1a9013a9904895011fac7268e8b/bayes_opt/util.py#L97
        utility_kwargs={"kappa": 2.5, "kappa_decay": 1, "kappa_decay_delay": 0},
        # https://github.com/ray-project/ray/issues/13962#issuecomment-890029380
        random_search_steps=3,
    )

    asha_scheduler = ASHAScheduler(
        time_attr="training_iteration",
        grace_period=3,
        reduction_factor=3,
        brackets=1,
    )

    my_tune_config = TuneConfig(
        search_alg=bayesopt,
        scheduler=asha_scheduler,
        metric="f1_score",
        mode="max",
        num_samples=config["n_samples"],
        reuse_actors=False,
        max_concurrent_trials=None,
    )

    tuner = Tuner(
        trainable=with_resources(
            with_parameters(
                Trainer,
                train_data=train_data,
                val_data=val_data,
                gender_weights=gender_w,
                i_bin_weights=i_bin_w,
                i_mul_weights=i_mul_w,
                lm=lm,
            ),
            {"cpu": 1, "gpu": 1},
        ),
        run_config=my_run_config,
        tune_config=my_tune_config,
        param_space=config,
    )

    results = tuner.fit().get_best_result()

    # store best parameters and path to model checkpoint
    with open(Path(__file__).parent / f"tuning_results.jsonl", "a") as f:
        best_params = results.config
        checkpoint = Path(results.checkpoint._local_path) / "model.pt"  # type: ignore
        best_params["checkpoint"] = str(checkpoint)  # type: ignore
        json.dump(best_params, f)
        f.write("\n")


""" NOTES
[ 1 ]   https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.bayesopt.BayesOptSearch.html#ray-tune-search-bayesopt-bayesoptsearch
"""
