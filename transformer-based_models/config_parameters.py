from ray import tune

params = {
    "optimizer": {
        "AdamW": {
            "lr": tune.uniform(1e-6, 5e-5),
            "weight_decay": tune.uniform(1e-3, 1e-1),
            "beta1": tune.uniform(5e-1, 9e-1),
            "beta2": tune.uniform(5e-1, 9e-1),
        }
    },
    "drop_rate": tune.uniform(1e-1, 5e-1),
    "batch_size": tune.uniform(8, 9),
    # num. tuning experiments to run
    "n_samples": 10,
    # num. epochs for each experiment
    "epochs": 3,
    # temperature (T) (https://arxiv.org/abs/2002.04792)
    "T": tune.uniform(2e-1, 8e-1)
}
