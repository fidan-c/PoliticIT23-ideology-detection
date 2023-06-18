
## Environment setup
This project is built with [Poetry](https://python-poetry.org/). To set up an environment follow the steps below:

```shell
# install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# configure poetry
poetry config virtualenvs.in-project true

# install dependencies and create virtual environment
cd transformer-based_models
poetry install --no-root

# manually activate environment (VS Code allows to configure this automatically)
source .venv/bin/activate
```

## Data
In the `datasets` folder we provide the training and test sets to run the experiments. The development files can be used to perform a quick test-run of the whole pipeline.

```text
datasets
│   development_es.csv
│   development_it.csv
│   politicES_dataset.csv
│   politicIT_train_set.csv
│   politicIT_test_set.csv
│   politicIT_test_set_with_labels.csv
```

## Running the experiments
Run the snippets below to replicate our experiments and train the four transformer-based models: `XLM-R (it-es)`, `XLM-R (it)`, `XLM-R (es)`, `BERT (it)`

```shell
# train XLM-R (it-es)
python main.py datasets/politicES_dataset.csv datasets/politicIT_train_set.csv xlm-roberta-base

# train XLM-R (it)
python main.py datasets/politicES_dataset.csv datasets/politicIT_train_set.csv xlm-roberta-base it

# train XLM-R (es)
python main.py datasets/politicES_dataset.csv datasets/politicIT_train_set.csv xlm-roberta-base es

# train BERT (it)
python main.py datasets/politicES_dataset.csv datasets/politicIT_train_set.csv dbmdz/bert-base-italian-cased it
```

## Hyper-parameters
In the table below we summarize the best hyper-parameter values obtained for the four transformer-based models described in our contribution.

| Parameters    | XLM-R (it-es)         | XLM-R (it)            | XLM-R (es)            | BERT (it)            |
| :---          |     :---:             |     :---:             |     :---:             |     :---:            |
| `lr`          | $1.56 \times 10^{-5}$ | $1.53 \times 10^{-5}$ | $2.04 \times 10^{-5}$ |$1.81 \times 10^{-5}$ |
| `weight_decay`| $0.07$                | $0.06$                | $0.004$               | $0.05$               |
| `beta1`       | $0.69$                | $0.71$                | $0.59$                | $0.80$               |
| `beta2`       | $0.88$                | $0.67$                | $0.89$                | $0.82$               |
| `dropout`     | $0.46$                | $0.22$                | $0.36$                | $0.41$               |
| `batch size`  | $16$                  | $16$                  | $16$                  | $16$                 |
| `T`           | $0.46$                | $0.31$                | $0.53$                | $0.80$               |

## Results
In the `results` folder we provide the outputs obtained for each model and their respective best hyper-parameters.

