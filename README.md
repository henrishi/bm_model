# Bayesian Measurement Models for Online Sparse Settings


The rise of popular mobile education applications produced data where a large number of students answers a small subset of items in a large question bank. Traditional linking approaches from the education measurement literature cannot scale to this context.

We propose and evaluate a set of models designed to overcome traditional limitations. The models take advantage of factorization techniques and Bayesian variational inference to meet the needs of this context.


# Introduction

Our code base leverage [Pyro](https://github.com/pyro-ppl/pyro)'s infrastructure for Bayesian stochastic variational inference. 

We implement the following models, Y_ij = 1 denotes that student i responded correctly to question j:

1. 2param model: p( Y_ij = 1 | theta_i, alpha_i, beta_j) = 1 / (1 + exp(- alpha_j * (theta_i - beta_j))
2. Factorization model: p(Y_ij = 1 | theta_i, delta_j, beta_j) = 1 / (1 + exp(- (inner_prod(theta_i, delta_j)  - beta_j))
3. Hierarchical model: similar to the factorization model, but now delta_j is replaced with the concatenation of 1) a vector of trainable parameters, 2) one or more matmul(H, X_j), the matrix transformations of observed question characteristics. H is a matrix of trainable parameters.

# Code structure

* `pyro_model.py` implements the models and the likelihood functions.
* `holder.py` wraps models in Model classes to allow for easy training and prediction, also contains classes for data loaders.
* `dataset.py` implements functions to parse the dataset and return the appropriate data loader.
* `experiment.py` implements the experiment classes to automate training, auto-stop after convergence, and evaluation.
* `custom_tvt.py` implements customized ways to divide the dataset into training, validation, and test sets (e.g. ensuring any question will show up in all three sets).
* `make_prediction.py` creates predictions for learning curve analysis.
* `run_experiment.py` creates models and data loaders and runs training and prediction.

# Running an example

You can run the model on a small dataset using the following line in a terminal:

```
python run_experiment.py
```

