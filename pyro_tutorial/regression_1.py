# Running through a Pyro tutorial
# with focus on API for stochastic
# variational inference (SVI). Based on
# tutorial at https://pyro.ai/examples/bayesian_regression.html
# 
# Author: Zhaolei (Henry) Shi -- zshi2@stanford.edu


import os
from functools import partial
from pdb import set_trace

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive



def typical_regression(x_data, y_data, num_iterations):
    # TYPICAL REGRESSION (NON BAYESIAN)

    # Regression model
    linear_reg_model = PyroModule[nn.Linear](3, 1)

    # Define loss and optimize
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)

    def train():
        # run the model forward on the data
        y_pred = linear_reg_model(x_data).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y_data)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        return loss

    for j in range(num_iterations):
        loss = train()
        if (j + 1) % 50 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))

    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in linear_reg_model.named_parameters():
        print(name, param.data.numpy())



def bayesian_regression(x_data, y_data, num_iterations):
    # BAYESIAN REGRESSION WITH SVI

    class BayesianRegression(PyroModule):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = PyroModule[nn.Linear](in_features, out_features)
            self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
            self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

        def forward(self, x, y=None):
            # forward() specifies the data generating process
            sigma = pyro.sample("sigma", dist.Uniform(0., 10.)) # this is the error term (typically called epsilon in regression equations)
            mean = self.linear(x).squeeze(-1)
            with pyro.plate("data", x.shape[0]):
                obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
            return mean

    """
    Guides -- posterior distribution classes

    The guide determines a family of distributions, and SVI aims to find an 
    approximate posterior distribution from this family that has the lowest
    KL divergence from the true posterior.
    """

    model = BayesianRegression(3, 1)

    """
    Under the hood, this defines a guide that uses a Normal distribution with
    learnable parameters corresponding to each sample statement in the model.
    e.g. in our case, this distribution should have a size of (5,) correspoding
    to the 3 regression coefficients for each of the terms, and 1 component
    contributed each by the intercept term and sigma in the model.
    """

    guide = AutoDiagonalNormal(model)

    adam = pyro.optim.Adam({"lr": 0.03}) # note this is from Pyro's optim module, not PyTorch's 
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    """
    We do not need to pass in learnable parameters to the optimizer
    (unlike the PyTorch example above) since that is determined by the guide
    code and happens behind the scenes within the SVI class automatically.
    To take an ELBO gradient step we simply call the step method of SVI.
    The data argument we pass to SVI.step will be passed to both
    model() and guide().
    """

    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(x_data, y_data)
        if (j+1) % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))


    # We can examine the optimized parameter values by fetching from Pyro’s param store.

    guide.requires_grad_(False) # not sure what this does

    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))


    # This gets us quantiles from the posterior distribution
    guide.quantiles([0.25, 0.5, 0.75])

    """
    Since Bayesian models give you a posterior distribution, 
    model evalution needs to be a compbination of sampling the posterior and
    running the samples through the model.

    We generate 800 samples from our trained model. Internally, this is done
    by first generating samples for the unobserved sites in the guide, and
    then running the model forward by conditioning the sites to values sampled
    from the guide. Refer to the Model Serving section for insight on how the
    Predictive class works.
    """

    def summary(samples):
        site_stats = {}
        for k, v in samples.items():
            site_stats[k] = {
                "mean": torch.mean(v, 0),
                "std": torch.std(v, 0),
                "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
                "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
            }
        return site_stats

    """
    Note that in return_sites, we specify both the outcome ("obs" site) as
    well as the return value of the model ("_RETURN") which captures the
    regression line. Additionally, we would also like to capture the regression
    coefficients (given by "linear.weight") for further analysis.
    """

    predictive = Predictive(model, guide=guide, num_samples=800,
                            return_sites=("linear.weight", "obs", "_RETURN"))
    samples = predictive(x_data)
    pred_summary = summary(samples)


if __name__ == '__main__':

    # for CI testing
    smoke_test = ('CI' in os.environ)
    assert pyro.__version__.startswith('1.4.0')
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)
    pyro.enable_validation(True)


    DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
    data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
    df = data[["cont_africa", "rugged", "rgdppc_2000"]]
    df = df[np.isfinite(df.rgdppc_2000)]
    df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

    # PyroModule is a way to append attributes to 
    # existing PyTorch classes
    assert issubclass(PyroModule[nn.Linear], nn.Linear)
    assert issubclass(PyroModule[nn.Linear], PyroModule)


    # Dataset: Add a feature to capture the interaction between "cont_africa" and "rugged"
    df["cont_africa_x_rugged"] = df["cont_africa"] * df["rugged"]

    data = torch.tensor(df[["cont_africa", "rugged", "cont_africa_x_rugged", "rgdppc_2000"]].values,
                            dtype=torch.float)
    x_data, y_data = data[:, :-1], data[:, -1]


    # run regressions
    num_iterations = 1500 if not smoke_test else 2
    # typical_regression(x_data, y_data, num_iterations)
    bayesian_regression(x_data, y_data, num_iterations)


