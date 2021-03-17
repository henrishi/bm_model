# Running through a Pyro tutorial
# with focus on low leve API
# https://pyro.ai/examples/bayesian_regression_ii.html
# 
# Author: Zhaolei (Henry) Shi -- zshi2@stanford.edu

import logging
import os
from pdb import set_trace

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO, Predictive

pyro.set_rng_seed(1)
assert pyro.__version__.startswith('1.4.0')

def model(is_cont_africa, ruggedness, log_gdp):
    a = pyro.sample("a", dist.Normal(0., 10.))
    b_a = pyro.sample("bA", dist.Normal(0., 1.))
    b_r = pyro.sample("bR", dist.Normal(0., 1.))
    b_ar = pyro.sample("bAR", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
    with pyro.plate("data", len(ruggedness)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)

def guide(is_cont_africa, ruggedness, log_gdp):
    a_loc = pyro.param('a_loc', torch.tensor(0.))
    a_scale = pyro.param('a_scale', torch.tensor(1.),
                         constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.),
                             constraint=constraints.positive)
    weights_loc = pyro.param('weights_loc', torch.randn(3))
    weights_scale = pyro.param('weights_scale', torch.ones(3),
                               constraint=constraints.positive)

    pyro.sample("a", dist.Normal(a_loc, a_scale))
    pyro.sample("bA", dist.Normal(weights_loc[0], weights_scale[0]))
    pyro.sample("bR", dist.Normal(weights_loc[1], weights_scale[1]))
    pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
    pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))

# Utility function to print latent sites' quantile information.
def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats



if __name__ == '__main__':


    logging.basicConfig(format='%(message)s', level=logging.INFO)
    # Enable validation checks
    pyro.enable_validation(True)
    smoke_test = ('CI' in os.environ)
    pyro.set_rng_seed(1)
    DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
    rugged_data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")

    # Prepare training data
    df = rugged_data[["cont_africa", "rugged", "rgdppc_2000"]]
    df = df[np.isfinite(df.rgdppc_2000)]
    df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
    train = torch.tensor(df.values, dtype=torch.float)

    svi = SVI(model,
          guide,
          optim.Adam({"lr": .05}),
          loss=Trace_ELBO())

    is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]
    pyro.clear_param_store()
    num_iters = 5000 if not smoke_test else 2
    for i in range(num_iters):
        elbo = svi.step(is_cont_africa, ruggedness, log_gdp)
        if i % 500 == 0:
            logging.info("Elbo loss: {}".format(elbo))


    # asessing the posterior distribution
    num_samples = 1000
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    svi_samples = {k: v.reshape(num_samples).detach().cpu().numpy()
                   for k, v in predictive(log_gdp, is_cont_africa, ruggedness).items()
                   if k != "obs"}

    for site, values in summary(svi_samples).items():
        print("Site: {}".format(site))
        print(values, "\n")
