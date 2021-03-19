"""
Pyro models for person-item response that also
accomodates specifying fixed parameters

Author: Zhaolei (Henry) Shi -- zshi2@stanford.edu
"""

from pdb import set_trace

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim

from pyro_model import sigmoid_torch

"""
 The 2 parameter traditional IRT model
"""

def pred_fn_2param(theta, alpha, beta, stu_id, ques_id):
    theta_i = theta[stu_id]
    alpha_j = alpha[ques_id]
    beta_j = beta[ques_id]
    prob = sigmoid_torch(alpha_j * (theta_i - beta_j))
    return prob

def model_2param(ques_id, stu_id, correct, n_ques = None, n_stu = None, subsample_size = None):
    if not n_ques: n_ques = torch.unique(ques_id).size()[0]
    if not n_stu: n_stu = torch.unique(stu_id).size()[0]

    # link params to models defined in model function
    theta = pyro.sample("theta", dist.Normal(torch.zeros(n_stu), torch.ones(n_stu)).independent(1))
    alpha = pyro.sample("alpha", dist.Normal(torch.zeros(n_ques), torch.ones(n_ques)).independent(1))
    beta = pyro.sample("beta", dist.Normal(torch.zeros(n_ques), torch.ones(n_ques)).independent(1))

    # actual parameters corresponding to data
    prob = pred_fn_2param(theta, alpha, beta, stu_id, ques_id)

    if subsample_size:
        with pyro.plate("data", size = len(correct), subsample_size = subsample_size) as ind:
            pyro.sample("obs", dist.Bernoulli(prob[ind]), obs=correct[ind])
    else:
        with pyro.plate("data", len(correct)):
            pyro.sample("obs", dist.Bernoulli(prob), obs=correct)

def guide_2param(ques_id, stu_id, correct, n_ques = None, n_stu = None, subsample_size = None):
    if not n_ques: n_ques = torch.unique(ques_id).size()[0]
    if not n_stu: n_stu = torch.unique(stu_id).size()[0]

    theta_loc = pyro.param('theta_loc', torch.randn(n_stu, dtype = torch.float64))
    theta_scale = pyro.param('theta_scale', torch.ones(n_stu, dtype = torch.float64),
                               constraint=constraints.positive)

    alpha_loc = pyro.param('alpha_loc', torch.ones(n_ques, dtype = torch.float64),
                               constraint=constraints.positive)
    alpha_scale = pyro.param('alpha_scale', torch.ones(n_ques, dtype = torch.float64),
                               constraint=constraints.positive)

    beta_loc = pyro.param('beta_loc', torch.randn(n_ques, dtype = torch.float64))
    beta_scale = pyro.param('beta_scale', torch.ones(n_ques, dtype = torch.float64),
                               constraint=constraints.positive)

    # link params to models defined in model function
    pyro.sample("theta", dist.Normal(theta_loc, theta_scale).independent(1))
    pyro.sample("alpha", dist.Normal(alpha_loc, alpha_scale).independent(1))
    pyro.sample("beta", dist.Normal(beta_loc, beta_scale).independent(1))

