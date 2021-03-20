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

class TwoPartTensor(object):

    def __init__(self, key_one, key_two, tensor_one, tensor_two):
        assert(all([x not in key_two for x in key_one]))
        self.key_one = key_one
        self.key_two = key_two
        self.key_all = torch.cat([key_one, key_two]).tolist()
        self.tensor_one = tensor_one
        self.tensor_two = tensor_two
        self.tensor_all = torch.cat([tensor_one, tensor_two])
        self.index_map = {key : n for n, key in enumerate(self.key_all)}

    def __getitem__(self, key):
        index = [self.index_map[x] for x in key.detach().numpy()]
        return self.tensor_all[index]


def pred_fn_2param(theta, alpha, beta, stu_id, ques_id):
    theta_i = theta[stu_id]
    alpha_j = alpha[ques_id]
    beta_j = beta[ques_id]
    prob = sigmoid_torch(alpha_j * (theta_i - beta_j))
    return prob

def model_2param(ques_id, stu_id, correct, fixed_ques_params, fixed_stu_params, subsample_size = None):

    fixed_stu_id = fixed_stu_params['encoded_id']
    fixed_theta_loc = fixed_stu_params['params']['theta_loc']
    fixed_ques_id = fixed_ques_params['encoded_id']
    fixed_alpha_loc = fixed_ques_params['params']['alpha_loc']
    fixed_beta_loc = fixed_ques_params['params']['beta_loc']

    n_trainable_ques = fixed_ques_params['n_trainable']
    trainable_ques_id = fixed_ques_params['trainable_id']
    n_trainable_stu = fixed_stu_params['n_trainable']
    trainable_stu_id = fixed_stu_params['trainable_id']

    # link params to models defined in model function
    theta = pyro.sample("theta", dist.Normal(torch.zeros(n_trainable_stu), torch.ones(n_trainable_stu)).independent(1))
    alpha = pyro.sample("alpha", dist.Normal(torch.zeros(n_trainable_ques), torch.ones(n_trainable_ques)).independent(1))
    beta = pyro.sample("beta", dist.Normal(torch.zeros(n_trainable_ques), torch.ones(n_trainable_ques)).independent(1))

    # combining fixed with trainable parameters
    theta_combined = TwoPartTensor(trainable_stu_id, fixed_stu_id, theta, fixed_theta_loc)
    alpha_combined = TwoPartTensor(trainable_ques_id, fixed_ques_id, alpha, fixed_alpha_loc)
    beta_combined = TwoPartTensor(trainable_ques_id, fixed_ques_id, beta, fixed_beta_loc)

    # actual parameters corresponding to data
    prob = pred_fn_2param(theta_combined, alpha_combined, beta_combined, stu_id, ques_id)

    if subsample_size:
        with pyro.plate("data", size = len(correct), subsample_size = subsample_size) as ind:
            pyro.sample("obs", dist.Bernoulli(prob[ind]), obs=correct[ind])
    else:
        with pyro.plate("data", len(correct)):
            pyro.sample("obs", dist.Bernoulli(prob), obs=correct)

def guide_2param(ques_id, stu_id, correct, fixed_ques_params, fixed_stu_params, subsample_size = None):
    n_trainable_ques = fixed_ques_params['n_trainable']
    n_trainable_stu = fixed_stu_params['n_trainable']

    theta_loc = pyro.param('theta_loc', torch.randn(n_trainable_stu, dtype = torch.float64))
    theta_scale = pyro.param('theta_scale', torch.ones(n_trainable_stu, dtype = torch.float64),
                               constraint=constraints.positive)

    alpha_loc = pyro.param('alpha_loc', torch.ones(n_trainable_ques, dtype = torch.float64),
                               constraint=constraints.positive)
    alpha_scale = pyro.param('alpha_scale', torch.ones(n_trainable_ques, dtype = torch.float64),
                               constraint=constraints.positive)

    beta_loc = pyro.param('beta_loc', torch.randn(n_trainable_ques, dtype = torch.float64))
    beta_scale = pyro.param('beta_scale', torch.ones(n_trainable_ques, dtype = torch.float64),
                               constraint=constraints.positive)

    # link params to models defined in model function
    pyro.sample("theta", dist.Normal(theta_loc, theta_scale).independent(1))
    pyro.sample("alpha", dist.Normal(alpha_loc, alpha_scale).independent(1))
    pyro.sample("beta", dist.Normal(beta_loc, beta_scale).independent(1))

