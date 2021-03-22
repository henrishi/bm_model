# Running through a Pyro tutorial
# with focus on low-level API for 
# inference: http://pyro.ai/examples/intro_part_ii.html

import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace

import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)


def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(weight, 0.75))

def scale_parametrized_guide(guess):
    a = pyro.param("a", torch.tensor(1.))
    b = pyro.param("b", torch.tensor(1.))
    return pyro.sample("weight", dist.Normal(a, torch.abs(b)))

if __name__ == '__main__':

    conditioned_scale = pyro.condition(scale, data={"measurement": 9.5})

    guess = 8.5

    pyro.clear_param_store()
    svi = pyro.infer.SVI(model=conditioned_scale,
                         guide=scale_parametrized_guide,
                         optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                         loss=pyro.infer.Trace_ELBO())


    losses, a,b  = [], [], []
    num_steps = 2500
    for t in range(num_steps):
        losses.append(svi.step(guess))
        a.append(pyro.param("a").item())
        b.append(pyro.param("b").item())
