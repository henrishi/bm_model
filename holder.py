""" Model class to be used with experiments
"""

from pdb import set_trace

import numpy as np

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.util import zero_grads, torch_item

from model_utils import get_train_val_test_vector, TRAIN_INT, VAL_INT, TEST_INT

"""
Model holders
"""

class Model(object):

    def __init__(self):
        """ initialize the model """
        pass

    def train_one_step(self, data_dict):
        """ take one step and return training results (as dict) """
        pass

    def predict(self, data_dict):
        """ make prediction on data_dict input """
        pass

class PyroModel(Model):

    def __init__(self, model, guide, pred_fn = None, lr = 0.05):
        self.model = model
        self.guide = guide
        self.pred_fn = pred_fn
        self.svi =  SVI(
            model,
            guide,
            optim.Adam({"lr": lr}),
            loss=Trace_ELBO()
        )

class Pyro2ParamModel(PyroModel):

    def train_one_step(self, data_dict):
        elbo = self.svi.step(data_dict['ques_id'], data_dict['stu_id'], data_dict['correct'], n_ques = data_dict['n_ques'], n_stu = data_dict['n_stu'])
        res = {'elbo' : elbo}
        return res

    def predict(self, data_dict):
        if not self.pred_fn:
            raise Exception('No prediction function specified.')
        param_store = pyro.get_param_store()
        prob = self.pred_fn(param_store['theta_loc'], param_store['alpha_loc'], param_store['beta_loc'], data_dict['stu_id'], data_dict['ques_id'])
        return prob

class PyroFactorizationModel(PyroModel):

    def __init__(self, model, guide, pred_fn = None, lr = 0.05, n_latent = 5, l2_factor = None):
        super().__init__(model, guide, pred_fn, lr = 0.05)
        self.n_latent = n_latent
        self.l2_factor = l2_factor
        # L2 regularization
        # overriding the parent's SVI
        if self.l2_factor:
            self.svi =  CustomLossSVI(
                model,
                guide,
                optim.Adam({"lr": lr}),
                l2_factor = l2_factor
            )
    
    def train_one_step(self, data_dict):
        elbo = self.svi.step(data_dict['ques_id'], data_dict['stu_id'], data_dict['correct'], self.n_latent, n_ques = data_dict['n_ques'], n_stu = data_dict['n_stu'])
        res = {'elbo' : elbo}
        return res

    def predict(self, data_dict):
        if not self.pred_fn:
            raise Exception('No prediction function specified.')
        param_store = pyro.get_param_store()
        prob = self.pred_fn(param_store['theta_loc'], param_store['alpha_loc'], param_store['beta_loc'], data_dict['stu_id'], data_dict['ques_id'])
        return prob

class PyroHierarchicalModel(PyroFactorizationModel):

    def __init__(self, model, guide, pred_fn = None, lr = 0.05, n_latent = 5, ques_attrib_n_latent = [], l2_factor = None):
        super().__init__(model = model, guide = guide, pred_fn = pred_fn, lr = lr, n_latent = n_latent, l2_factor = l2_factor)
        self.ques_attrib_n_latent = ques_attrib_n_latent
        self.n_ques_attrib = len(ques_attrib_n_latent)

    def train_one_step(self, data_dict):
        elbo = self.svi.step(data_dict['ques_id'], data_dict['stu_id'], data_dict['correct'], self.n_latent, data_dict['ques_attrib_list'], self.ques_attrib_n_latent, n_ques = data_dict['n_ques'], n_stu = data_dict['n_stu'])
        res = {'elbo' : elbo}
        return res

    def predict(self, data_dict):
        if not self.pred_fn:
            raise Exception('No prediction function specified.')
        param_store = pyro.get_param_store()
        ques_hmat_list = [param_store['ques_hmat_{}_loc'.format(i)] for i in range(self.n_ques_attrib)]
        prob = self.pred_fn(
                    theta = param_store['theta_loc'], alpha = param_store['alpha_loc'],
                    beta = param_store['beta_loc'], ques_hmat_list = ques_hmat_list,
                    stu_id = data_dict['stu_id'], ques_id = data_dict['ques_id'],
                    ques_attrib_list = data_dict['ques_attrib_list']
                )
        return prob


"""
Replaces Pyro's SVI class
"""

class CustomLossSVI(object):

    def __init__(self, model, guide, optimizer, l2_factor):
        self.model = model
        self.guide = guide
        # define optimizer and loss function
        self.optimizer = optimizer
        self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        self.l2_factor = l2_factor
        self.l2_term = None

    def step(self, *args, **kwargs):

        # compute loss
        loss = self.loss_fn(self.model, self.guide, *args, **kwargs) + self.compute_l2_term(*args, **kwargs)
        loss.backward()

        tr = poutine.trace(self.guide).get_trace(*args, **kwargs)
        params = [site['value'].unconstrained() for name, site in tr.nodes.items() if site['type'] == 'param']

        # Copied from Pyro SVI source code
        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optimizer(params)

        # zero gradients
        zero_grads(params)

        return torch_item(loss)

    def compute_l2_term(self, *args, **kwargs):
        tr = poutine.trace(self.guide).get_trace(*args, **kwargs)
        params = {name : site['value'].unconstrained() for name, site in tr.nodes.items() if site['type'] == 'param'}

        # only account for theta and alpha for now
        l2_term = params['theta_loc'].square().mean(dim = 1).sum() + \
                  params['alpha_loc'].square().mean(dim = 1).sum()
        l2_term = self.l2_factor * l2_term
        return l2_term

"""
Data holders
"""

class PyroDataset(object):

    def __init__(self, ques_id, stu_id, correct, use_tvt = False, tvt_vector = None, ques_encoder = None, stu_encoder = None):
        self.ques_id = ques_id
        self.stu_id = stu_id
        self.correct = correct
        self.n_ques = torch.unique(ques_id).size()[0]
        self.n_stu = torch.unique(stu_id).size()[0]
        self.stu_encoder = stu_encoder
        self.ques_encoder = ques_encoder

        if use_tvt or (tvt_vector is not None):
            # use_tvt is True if tvt_vector is set,
            # regardless of use_tvt value
            self.tvt_vector = get_train_val_test_vector(n = len(correct), train = 0.8, val = 0.1, test = 0.1) if tvt_vector is None else tvt_vector
            self.use_tvt = True
        else:
            self.use_tvt = False

    def get_use_tvt(self):
        return self.use_tvt

    def get_training_data(self):
        if self.use_tvt:
            out = {
                'ques_id' : self.ques_id[self.tvt_vector == TRAIN_INT],
                'stu_id' : self.stu_id[self.tvt_vector == TRAIN_INT],
                'correct' : self.correct[self.tvt_vector == TRAIN_INT],
                'n_ques' : self.n_ques,
                'n_stu' : self.n_stu,
                'use_tvt' : self.use_tvt,
            }
        else:
            out = {
                'ques_id' : self.ques_id,
                'stu_id' : self.stu_id,
                'correct' : self.correct,
                'n_ques' : self.n_ques,
                'n_stu' : self.n_stu,
                'use_tvt' : self.use_tvt,
            }
        return out

    def get_validation_data(self):
        if not self.use_tvt:
            raise Exception('Dataset has no validation set.')
        else:
            out = {
                'ques_id' : self.ques_id[self.tvt_vector == VAL_INT],
                'stu_id' : self.stu_id[self.tvt_vector == VAL_INT],
                'correct' : self.correct[self.tvt_vector == VAL_INT],
                'n_ques' : self.n_ques,
                'n_stu' : self.n_stu,
                'use_tvt' : self.use_tvt,
            }
        return out

    def get_test_data(self):
        if not self.use_tvt:
            raise Exception('Dataset has no validation set.')
        else:
            out = {
                'ques_id' : self.ques_id[self.tvt_vector == TEST_INT],
                'stu_id' : self.stu_id[self.tvt_vector == TEST_INT],
                'correct' : self.correct[self.tvt_vector == TEST_INT],
                'n_ques' : self.n_ques,
                'n_stu' : self.n_stu,
                'use_tvt' : self.use_tvt,
            }
        return out

class AttribDataset(PyroDataset):

    def __init__(self, ques_id, stu_id, correct, ques_attrib_list, use_tvt = False, tvt_vector = None, ques_encoder = None, stu_encoder = None):
        super().__init__(ques_id = ques_id, stu_id = stu_id, correct = correct, use_tvt = use_tvt, tvt_vector = tvt_vector, ques_encoder = ques_encoder, stu_encoder = stu_encoder)
        self.ques_attrib_list = ques_attrib_list

    def get_training_data(self):
        out = super().get_training_data()
        out['ques_attrib_list'] = self.ques_attrib_list
        return out

    def get_validation_data(self):
        out = super().get_validation_data()
        out['ques_attrib_list'] = self.ques_attrib_list
        return out

    def get_test_data(self):
        out = super().get_test_data()
        out['ques_attrib_list'] = self.ques_attrib_list
        return out
