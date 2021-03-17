""" Run models for prediction sections of paper
"""

import logging
import os
import pickle
import argparse
from pdb import set_trace

import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import pyro

import pyro_model
from dataset import create_dataset
from holder import Pyro2ParamModel, PyroFactorizationModel, PyroHierarchicalModel, PyroDataset
from experiment import EvalExperiment

from utils import set_logger

def get_candidate_models():
    return {
        'Pyro2Param' : Pyro2ParamModel(model = pyro_model.model_2param, guide = pyro_model.guide_2param, pred_fn = pyro_model.pred_fn_2param),
        'PyroFactor_l3' : PyroFactorizationModel(model = pyro_model.model_factorization, guide = pyro_model.guide_factorization, n_latent = 3, pred_fn = pyro_model.pred_fn_factorization),
        'PyroFactor_l5' : PyroFactorizationModel(model = pyro_model.model_factorization, guide = pyro_model.guide_factorization, n_latent = 5, pred_fn = pyro_model.pred_fn_factorization),
        'PyroFactor_l10' : PyroFactorizationModel(model = pyro_model.model_factorization, guide = pyro_model.guide_factorization, n_latent = 10, pred_fn = pyro_model.pred_fn_factorization),
        'PyroFactor_l20' : PyroFactorizationModel(model = pyro_model.model_factorization, guide = pyro_model.guide_factorization, n_latent = 20, pred_fn = pyro_model.pred_fn_factorization),
        'PyroHier_l3_ql2' : PyroHierarchicalModel(model = pyro_model.model_hierarchical, guide = pyro_model.guide_hierarchical, n_latent = 3, ques_attrib_n_latent = [2], pred_fn = pyro_model.pred_fn_hierarchical, l2_factor = 0.1),
    }


def run_experiment(data_file_path, data_name, use_tvt = True):

    logging.info('Working on estimations with {} from {}.'.format(data_name, data_file_path))

    export_dir = os.path.join(output_dir, data_name)
    if not os.path.isdir(export_dir):
        logging.info('Creating directory {}.'.format(export_dir))
        os.makedirs(export_dir)

    logging.info("Processing the data.")
    input_df = pd.read_feather(data_file_path)
    
    my_dataset = create_dataset(input_df, use_tvt)
    data_export_path = os.path.join(export_dir, 'data.pkl')
    pickle.dump(my_dataset, open(data_export_path, 'wb'))
    logging.info('Wrote dataset to {}.'.format(data_export_path))

    logging.info("Setting up the models for experiment.")
    my_models = get_candidate_models()
    my_experiment = EvalExperiment(
                        models = [my_models[x] for x in my_models.keys()],
                        model_names = list(my_models.keys()),
                        dataset = my_dataset,
                        req_freq = 10,
                        max_iter = 2000 if not smoke_test else 2,
                        record_param_freq = 10
                    )

    logging.info("Running experiment.")
    my_experiment.train_and_save(export_dir = export_dir, auto_stop = True, stop_threshold = 0.0005)

    logging.info("Making predictions.")
    my_experiment.make_prediction(export_dir = export_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', default = 'work_dir', help = 'Path to working directory.', type = str)
    args = parser.parse_args()

    data_dir = os.path.join(args.work_dir, 'data')
    output_dir = os.path.join(args.work_dir, 'predict')

    pyro.enable_validation(True)
    pyro.set_rng_seed(289012)
    torch.manual_seed(1299)
    np.random.seed(1299)

    smoke_test = False

    set_logger('log_run_experiment.log')

    run_experiment(os.path.join(data_dir, 'example.feather'), 'example', use_tvt = True)
    