""" Make predictions using parameters along the learning curve
"""

import logging
import os
import argparse
import pickle
from pdb import set_trace

from experiment import EvalExperiment
from run_experiment import get_candidate_models

from utils import set_logger

def get_param_paths(param_dir, model_name):
    dir_content = os.listdir(param_dir)
    out = [os.path.join(param_dir, x) for x in dir_content if model_name in x]
    return out

def make_prediction(predict_dir, data_name):

    logging.info('Working with parameters of {}.'.format(data_name))
    working_dir = os.path.join(predict_dir, data_name)

    data_path = os.path.join(working_dir, 'data.pkl')
    my_dataset = pickle.load(open(data_path, 'rb'))
    logging.info('Loaded dataset at {}.'.format(data_path))

    logging.info("Setting up the models for prediction.")
    my_models = get_candidate_models()
    # Creating the experiment object to make predictions,
    # not to run experiments
    my_experiment = EvalExperiment(
                        models = [my_models[x] for x in my_models.keys()],
                        model_names = list(my_models.keys()),
                        dataset = my_dataset
                    )

    logging.info("Making predictions.")
    export_dir = os.path.join(working_dir, 'prediction')
    if not os.path.isdir(export_dir):
        logging.info('Creating directory {}.'.format(export_dir))
        os.makedirs(export_dir)

    for model_name in my_models:
        param_dir = os.path.join(working_dir, 'params')
        param_paths = get_param_paths(param_dir, model_name)
        for param_path in param_paths:
            my_experiment.load_param_and_predict(model_name, param_path, export_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_dir', default = 'work_dir/predict', help = 'Path to working directory.', type = str)
    args = parser.parse_args()

    set_logger('log_make_prediction.log')

    make_prediction(args.predict_dir, 'example')
