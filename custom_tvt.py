""" Customized assignment to train, validation, test sets
"""

import logging
import os
from pdb import set_trace
from math import ceil

import numpy as np
import pandas as pd

import torch
import pyro

from model_utils import TRAIN_INT, VAL_INT, TEST_INT, get_train_val_test_vector

def at_least_one(probs, values, n):
    """
    Assign vector of length n to values with probs, but make
    sure at least 1 of each value is retained

    probs : list of probabilities
    values : list of corresponding values to be sampled
    n : length of output vector
    """

    assert len(values) <= n
    assert np.abs(np.sum(np.array(probs)) - 1) < 1e-7    # adds up to 1.

    out = []
    sorted_pairs = sorted(zip(probs, values), key = lambda x: x[0])
    for p, v in sorted_pairs:
        out.extend([v] * ceil(p * n))
    out = out[0:n]  # make sure the output is longer than n
                    # values with the largest probs gets
                    # cut off
    out = np.random.choice(out, size = n, replace = False).tolist()
    for v in values:
        assert v in out

    return out

def get_overlapping_tvt_data(input_df):

    # Keeping only questions with 
    # at least 3 exposures (to different students)
    by_ques = (
            input_df.
                groupby(['question_id']).
                agg({
                    'student_id': 'nunique',
                    'correct': 'count',
                }).rename(
                    columns = {
                        'student_id' : 'n_stu',
                        "correct" : 'n_obs'
                    }
                )
        )

    ques_pop = by_ques[by_ques['n_stu'] >= 3].index.values

    pop = input_df.merge(
            pd.DataFrame({'question_id' : ques_pop}),
            how = 'inner'
          )

    # Assign tvt within question_id
    def add_tvt_col(df):
        out = df.copy()
        out['tvt_vector'] = at_least_one(probs = [0.8, 0.1, 0.1], values = [TRAIN_INT, VAL_INT, TEST_INT], n = df.shape[0])
        return(out)

    pop = pop.groupby(['question_id']).apply(add_tvt_col)
    pop = pop.reset_index(drop = True)

    tvt = torch.tensor(pop['tvt_vector'].copy())

    return pop, tvt

def get_byhw_tvt_data(input_df):

    # Random selection by homework_id * student_id
    assert 'homeworkid' in input_df.columns

    sampling_pop = input_df[['homeworkid', 'student_id']].drop_duplicates(ignore_index = True)
    sampling_pop['tvt_vector'] = get_train_val_test_vector(n = sampling_pop.shape[0], train = 0.8, val = 0.1, test = 0.1)

    pop = input_df.merge(sampling_pop, how = 'inner')
    tvt = torch.tensor(pop['tvt_vector'].copy())

    return pop, tvt
