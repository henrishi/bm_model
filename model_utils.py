# Utility functions for running models
import numpy as np
import pandas as pd
import torch

TRAIN_INT = 0
VAL_INT = 1
TEST_INT = 2

def parse_tvt(tvt_vector):
    parse_dict = {
        TRAIN_INT : 'train',
        VAL_INT : 'val',
        TEST_INT : 'test'
    }
    out = [parse_dict[x] for x in tvt_vector]
    return np.array(out)

def sigmoid_np(z):
    return (1. / (1. + np.exp(- z)))

# Utility function to print latent sites' quantile information.
def percentile_summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

def get_df_from_np_array(np_array):
    x = np.transpose(np_array)
    x = pd.DataFrame(x)
    x.columns = ['col_' + str(c) for c in x.columns]
    return x

# assignment to training, validation, and test sets
def get_train_val_test_vector(n, train, val, test):
    out = np.random.choice([TRAIN_INT, VAL_INT, TEST_INT], size = n, p = [train, val, test])
    return torch.tensor(out)