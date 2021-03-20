""" Data loaders
"""

from pdb import set_trace

import numpy as np

import torch
from sklearn import preprocessing

from holder import PyroDataset, AttribDataset, FixedParamsDataset

def create_dataset(df, use_tvt, tvt_vector = None):
    person_encoder = preprocessing.LabelEncoder()
    item_encoder = preprocessing.LabelEncoder()

    person_encoder.fit(df.student_id)
    item_encoder.fit(df.question_id)

    df['student_id_encoded'] = person_encoder.transform(df.student_id)
    df['question_id_encoded'] = item_encoder.transform(df.question_id)

    df_tensor = torch.tensor(
                            df[['student_id_encoded', 'question_id_encoded', 'correct']].values,
                            dtype = torch.int64
                        )
    x_data, y_data = df_tensor[:, :-1], df_tensor[:, -1]
    y_data = y_data.double()

    return PyroDataset(
                ques_id = x_data[:, 1], stu_id = x_data[:, 0], correct = y_data,
                use_tvt = use_tvt,
                tvt_vector = tvt_vector,
                ques_encoder = item_encoder, stu_encoder = person_encoder
            )

def create_attrib_dataset(df, use_tvt, tvt_vector = None):
    person_encoder = preprocessing.LabelEncoder()
    item_encoder = preprocessing.LabelEncoder()

    person_encoder.fit(df.student_id)
    item_encoder.fit(df.question_id)

    df['student_id_encoded'] = person_encoder.transform(df.student_id)
    df['question_id_encoded'] = item_encoder.transform(df.question_id)

    df_tensor = torch.tensor(
                            df[['student_id_encoded', 'question_id_encoded', 'correct']].values,
                            dtype = torch.int64
                        )
    x_data, y_data = df_tensor[:, :-1], df_tensor[:, -1]
    y_data = y_data.double()

    # question attributes
    sub_data = torch.zeros(df.shape[0])
    for i, sub in enumerate(['ENGLISH', 'MATH', 'CHINESE']):
        sub_data[df['subject'] == sub] = i
    sub_data = sub_data.long()
    sub_data = torch.nn.functional.one_hot(sub_data)
    sub_data = sub_data.double()

    ques_attrib_list = [sub_data]

    return AttribDataset(
                ques_id = x_data[:, 1], stu_id = x_data[:, 0], correct = y_data,
                ques_attrib_list = ques_attrib_list,
                use_tvt = use_tvt,
                tvt_vector = tvt_vector,
                ques_encoder = item_encoder, stu_encoder = person_encoder
            )

def create_fixedparam_dataset(df, fixed_params, use_tvt, tvt_vector = None):

    # Take out responses that are interact
    # fixed question id with fixed student if.
    # Data won't do anything for them.
    fp_stu = fixed_params['stu_param']['id']
    fp_ques = fixed_params['ques_param']['id']
    df = df[~(df['student_id'].isin(fp_stu) & df['question_id'].isin(fp_ques))]
    df = df.reset_index(drop = True)


    # Now filter reverse direction by removing
    # the parameters and keeping only those in
    # the dataset
    select_stu = np.isin(fp_stu, df.student_id.values)
    select_ques = np.isin(fp_ques, df.question_id.values)
    fixed_stu_params = {
        'id' : fp_stu[select_stu],
        'params' : {
            name : torch.tensor(fixed_params['stu_param']['params'][name][select_stu], dtype = torch.float64) \
            for name in fixed_params['stu_param']['params']
        },
    }
    fixed_ques_params = {
        'id' : fp_ques[select_ques],
        'params' : {
            name : torch.tensor(fixed_params['ques_param']['params'][name][select_ques], dtype = torch.float64) \
            for name in fixed_params['ques_param']['params']
        },
    }

    # typical processing
    person_encoder = preprocessing.LabelEncoder()
    item_encoder = preprocessing.LabelEncoder()

    person_encoder.fit(df.student_id)
    item_encoder.fit(df.question_id)

    df['student_id_encoded'] = person_encoder.transform(df.student_id)
    df['question_id_encoded'] = item_encoder.transform(df.question_id)

    df_tensor = torch.tensor(
                            df[['student_id_encoded', 'question_id_encoded', 'correct']].values,
                            dtype = torch.int64
                        )
    x_data, y_data = df_tensor[:, :-1], df_tensor[:, -1]
    y_data = y_data.double()

    # adding encoded ids to fixed params
    fixed_stu_params['encoded_id'] = torch.tensor(person_encoder.transform(fixed_stu_params['id']), dtype = torch.int64)
    fixed_ques_params['encoded_id'] = torch.tensor(item_encoder.transform(fixed_ques_params['id']), dtype = torch.int64)

    return FixedParamsDataset(
                ques_id = x_data[:, 1], stu_id = x_data[:, 0], correct = y_data,
                fixed_stu_params = fixed_stu_params, fixed_ques_params = fixed_ques_params,
                use_tvt = use_tvt,
                tvt_vector = tvt_vector,
                ques_encoder = item_encoder, stu_encoder = person_encoder
            )
