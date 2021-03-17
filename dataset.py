""" Data loaders
"""

import torch
from sklearn import preprocessing

from holder import PyroDataset, AttribDataset

def create_dataset(df, use_tvt, tvt_vector = None):
    person_encoder = preprocessing.LabelEncoder()
    item_encoder = preprocessing.LabelEncoder()

    person_encoder.fit(df.student_id)
    item_encoder.fit(df.question_id)

    df['student_id_encoded'] = person_encoder.transform(df.student_id)
    df['question_id_encoded'] = item_encoder.transform(df.question_id)

    df_tensor = torch.tensor(
                            df[['student_id_encoded', 'question_id_encoded', 'correct']].values,
                            dtype=torch.int64
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
                            dtype=torch.int64
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