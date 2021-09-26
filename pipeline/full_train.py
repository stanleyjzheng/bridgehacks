import tensorflow as tf
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(os.path.split(dir_path)[0], 'utils'))
sys.path.append('./utils')
sys.path.append('./pipeline')

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from preprocess import load_df, series_to_supervised, oof_idx, scale_data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from full_inference import infer_only_meta
from tf_folds import walk_forward_validation_meta
import glob
import pandas as pd


def citywise_stack_meta(df_path, window_size=50, model_type='lstm'):
    """Stack models on city and use metadata

    Args:
        df_path (str): csv path
        window_size (int, optional): window size. Defaults to 50.
        model_type (str, optional): model name, lstm, gru, etc. Defaults to 'lstm'.
    """
    dfs = []
    models = []

    tr_idx = 0

    df = load_df(df_path, additional_features=True)
    df = df[[i for i in df.columns if i != 'total_cases'] + ['total_cases']]
    del df['total_cases_nextday']
    print(df)

    tr, scaler = scale_data(df.values)

    data = series_to_supervised(tr, n_in=window_size)
    model = walk_forward_validation_meta(data, model_type=model_type)
    models.append(model)

    save_models(models, stack=True, model_type=model_type)


def citywise_cv_meta(df_path, window_size=50, model_type='lstm'):
    """Simultaneousluy train models on city and use metadata

    Args:
        df_path (str): csv path
        window_size (int, optional): window size. Defaults to 50.
        model_type (str, optional): model name, lstm, gru, etc. Defaults to 'lstm'.
    """
    dfs = []
    models = []

    tr_idx = 0

    df = load_df(df_path, additional_features=True)
    df = df[[i for i in df.columns if i != 'total_cases'] + ['total_cases']]
    del df['total_cases_nextday']

    tr, scaler = scale_data(df.values)

    data = series_to_supervised(tr, n_in=window_size)
    model = walk_forward_validation_meta(data, model_type=model_type)
    models.append(model)
    save_models(models, stack=False, model_type=model_type)


def save_models(models, stack=False, model_type="lstm"):
    """Save models to directory (tensorflow only)

    Args:
        models (list of tf.keras.model): models to save
        stack (bool, optional): save two copies of the same fold. Defaults to False.
        model_type (str, optional): incorporated into save path. Defaults to "lstm".
    """
    save_dir = './models_task2'
    p1 = True
    fold = 0

    for i in models:
        meta = "meta"
        if stack:
            if p1:
                name = f"tf_fold_{fold}_{model_type}_stackp1_{meta}.h5"
                p1 = False
            else:
                name = f"tf_fold_{fold}_{model_type}_stackp2_{meta}.h5"
                fold += 1
                p1 = True
        else:
            name = f"tf_fold_{fold}_{model_type}_nostack_{meta}.h5"
            fold += 1
        print(os.path.join(save_dir, name))
        i.save(os.path.join(save_dir, name))


def model_prediction(data_csv, number_of_days):
    """Predict future cases using trained models

    Args:
        data_csv (str): csv path
        number_of_days (int): number of days to predict into the future
    """
    model_type = 'lstm'
    save_dir = 'models_task2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for model_type in ["lstm", "gru", "cnn"]:
        citywise_stack_meta(data_csv, model_type=model_type)
    tf_meta_models = []
    print(f'./{save_dir}/*')
    for i in glob.glob(f'./{save_dir}/*'):
        tf_meta_models.append(tf.keras.models.load_model(i))

    results = infer_only_meta(pd.read_csv(data_csv), tf_meta_models, future_pred=number_of_days)


if __name__ == '__main__':
    model_prediction('./input/observations_1.csv', 100)
