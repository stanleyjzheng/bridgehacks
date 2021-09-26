import tensorflow as tf
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(os.path.split(dir_path)[0], 'utils'))

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from preprocess import load_df, series_to_supervised, oof_idx, scale_data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from full_inference import infer_all


def citywise_stack_meta(df_paths):
    dfs = []
    models = []

    tr_idx = 0

    df = load_df(os.path.join(os.path.split(dir_path)[0], 'input', i), additional_features=True)
    df = df[[i for i in df.columns if i != 'total_cases'] + ['total_cases']]
    del df['total_cases_nextday']
    dfs.append(df)

    scalers = []
    trs = []

    dfs_tr = [dfs[i] for i in tr_idx]
    for df in dfs_tr:
        tr, scaler = scale_data(df.values)
        trs.append(tr)
        scalers.append(scaler)

    master_preds = []
    for tr in trs:
        data = series_to_supervised(tr, n_in=window_size)
        model = walk_forward_validation_meta(data)
        models.append(model)

    save_models(models, stack=True)


def citywise_cv_meta(df_paths):
    dfs = []
    models = []

    tr_idx = 0

    df = load_df(os.path.join(os.path.split(dir_path)[0], 'input', i), additional_features=True)
    df = df[[i for i in df.columns if i != 'total_cases'] + ['total_cases']]
    del df['total_cases_nextday']
    dfs.append(df)

    scalers = []
    trs = []

    dfs_tr = [dfs[i] for i in tr_idx]
    for df in dfs_tr:
        tr, scaler = scale_data(df.values)
        trs.append(tr)
        scalers.append(scaler)

    tr = np.concatenate(trs, axis=0)


    data = series_to_supervised(tr, n_in=window_size)
    model = walk_forward_validation_meta(data)
    models.append(model)
    save_models(models, stack=False)


def save_models(models, stack=False):
    save_dir = './models_task2'
    p1 = True
    fold = 0

    for i in models:
        meta = "meta" if num_cols > 5 else "nometa"
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
    citywise_cv_meta([data_csv])
    citywise_stack_meta([data_csv])
    results = infer_only_meta(pd.read_csv(data.csv), future_pred=number_of_days)



if __name__ == '__main__':
    model_prediction('./input/observations_1.csv')