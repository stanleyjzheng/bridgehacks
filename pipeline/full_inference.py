import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(os.path.split(dir_path)[0], 'utils'))

import glob
import xgboost as xgb
import tensorflow as tf
from preprocess import process_df, series_to_supervised, oof_idx, scale_data
import pickle
import pandas as pd
from full_inference import infer_all


def push(x, y):
    push_len = len(y)
    assert len(x) >= push_len
    x[:-push_len] = x[push_len:]
    x[-push_len:] = y
    return x


# def infer_sequential(window, model):


# def infer_mlp(window, model):


# def infer_meta(window, model):


# def infer_xgb(window, model):


def infer_all(df, nometa=True, meta=True, oof=False):
    if nometa:
        nometa_df = process_df(df)
        del nometa_df['total_cases_nextday'], nometa_df['infected_vaccinated'], nometa_df['total_cases_nextday'], nometa_df['total_vaccinated'], nometa_df['infected_unvaccinated']
        nometa_data, nometa_scaler = scale_data(nometa_df.values)
        nometa_windowed = series_to_supervised(tr, n_in=window_size)
    if meta:
        meta_df = process_df(df, additional_features=True)
        del meta_df['total_cases_nextday']
        meta_data, meta_scaler = scale_data(meta_df.values)
        meta_windowed = series_to_supervised(tr, n_in=window_size)
        meta_windowed = np.asarray(meta_windowed).reshape(-1, num_cols_meta, window_size+1)
    if oof:
        preds = []
        for i in range(len(meta_windowed[:, :, :-1])):
            nometa_pred = []
            meta_pred = []
            for model in tf_meta_models:
                meta_pred.append(model.predict(np.asarray([meta_windowed[i]]))[0])
            for model in infer_xgb:
                nometa_pred.append(model.predict(np.asarray([nometa_windowed[i]]))[0])
            for model in infer_mlp:
                nometa_pred.append(model.predict(np.asarray([nometa_windowed[i]]))[0])
            for model in infer_sequential:
                te = np.expand_dims(nometa_windowed[i], axis=-1)
                nometa_pred.append(model.predict(np.asarray([te]))[0])
            nometa_pred = np.array(nometa_pred)
            nometa_pred = np.mean(nometa_pred, axis=0)
            meta_pred = np.array(meta_pred)
            meta_pred = np.mean(meta_pred, axis=0)
            meta_windowed[:, :, i+1] = meta_pred
            nometa_windowed[:, i+1] = nometa_pred
            preds.append(nometa_scaler.inverse_transform(nometa_pred*0.3+meta_pred[-1]*0.7))
        plt.plot(df['total_cases'].values[window_size:], label='Expected')
        plt.plot(preds, label='Predicted')
        plt.show()
        print(f"{te_idx[0]} mse {mean_squared_error(df['total_cases'].values[window_size:], preds)}")


    final_output = []

    nometa_window = nometa_windowed[-1]
    meta_window = meta_windowed[-1]
    meta_pred = []
    nometa_pred = []
    for model in tf_meta_models:
        meta_pred.append(model.predict(np.asarray([meta_windowed[i]]))[0])
    for model in infer_xgb:
        nometa_pred.append(model.predict(np.asarray([nometa_windowed[i]]))[0])
    for model in infer_mlp:
        nometa_pred.append(model.predict(np.asarray([nometa_windowed[i]]))[0])
    for model in infer_sequential:
        te = np.expand_dims(nometa_windowed[i], axis=-1)
        nometa_pred.append(model.predict(np.asarray([te]))[0])
    nometa_pred = np.array(nometa_pred)
    nometa_pred = np.mean(nometa_pred, axis=0)
    meta_pred = np.array(meta_pred)
    meta_pred = np.mean(meta_pred, axis=0)
    final_output.append(meta_scaler.inverse_transform(nometa_pred*0.3+meta_pred[-1]*0.7))

    nometa_window = push(nometa_window, [nometa_pred*0.3+meta_pred[-1]*0.7])
    meta_window = push(meta_window, [meta_pred])



if __name__ == '__main__':
    window_size = 50
    num_cols_meta = 6
    num_cols = 4
    future_pred = 100

    xgb_models = []
    tf_sequential_models = []
    tf_meta_models = []
    tf_mlp_models = []

    for i in glob.glob(f'{}/*'):
        if 'xgb' in i:
            with open(i, 'rb') as handle:
                trained_model = pickle.load(handle)
                xgb_models.append(trained_model)
        elif 'tf' in i:
            if 'nometa' in i:
                if 'lstm' in i or 'gru' in i or 'cnn' in i:
                    tf_sequential_models.append(tf.keras.models.load_model(i))
                elif 'mlp' in i:
                    tf_mlp_models.append(tf.keras.models.load_model(i))
            else:
                tf_meta_models.append(tf.keras.models.load_model(i))
    infer_all(pd.read_csv('./input/observations_1.csv'))