import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(os.path.split(dir_path)[0], 'utils'))

from preprocess import load_df, series_to_supervised, oof_idx, scale_data
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.multioutput import MultiOutputRegressor

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


def train_test_split(data, n_test):
    """Train test split by chopping off samples from time series at the end

    Args:
        data (np.ndarray): time series data
        n_test ([type]): number of samples to test on

    Returns:
        np.ndarray: train, test
    """
	return data[:-n_test, :], data[-n_test:, :]


def xgboost_forecast(train, testX):
    """
    Trains an XGBRegressor on past data
    """
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]

    # fit model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0], model


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    """Validate by getting MAE on next day

    Args:
        train (np.ndarray): train array
        model_type ([type]): model name to load from

    Returns:
        xgb.XGBRegressor: model
    """
	predictions = []
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat, model = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
    
	return predictions, model


def citywise_cv(df_paths):
    """
    Concatenate models in 3 folds (validating on entirely different cities), using metadata

    Args:
        df_paths (list): list of paths to train on in folds
    """
    dfs = []
    models = []

    for i in df_paths:
        df = load_df(os.path.join(os.path.split(dir_path)[0], 'input', i))
        del df['infected_unvaccinated'], df['infected_vaccinated'], df['total_cases_nextday'], df['total_vaccinated']
        dfs.append(df)
    
    for n, (tr_idx, te_idx) in enumerate(KFold(n_splits=3, random_state=42, shuffle=True).split(dfs)):
        scalers = []
        trs = []

        dfs_tr = [dfs[i] for i in tr_idx]
        for df in dfs_tr:
            tr, scaler = scale_data(df.values)
            trs.append(tr)
            scalers.append(scaler)

        tr = np.concatenate(trs, axis=0)
        data = series_to_supervised(tr, n_in=window_size, shift=future)
        yhat, model = walk_forward_validation(data, 1)
        models.append(model)

        te = dfs[te_idx[0]]
        te, scaler = scale_data(te.values)
        data = series_to_supervised(te, n_in=window_size, shift=future)
        preds = []
        for i in data[:, :-1]:
            yhat = model.predict(np.asarray([i]))
            preds.append(yhat[0])
        plt.plot(dfs[te_idx[0]]['total_cases'].values[window_size + future:], label='Expected')
        plt.plot(scaler.inverse_transform(preds), label='Predicted')
        plt.show()
        plt.clf()
        print(f"{te_idx[0]} mse {mean_squared_error(df['total_cases'].values[window_size + future:], scaler.inverse_transform(preds))}")
    save_model(models)


def citywise_stack(df_paths):
    """
    Stack models in 3 folds (validating on entirely different cities)

    Args:
        df_paths (list): list of paths to train on in folds
    """
    dfs = []
    models = []

    for i in df_paths:
        df = load_df(os.path.join(os.path.split(dir_path)[0], 'input', i))
        del df['infected_unvaccinated'], df['infected_vaccinated'], df['total_cases_nextday'], df['total_vaccinated']
        dfs.append(df)

    for n, (tr_idx, te_idx) in enumerate(KFold(n_splits=3, random_state=42, shuffle=True).split(dfs)):
        scalers = []
        trs = []

        dfs_tr = [dfs[i] for i in tr_idx]
        for df in dfs_tr:
            tr, scaler = scale_data(df.values)
            trs.append(tr)
            scalers.append(scaler)
        
        master_preds = []
        for tr in trs:
            data = series_to_supervised(tr, n_in=window_size, shift=future)
            yhat, model = walk_forward_validation(data, 1)
            models.append(model)

            te = dfs[te_idx[0]]
            te, scaler = scale_data(te.values)
            data = series_to_supervised(te, n_in=window_size, shift=future)
            preds = []
            for i in data[:, :-1]:
                yhat = model.predict(np.asarray([i]))
                preds.append(yhat[0])
            master_preds.append(np.array(preds))
        master_preds = np.mean(np.array(master_preds), axis=0)


        plt.plot(dfs[te_idx[0]]['total_cases'].values[window_size + future:], label='Expected')
        plt.plot(scaler.inverse_transform(master_preds), label='Predicted')
        plt.show()
        plt.clf()
        print(f"{te_idx[0]} mse {mean_squared_error(df['total_cases'].values[window_size + future:], scaler.inverse_transform(preds))}")
    save_model(models, stack=True)


def save_model(models, stack=False):
    """Save models to directory (tensorflow only)

    Args:
        models (list of tf.keras.model): models to save
        stack (bool, optional): save two copies of the same fold. Defaults to False.
        model_type (str, optional): incorporated into save path. Defaults to "lstm".
    """
    p1 = True
    fold = 0

    for i in models:
        if stack:
            if p1:
                name = f"xgb_fold_{fold}_stackp1.pkl"
                p1 = False
            else:
                name = f"xgb_fold_{fold}_stackp2.pkl"
                fold += 1
                p1 = True
        else:
            name = f"xgb_fold_{fold}_nostack.pkl"
            fold += 1

        print(name)
        with open(os.path.join('models', name), 'wb') as f:
            pickle.dump(i, f)


if __name__ == '__main__':
    future = 0
    window_size = 50
    meta_label = False

    citywise_cv([f'observations_{i+1}.csv' for i in range(3)])
    citywise_stack([f'observations_{i+1}.csv' for i in range(3)])
    # citywise_cv_meta([f'observations_{i+1}.csv' for i in range(3)])
