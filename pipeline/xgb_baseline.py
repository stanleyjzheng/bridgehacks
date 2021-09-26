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


def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]


def xgboost_forecast(train, testX):
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
    """concatenates two city cv for one model"""
    dfs = []
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


def citywise_stack(df_paths):
    """stacks two models per city cv"""
    dfs = []
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


if __name__ == '__main__':
    future = 0
    window_size = 50
    meta_label = False

    citywise_cv([f'observations_{i+1}.csv' for i in range(3)])
    citywise_stack([f'observations_{i+1}.csv' for i in range(3)])
    # citywise_cv_meta([f'observations_{i+1}.csv' for i in range(3)])
