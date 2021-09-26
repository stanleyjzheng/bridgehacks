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


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i-future))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

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


def xgboost_forecast_metalabel(train, testX):
    train = np.asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=600)
    model = MultiOutputRegressor(model).fit(trainX, trainy)

    yhat = model.predict(np.asarray([testX]))
    return yhat, model


def walk_forward_validation_meta(data):
	train, test = train_test_split(data, 1)
	# seed history with training dataset
	history = [x for x in train]
    testX, testy = test[0, :-1], test[0, -1]
    yhat, model = xgboost_forecast(history, testX)
    predictions.append(yhat)
    history.append(test[i])
    
	return error, test[:, 1], predictions, model

# def xgboost_forecast_metalabel(train, testX):
#     # transform list into array
#     train = np.asarray(train)
#     # split into input and output columns
    
#     target_preds = []
#     models = []
#     for target in range(train.shape[1]):
#         trainX, trainy = train[:, [i for i in range(train.shape[1]) if i!=target]], train[:, target]
#         model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200)
#         model.fit(trainX, trainy)
#         yhat = model.predict(np.asarray([testX[target]]))
#         target_preds.append(yhat[0])

#     return target_preds, models


# # walk-forward validation for univariate data
# def walk_forward_validation_meta(data, n_test, meta_label=False):
#     predictions = []
#     models = []
#     # split dataset
#     train, test = train_test_split(data, n_test)
#     # seed history with training dataset
#     # step over each time-step in the test set
#     target_preds = []
#     for target in range(test.shape[0]):
#         history = [x for x in train]
#         trainX, trainy = train[:, [i for i in range(train.shape[1]) if i!=target]], train[:, target]
#         # fit model on history and make a prediction
#         yhat, model = xgboost_forecast(history, testX)
#         # store forecast in list of predictions
#         target_preds.append(yhat)
#         # add actual observation to history for the next loop
#         history.append(test[0])
#         # summarize progress
#     models.append(model)
#     predictions.append(np.array(target_preds))

#     return predictions, model


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
	error = mean_squared_error(test[:, -1], predictions)
    
	return error, test[:, 1], predictions, model

def model_cv(train, testX):
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200)
    model.fit(trainX, trainy)
    # make a one-step prediction
    preds = []
    for i in testX:
        yhat = model.predict(np.asarray([i]))
        preds.append(yhat[0])
    return preds

def train_cross_validation(df, splits):
    window_size = 20

    cv = TimeSeriesSplit(n_splits=5)

    oof = np.zeros(df['total_cases'].values.shape)

    for idx, (tr_idx, te_idx) in enumerate(cv.split(df['total_cases'])):
        print(f"fold: {idx}")
        train = df.values[tr_idx]
        train, scaler = scale_data(train)
        train = series_to_supervised(train, n_in=window_size)

        oof_te_index = oof_idx(df['total_cases'].values, te_idx)
        test = df.values[oof_te_index]
        test = scaler.transform(test)
        test = series_to_supervised(test, n_in=window_size)

        # oof[te_idx] = scaler.inverse_transform(model_cv(train, test[:, :-1]))
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # split test row into input and output columns
            testX, testy = test[i, :-1], test[i, -1]
            # fit model on history and make a prediction
            yhat, model = xgboost_forecast(history, testX)
            # store forecast in list of predictions
            oof[te_idx[i]] = scaler.inverse_transform([yhat])
            # add actual observation to history for the next loop
            test[i][0] = yhat
            history.append(test[i])


    print(mean_squared_error(df['total_cases'].values, oof))
    plt.plot(df.index, df['total_cases'].values)
    plt.plot(df.index, oof)
    plt.show()
    return model

def citywise_cv(df_paths):
    dfs = []
    for i in df_paths:
        dfs.append(load_df(i))
    
    for n, (tr_idx, te_idx) in enumerate(KFold(n_splits=3, random_state=42, shuffle=True).split(dfs)):
        scalers = []
        trs = []
        
        for i in dfs[tr_idx]:
            del df['infected_unvaccinated'], df['infected_vaccinated'], df['total_cases_nextday'], df['total_vaccinated']
            tr, scaler = scale_data(df.values)
            trs.append(tr)
            scalers.append(scaler)
        
        tr = np.concatenate(trs, axis=-1)
        data = series_to_supervised(tr, n_in=window_size)
        yhat, model = walk_forward_validation(data, 10, meta_label)

        te = dfs[te_idx]
        te, scaler = scale_data(df.values)
        data = series_to_supervised(te, n_in=window_size)
        preds = []
        for i in data[:, :-1]:
            yhat = model.predict(np.asarray([i]))
            preds.append(yhat[0])
        plt.plot(df['total_cases'].values[window_size + future:], label='Expected')
        plt.plot(scaler.inverse_transform(preds), label='Predicted')
        plt.show()
        plt.clf()
        print(f"{csv_name} mse {mean_squared_error(df['total_cases'].values[window_size + future:], scaler.inverse_transform(preds))}")



if __name__ == '__main__':
    future = 3
    window_size = 50
    meta_label = True

    csv_name = 'observations_1.csv'
    df = load_df(os.path.join(os.path.split(dir_path)[0], 'input', csv_name))
    if not meta_label:
        del df['infected_unvaccinated'], df['infected_vaccinated'], df['total_cases_nextday'], df['total_vaccinated']
    else:
        del df['total_cases_nextday']

    tr, scaler = scale_data(df.values)
    data = series_to_supervised(tr, n_in=window_size)
    yhat, model = walk_forward_validation(data, 10, meta_label)

    csv_name = 'observations_2.csv'
    df = load_df(os.path.join(os.path.split(dir_path)[0], 'input', csv_name))
    del df['infected_unvaccinated'], df['infected_vaccinated'], df['total_cases_nextday'], df['total_vaccinated']

    tr, scaler = scale_data(df.values)
    data = series_to_supervised(tr, n_in=window_size)
    preds = []
    for i in data[:, :-1]:
        yhat = model.predict(np.asarray([i]))
        print(yhat.shape)
        preds.append(yhat[0])
    plt.plot(df['total_cases'].values[window_size + future:], label='Expected')
    plt.plot(scaler.inverse_transform(preds), label='Predicted')
    plt.show()
    plt.clf()
    print(f"{csv_name} mse {mean_squared_error(df['total_cases'].values[window_size + future:], scaler.inverse_transform(preds))}")

    csv_name = 'observations_3.csv'
    df = load_df(os.path.join(os.path.split(dir_path)[0], 'input', csv_name))
    del df['infected_unvaccinated'], df['infected_vaccinated'], df['total_cases_nextday'], df['total_vaccinated']

    tr, scaler = scale_data(df.values)
    data = series_to_supervised(tr, n_in=window_size)
    preds = []
    for i in data[:, :-1]:
        yhat = model.predict(np.asarray([i]))
        preds.append(yhat[0])
    plt.plot(df['total_cases'].values[window_size + future:], label='Expected')
    plt.plot(scaler.inverse_transform(preds), label='Predicted')
    plt.show()
    plt.clf()
    print(f"{csv_name} mse {mean_squared_error(df['total_cases'].values[window_size + future:], scaler.inverse_transform(preds))}")

    # cross_validation(df, 5)
