import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(os.path.split(dir_path)[0], 'utils'))
print(os.path.join(os.path.split(dir_path)[0], 'utils'))

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from cv import PurgedGroupTimeSeriesSplit
from preprocess import load_df
import numpy as np


def scale_data(df):
    """
    Scale data using StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)
    return scaled_data, scaler


def get_model():
    model_mlp = tf.keras.models.Sequential()
    model_mlp.add(tf.keras.layers.Dense(100, activation='relu', input_dim=1))
    model_mlp.add(tf.keras.layers.Dense(1))
    model_mlp.compile(loss='mse', optimizer='adam')
    return model_mlp


def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    # Target timestep (t=lag)
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


if __name__ == '__main__':
    csv_name = 'observations_1.csv'

    df = load_df(os.path.join(os.path.split(dir_path)[0], 'input', csv_name))
    del df['infected_unvaccinated'], df['infected_vaccinated'], df['total_cases']

    cv = PurgedGroupTimeSeriesSplit(
            n_splits=5, 
            max_train_group_size=300,
            group_gap=10, 
            max_test_group_size=100
        )

        # this is not the actual train/test columns; you will have to make those
        # just a demo
    for idx, (tr_idx, te_idx) in enumerate(cv.split(X=df['total_cases_nextday'], y=df['total_cases_nextday'], groups=df.index)):
        print(len(tr_idx))
        print(len(te_idx))

        x_train = df['total_cases_nextday'].values[tr_idx]
        x_test = df['total_cases_nextday'].values[te_idx]

        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        x_train, scaler = scale_data(x_train)
        x_test = scaler.transform(x_test)

        x_train = windowed_dataset(x_train, window_size=20, batch_size=32, shuffle_buffer=10)
        x_test = windowed_dataset(x_test, window_size=20, batch_size=32, shuffle_buffer=10)

        # model = get_model()

        model = tf.keras.models.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
        model.compile(loss='mse',
                    optimizer=optimizer)
        model.fit(x_train, epochs=100, validation_data=x_test)
