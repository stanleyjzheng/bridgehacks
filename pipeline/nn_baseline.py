import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(os.path.split(dir_path)[0], 'utils'))
print(os.path.join(os.path.split(dir_path)[0], 'utils'))

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from preprocess import load_df, series_to_supervised, oof_idx
import numpy as np
import matplotlib.pyplot as plt


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
    model_mlp.add(tf.keras.layers.Dense(100, activation='relu'))
    model_mlp.add(tf.keras.layers.Dense(1))
    model_mlp.compile(loss='mse', optimizer='adam')
    return model_mlp


def single_train(csv_name):
    window_size = 20

    df = load_df(os.path.join(os.path.split(dir_path)[0], 'input', csv_name))
    del df['infected_unvaccinated'], df['infected_vaccinated'], df['total_cases']

    cv = TimeSeriesSplit(n_splits=5)

    oof = np.zeros(df['total_cases_nextday'].values.shape)

    for idx, (tr_idx, te_idx) in enumerate(cv.split(df['total_cases_nextday'])):
        print(f"Fold: {idx}")

        print(te_idx)

        fold_train = df['total_cases_nextday'].values[tr_idx]
        fold_test = df['total_cases_nextday'].values[te_idx]

        fold_train = np.expand_dims(fold_train, -1)
        fold_test = np.expand_dims(fold_test, -1)

        fold_train, scaler = scale_data(fold_train)
        fold_test = scaler.transform(fold_test)

        x_train, y_train = series_to_supervised(fold_train)
        x_test, y_test = series_to_supervised(fold_test)

        model = get_model()

        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        #     tf.keras.layers.Dense(1)
        # ])
        # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
        optimizer = 'adam'
        # model.compile(loss=tf.keras.losses.Huber(),
        model.compile(loss = 'mse',
                    optimizer=optimizer, metrics=['mse'])
        model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

        oof_te_index = oof_idx(df['total_cases_nextday'].values, te_idx)
        window = df['total_cases_nextday'].values[oof_te_index]
        window = np.expand_dims(window, -1)
        window = scaler.transform(window)
        for i in range(0, len(window) - window_size):
            window_x = window[i:i+window_size].reshape(-1, window_size)
            oof[oof_te_index[window_size+i]] = scaler.inverse_transform(model.predict(window_x))

    plt.plot(df.index, oof)
    plt.plot(df.index, df['total_cases_nextday'])
    plt.show()


def train_val_different(train_csv, val_csv):
    train_csvs = []
    for i in train_csv:
        train_csvs.append(load_df(os.path.join(os.path.split(dir_path)[0], 'input', i)))
    
    val_csvs = []
    for i in val_csv:
        val_csvs.append(load_df(os.path.join(os.path.split(dir_path)[0], 'input', i)))
    

    scaler = StandardScaler()
    train = []
    for i in train_csvs:
        del i['infected_unvaccinated'], i['infected_vaccinated'], i['total_cases']
        df_train = i['total_cases_nextday'].values
        print(df_train.shape)
        df_train = np.expand_dims(df_train, -1)
        print(df_train.shape)
        scaler.fit(df_train)
        train.append(scaler.transform(df_train))
    train = np.concatenate(train)

    val = []
    for i in val_csvs:
        del i['infected_unvaccinated'], i['infected_vaccinated'], i['total_cases']
        df_val = i['total_cases_nextday'].values
        df_val = np.expand_dims(df_val, -1)
        val.append(scaler.transform(df_val))
    val = np.concatenate(val)

    train = windowed_dataset(train, window_size=20, batch_size=32, shuffle_buffer=10)
    test = windowed_dataset(val, window_size=20, batch_size=32, shuffle_buffer=10)

    model = get_model()

    model.compile(loss = 'mse',
                optimizer='adam', metrics=['mse'])
    model.fit(train, epochs=100, validation_data=test)

    for i in val_csvs:
        df_val = i['total_cases_nextday'].values
        df_val = np.expand_dims(df_val, -1)
        val = scaler.transform(df_val)
        preds = scaler.inverse_transform(model.predict(val)[:, 0, 0])
        plt.plot(i.index, i['total_cases_nextday'].values)
        plt.plot(i.index, preds)
        plt.show()


if __name__ == '__main__':
    single_train('observations_1.csv')
    # train_val_different(['observations_1.csv'], ['observations_2.csv', 'observations_3.csv'])
    # train_val_different(['observations_2.csv', 'observations_3.csv'], ['observations_1.csv'])