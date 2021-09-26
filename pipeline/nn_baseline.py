import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(os.path.split(dir_path)[0], 'utils'))
print(os.path.join(os.path.split(dir_path)[0], 'utils'))

import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from preprocess import load_df, series_to_supervised_old_, oof_idx, scale_data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau


def get_model():
    model_mlp = tf.keras.models.Sequential()
    model_mlp.add(tf.keras.layers.Dense(100, activation='relu'))
    model_mlp.add(tf.keras.layers.Dense(100, activation='relu'))

    model_mlp.add(tf.keras.layers.Dense(1))
    model_mlp.compile(loss='mse', optimizer='adam')
    return model_mlp

def get_lstm():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        tf.keras.layers.Dense(1, )
    ])
    # model.compile(loss=tf.keras.losses.Huber(),
    model.compile(loss = 'mse',optimizer='adam')
    return model

def get_cnn():
    model_cnn = tf.keras.models.Sequential()
    model_cnn.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))
    model_cnn.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model_cnn.add(tf.keras.layers.Flatten())
    model_cnn.add(tf.keras.layers.Dense(50, activation='relu'))
    model_cnn.add(tf.keras.layers.Dense(1))
    model_cnn.compile(loss='mse', optimizer='adam')
    return model_cnn

def create_ae_mlp(num_columns, num_labels, hidden_units, dropout_rates, ls = 1e-2, lr = 1e-3):
    
    inp = tf.keras.layers.Input(shape = (num_columns, ))
    x0 = tf.keras.layers.BatchNormalization()(inp)
    
    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('swish')(encoder)
    
    decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
    decoder = tf.keras.layers.Dense(num_columns, name = 'decoder')(decoder)

    x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation('swish')(x_ae)
    x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)

    out_ae = tf.keras.layers.Dense(num_labels, activation = 'sigmoid', name = 'ae_action')(x_ae)
    
    x = tf.keras.layers.Concatenate()([x0, encoder])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)
    
    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)
        
    out = tf.keras.layers.Dense(num_labels, activation = 'sigmoid', name = 'action')(x)
    
    model = tf.keras.models.Model(inputs = inp, outputs = [decoder, out_ae, out])
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                    loss = {'decoder': tf.keras.losses.MeanSquaredError(), 
                            # 'ae_action': tf.keras.losses.BinaryCrossentropy(label_smoothing = ls),
                            # 'action': tf.keras.losses.BinaryCrossentropy(label_smoothing = ls), 
                            'ae_action': tf.keras.losses.MeanAbsoluteError(),
                            'action': tf.keras.losses.MeanAbsoluteError(), 
                            },
                    metrics = {'decoder': tf.keras.metrics.MeanAbsoluteError(name = 'MAE'), 
                                'ae_action': tf.keras.metrics.MeanAbsoluteError(name = 'MAE'), 
                                'action': tf.keras.metrics.MeanAbsoluteError(name = 'MAE'), 
                                }, 
                    )
    
    return model

def single_train(csv_name):
    window_size = 20
    model_type = 'lstm'

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

        x_train, y_train = series_to_supervised_old_(fold_train)
        x_test, y_test = series_to_supervised_old_(fold_test)

        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        if model_type == 'mlp':
            model = get_model()
        elif model_type == 'lstm':
            model = get_lstm()
        elif model_type == 'cnn':
            model = get_cnn()
        elif model_type == 'ae_mlp':
            model = create_ae_mlp(window_size, 1, [96, 96, 896, 448, 448, 256], [0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882, 0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448], 1e-8)

        print(x_train.shape)
        model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=128, verbose=2)

        oof_te_index = oof_idx(df['total_cases_nextday'].values, te_idx)
        window = df['total_cases_nextday'].values[oof_te_index]
        window = np.expand_dims(window, -1)
        window = scaler.transform(window)
        for i in range(0, len(window) - window_size):
            window_x = window[i:i+window_size].reshape(-1, window_size)
            window_x = np.expand_dims(window_x, -1)
            if model_type == 'mlp':
                pred = model.predict(window_x)[:, -1, 0]
            else:
                pred = model.predict(window_x)
            oof[oof_te_index[window_size+i]] = scaler.inverse_transform(pred)
            window[i+window_size] = pred


    print('Validation rmse:', np.sqrt(mean_squared_error(oof, df['total_cases_nextday'].values)))
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
    model.fit(train, epochs=100, validation_data=test, batch_size=32)

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