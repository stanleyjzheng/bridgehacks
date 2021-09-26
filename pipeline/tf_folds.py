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


def get_mlp():
    model_mlp = tf.keras.models.Sequential()
    model_mlp.add(tf.keras.layers.Dense(100, activation='relu', input_shape=[window_size]))
    model_mlp.add(tf.keras.layers.Dense(100, activation='relu'))

    model_mlp.add(tf.keras.layers.Dense(num_cols))
    model_mlp.compile(loss='mse', optimizer='adam')
    return model_mlp


def get_lstm():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dense(num_cols)
    ])
    model.compile(loss = 'mse', optimizer='adam')
    return model


def get_gru():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128)),
        tf.keras.layers.Dense(num_cols)
    ])
    model.compile(loss = 'mse', optimizer='adam')
    return model


def get_cnn():
    model_cnn = tf.keras.models.Sequential()
    model_cnn.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))
    model_cnn.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model_cnn.add(tf.keras.layers.Flatten())
    model_cnn.add(tf.keras.layers.Dense(50, activation='relu'))
    model_cnn.add(tf.keras.layers.Dense(num_cols))
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
        
    out = tf.keras.layers.Dense(num_labels, name = 'action')(x)
    
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
        model = walk_forward_validation(data, model_type)

        te = dfs[te_idx[0]]
        te, scaler = scale_data(te.values)
        data = series_to_supervised(te, n_in=window_size, shift=future)
        preds = []
        for i in data[:, :-1]:
            if model_type == 'lstm' or model_type == 'cnn' or model_type == 'gru':
                i = np.expand_dims(i, axis=-1)
            yhat = model.predict(np.asarray([i]))
            if model_type == 'ae_mlp':
                preds.append(yhat[2][0])
            else:
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
            model = walk_forward_validation(data, model_type)

            te = dfs[te_idx[0]]
            te, scaler = scale_data(te.values)
            data = series_to_supervised(te, n_in=window_size, shift=future)
            preds = []
            for i in data[:, :-1]:
                if model_type == 'lstm' or model_type == 'cnn' or model_type == 'gru':
                    i = np.expand_dims(i, axis=-1)
                yhat = model.predict(np.asarray([i]))
            if model_type == 'ae_mlp':
                preds.append(yhat[2][0])
            else:
                preds.append(yhat[0])
            master_preds.append(np.array(preds))
        master_preds = np.mean(np.array(master_preds), axis=0)


        plt.plot(dfs[te_idx[0]]['total_cases'].values[window_size + future:], label='Expected')
        plt.plot(scaler.inverse_transform(master_preds), label='Predicted')
        plt.show()
        plt.clf()
        print(f"{te_idx[0]} mse {mean_squared_error(df['total_cases'].values[window_size + future:], scaler.inverse_transform(preds))}")


def walk_forward_validation(train, model_type):
    train = np.asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]

    if model_type == 'mlp':
        model = get_mlp()
    elif model_type == 'lstm':
        model = get_lstm()
        trainX = trainX.reshape(-1, window_size, num_cols)
        trainy = np.expand_dims(trainy, axis=-1)
        trainy = np.expand_dims(trainy, axis=-1)
    elif model_type == 'gru':
        model = get_gru()
        trainX = trainX.reshape(-1, window_size, num_cols)
        trainy = np.expand_dims(trainy, axis=-1)
        trainy = np.expand_dims(trainy, axis=-1)
    elif model_type == 'cnn':
        model = get_cnn()
        trainX = trainX.reshape(-1, window_size, num_cols)
        trainy = np.expand_dims(trainy, axis=-1)
        trainy = np.expand_dims(trainy, axis=-1)
    elif model_type == 'ae_mlp':
        model = create_ae_mlp(window_size, 1, [96, 96, 896, 448, 448, 256], [0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882, 0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448], 1e-4)

    model.fit(trainX, trainy, epochs=20)
    return model


def walk_forward_validation_meta(train):
    print(train.shape)
    train = np.asarray(train).reshape(-1, num_cols, window_size+1)
    trainX, trainy = train[:, :, :-1], train[:, :, -1]

    if model_type == 'mlp':
        model = get_mlp()
    elif model_type == 'lstm':
        model = get_lstm()
    elif model_type == 'gru':
        model = get_gru()
    elif model_type == 'cnn':
        model = get_cnn()
    elif model_type == 'ae_mlp':
        model = create_ae_mlp(window_size, 1, [96, 96, 896, 448, 448, 256], [0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882, 0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448], 1e-4)

    model.fit(trainX, trainy, epochs=20)
    return model


def citywise_stack_meta(df_paths):
    dfs = []
    for i in df_paths:
        df = load_df(os.path.join(os.path.split(dir_path)[0], 'input', i), additional_features=True)
        df = df[[i for i in df.columns if i != 'total_cases'] + ['total_cases']]
        del df['total_cases_nextday']
        print(df)
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
            data = series_to_supervised(tr, n_in=window_size)
            model = walk_forward_validation_meta(data)

            te = dfs[te_idx[0]]
            te, scaler = scale_data(te.values)
            data = series_to_supervised(te, n_in=window_size)
            data = np.asarray(data).reshape(-1, num_cols, window_size+1)

            preds = []
            for i in data[:, :, :-1]:
                yhat = model.predict(np.asarray([i]))
                preds.append(yhat[0])
            master_preds.append(np.array(preds))
        # average preds of two models
        master_preds = np.mean(np.array(master_preds), axis=0)

        plt.plot(dfs[te_idx[0]]['total_cases'].values[window_size + future:], label='Expected')
        scaled = scaler.inverse_transform(master_preds)
        plt.plot(scaled[:, -1], label='Predicted')
        plt.show()
        plt.clf()
        print(f"{te_idx[0]} mse {mean_squared_error(df['total_cases'].values[window_size + future:], scaled[:, -1])}")


def citywise_cv_meta(df_paths):
    dfs = []
    for i in df_paths:
        df = load_df(os.path.join(os.path.split(dir_path)[0], 'input', i), additional_features=True)
        df = df[[i for i in df.columns if i != 'total_cases'] + ['total_cases']]
        del df['total_cases_nextday']
        print(df)
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


        data = series_to_supervised(tr, n_in=window_size)
        model = walk_forward_validation_meta(data)

        te = dfs[te_idx[0]]
        te, scaler = scale_data(te.values)
        data = series_to_supervised(te, n_in=window_size)
        data = np.asarray(data).reshape(-1, num_cols, window_size+1)

        preds = []
        for i in data[:, :, :-1]:
            yhat = model.predict(np.asarray([i]))
            preds.append(yhat[0])

        plt.plot(dfs[te_idx[0]]['total_cases'].values[window_size + future:], label='Expected')
        scaled = scaler.inverse_transform(preds)
        plt.plot(scaled[:, -1], label='Predicted')
        plt.show()
        plt.clf()
        print(f"{te_idx[0]} mse {mean_squared_error(df['total_cases'].values[window_size + future:], scaled[:, -1])}")


if __name__ == '__main__':
    future = 0
    window_size = 50
    meta_label = False
    model_type = 'gru'
    num_cols = 6

    # citywise_cv([f'observations_{i+1}.csv' for i in range(3)])
    # citywise_stack([f'observations_{i+1}.csv' for i in range(3)])
    # citywise_stack_meta([f'observations_{i+1}.csv' for i in range(3)])
    citywise_cv_meta([f'observations_{i+1}.csv' for i in range(3)])
