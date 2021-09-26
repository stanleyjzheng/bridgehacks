import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_df(csv, additional_features=False, population=None):
    """
    Preprocesses the data from the csv file.
    :param csv: The csv filepath to preprocess.
    :param return_target: Whether to return the target or not.
    :param combine_cases: Whether to combine the vaccinated and unvaccinated cases
    :param cumulative: whether to make cases cumulative; only returns active cases if false.
    :return: pd.DataFrame The preprocessed data.
    """
    df = pd.read_csv(csv)

    return process_df(df, additional_features=additional_features)


def process_df(df, additional_features=False):
    """
    Preprocesses the data from the csv file.
    :return: pd.DataFrame
    """
    df['total_cases'] = df['infected_unvaccinated'] + df['infected_vaccinated']
    df['total_cases_nextday'] = df['total_cases'].shift(1)
    df.drop(df.head(1).index,inplace=True)
    if additional_features:
        # cumulative, increasing days consecuetive, eligible to be sick
        prev = 0.0
        cum = 0
        rolling_cum = []
        increasing_days = []
        days = 0
        for i in df['total_cases'].values:
            if i>prev:
                cum += i
                rolling_cum.append(cum)
                days += 1
                increasing_days.append(days)
            else:
                days=0
                increasing_days.append(days)
                rolling_cum.append(cum)

        df['days_increasing'] = increasing_days
        df['cumulative_cases'] = rolling_cum
        df = df[[i for i in df.columns if i != 'total_cases'] + ['total_cases']]

    return df


def series_to_supervised_old_(data, window=20):
    """
    Old/deprecated version of series_to_supervised. Used for testing.
    """
    data = np.array(data)
    x = []
    y = []
    for i in range(0, len(data)-window-1):
        x.append(data[i:i+window])
        y.append(data[i+window+1])

    x = np.concatenate(x).reshape(-1, window)
    y = np.concatenate(y)
    return x, y


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, shift=0):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
        Returns np.ndarray if np.ndarray is inputted.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i-shift))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def scale_data(df):
    """
    Scale data using scikit StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)
    return scaled_data, scaler


def oof_idx(data, te_idx, window=20):
    """
    Convert out of fold index to accomidate for a pre-window 
    """
    return list(range(min(te_idx)-window, max(te_idx)+1))


if __name__ == '__main__':
    """
    quick testing
    """
    load_df('./input/observations_1.csv')
