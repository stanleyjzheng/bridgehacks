import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_df(csv, return_target=True, combine_cases=False, culmulative=False):
    """
    Preprocesses the data from the csv file.
    :param csv: The csv filepath to preprocess.
    :param return_target: Whether to return the target or not.
    :param combine_cases: Whether to combine the vaccinated and unvaccinated cases
    :param cumulative: whether to make cases cumulative; only returns active cases if false.
    :return: pd.DataFrame The preprocessed data.
    """
    df = pd.read_csv(csv)
    df['total_cases'] = df['infected_unvaccinated'] + df['infected_vaccinated']
    df['total_cases_nextday'] = df['total_cases'].shift(1)
    df.drop(df.head(1).index,inplace=True)
    return df


def series_to_supervised(data, window=20):
    data = np.array(data)
    x = []
    y = []
    for i in range(0, len(data)-window-1):
        x.append(data[i:i+window])
        y.append(data[i+window+1])

    x = np.concatenate(x).reshape(-1, window)
    y = np.concatenate(y)
    return x, y


def scale_data(df):
    """
    Scale data using StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)
    return scaled_data, scaler


def oof_idx(data, te_idx, window=20):
    return list(range(min(te_idx)-window, max(te_idx)+1))


# def make_oof_prediction(data, te_idx, window=10):
#     for 


if __name__ == '__main__':
    load_df('./input/observations_1.csv')
