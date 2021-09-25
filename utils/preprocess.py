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

if __name__ == '__main__':
    load_df('./input/observations_1.csv')