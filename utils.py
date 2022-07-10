import os
import pandas as pd
import numpy as np


def create_folder(name):
    if os.path.exists(f'./{name}'):
        os.chdir(f'./{name}')
        for f in os.listdir(f'.'):
            os.remove(f)
        os.chdir("..")
    else:
        os.mkdir(f'./{name}')


def read_csv_as_dtypes(path, dtypes=None):
    df = pd.read_csv(path, index_col=0)
    if dtypes:
        df.astype(dtypes)
    df.replace(' ', np.NaN).dropna()
    return df.replace(' ', np.NaN).dropna()