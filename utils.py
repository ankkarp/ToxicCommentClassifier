import os
import pandas as pd
import numpy as np


def create_folder(name):
    """
    Метод для удаления папки  со всеми ее файлами

    Параметры:
        name : str
            путь с папке

    Ничего не возвращает
    """
    if os.path.exists(f'./{name}'):
        os.chdir(f'./{name}')
        for f in os.listdir(f'.'):
            os.remove(f)
        os.chdir("..")
    else:
        os.mkdir(f'./{name}')


def load_csv_as_df(path, dtypes=None):
    """
    Метод для загрузки csv как pd.Dataframe с колонками определенных типов данных, очищая датафрейм от пустых полей

    Параметры:
        path: str
            путь к csv файлу
        dtypes: dict (default: None - приводить к типам данных не требуется)
            словарь типов данных колонок формата:
                название колонки: тип данных

    Возвращает:
        pd.DataFrame
            очищенный датафрейм
    """
    df = pd.read_csv(path, index_col=0)
    if dtypes:
        df.astype(dtypes)
    df.replace(' ', np.NaN).dropna()
    return df.replace(' ', np.NaN).dropna()