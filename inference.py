import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np

from BertClassifier import BertClassifier
from Dataset import Dataset
from metrics import f1_score
from utils import load_csv_as_df

pd.options.mode.chained_assignment = None


class Tester:
    """
    Класс тестировщика, позволяет провести инференс модели
    """
    def __init__(self, model=None, vocab='DeepPavlov/rubert-base-cased-conversational', token_len=64, batch_sz=16):
        """
        Конструктор тестировщика

        Параметры:
            model - модель, которую нужно протестировать
            vocab
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {self.device}')
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(vocab)
        self.model = model
        self.token_len = token_len
        self.batch_sz = batch_sz
        self.res_df = None

    def analyse_results(self, df):
        """
        Метод, позволяющий получить точность модели по классам, основываясть на таблице результатов инференса

        Параметры:
            df: pd.DataFrame
                таблица исходный данных с колонками:
                    text : str - входной текст
                    class_true: str - истинный лейбл

        Возвращает:
            pd.DataFrame
                таблица данных инференса с колонками:
                    text : str
                        входной текст
                    class_true: str
                        истинный лейбл
                    class_prediction: str
                        предсказанный лейбл
                    probabilities: float
                        вероятность предсказанного класса
        """
        res_info = pd.DataFrame(index=["toxic", "non-toxic", "total"], columns=["accuracy"])
        res_info.loc["total", "accuracy"] = len(df.loc[df["class_prediction"] == df['class_true']]) / len(df)
        for cl in res_info.index[:-1]:
            pred_n_true = df.loc[df["class_true"] == cl, ['class_prediction', 'class_true']].values.T
            print(pred_n_true)
            res_info.loc[cl, "accuracy"] = np.mean(np.array_equal(*pred_n_true))
        return res_info

    def load_model(self, path: str):
        """
        Функция загрузки модели.

        Параметры:
            path: str - путь к файлу модели

        Возвращает:
            ничего
        """
        self.model = torch.load(path, map_location=self.device)

    def test(self, test_csv: str, export_file=None, analyse=True):
        labels = {0: "non-toxic", 1: "toxic"}
        if self.model:
            test_df = load_csv_as_df(test_csv)
            if "comment" in test_df.columns:
                self.res_df = test_df.copy().rename(columns={"comment": "text", "toxic": "class_true"})
            self.res_df["class_prediction"] = np.NaN
            self.res_df["probabilities"] = np.NaN
            test = Dataset(self.tokenizer, test_df,  self.token_len)
            test_dataloader = DataLoader(test, batch_size=self.batch_sz)
            self.model.eval()
            i = 0
            with torch.no_grad():
                for x_batch, y_batch in tqdm(test_dataloader):
                    mask = x_batch['attention_mask'].to(self.device)
                    input_id = x_batch['input_ids'].squeeze(1).to(self.device)
                    output = self.model(input_id, mask) if isinstance(self.model, BertClassifier) else \
                        self.model(input_id, mask, return_dict=False)[0]
                    probs, preds = torch.softmax(output, dim=1).max(dim=1)
                    self.res_df.loc[self.res_df.index[i: i + len(y_batch)], "class_prediction"] = \
                        preds.cpu().detach().numpy()
                    self.res_df.loc[self.res_df.index[i: i + len(y_batch)], "probabilities"] = \
                        probs.cpu().detach().numpy()
                    i += len(y_batch)
            self.res_df["class_true"].replace(labels, inplace=True)
            self.res_df["class_prediction"].replace(labels, inplace=True)
            print(self.res_df.info())
            if export_file:
                self.res_df.to_csv(f"./{export_file}")
            if analyse:
                print(self.analyse_results(self.res_df))
            return self.res_df
        else:
            print("Нет модели")
