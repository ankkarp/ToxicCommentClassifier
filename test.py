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
    def __init__(self, model, vocab='DeepPavlov/rubert-base-cased-conversational', token_len=64, batch_sz=16):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Используется {self.device}')
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(vocab)
        self.model = model
        self.token_len = token_len
        self.batch_sz = batch_sz
        self.res_df = None

    def analyse_results(self, df):
        """
        Метод анализа резултатов тестированияю. Подсчитывает метрики по данным
        """
        if len(df):
            res_info = pd.DataFrame(index=["toxic", "non-toxic", "total"], columns=["accuracy", "f1_score"])
            res_info["total"]["accuracy"] = \
                len(df[self.res_df["class_prediction"] == self.res_df['class_true']]) / len(self.res_df)
            res_info["total"]["f1_score"] = f1_score(self.res_df['class_prediction'], self.res_df['class_true'])
            for cl in [res_info.index[:-1]]:
                cl_df = df[df["class_true"] == cl]
                res_info[cl]["accuracy"] = len(cl_df[cl_df['class_prediction'] == cl_df['class_true']]) \
                                           / len(df)
                res_info[cl]["f1_score"] = f1_score(cl_df['class_prediction'], cl_df['class_true'])
            return res_info
        else:
            print("Модель еще не была протестирована")

    def load_model(self, path):
        self.model = torch.load(path, map_location=self.device)

    def test(self, test_csv: str, export_file=None, analyse=True):
        if self.model:
            test_df = load_csv_as_df(test_csv).iloc[:17]
            if "text" not in test_df.columns:
                # print(test_df.columns)
                self.res_df = test_df.copy().rename(columns={"comment": "text", "toxic": "class_true"})
                self.res_df["class_true"].replace({0: "non-toxic", 1: "toxic"}, inplace=True)
            self.res_df["class_prediction"] = np.NaN
            self.res_df["probabilities"] = np.NaN
            test = Dataset(self.tokenizer, test_df,  self.token_len)
            test_dataloader = DataLoader(test, batch_size=self.batch_sz)
            self.model.eval()
            with torch.no_grad():
                for i, (x_batch, _) in enumerate(tqdm(test_dataloader)):
                    mask = x_batch['attention_mask'].to(self.device)
                    input_id = x_batch['input_ids'].squeeze(1).to(self.device)
                    output = self.model(input_id, mask) if isinstance(self.model, BertClassifier) else \
                        self.model(input_id, mask, return_dict=False)[0]
                    # print(output, output.argmax(dim=1))
                    self.res_df.loc[self.res_df.index[i * self.batch_sz: (i + 1) * self.batch_sz], "class_prediction"] = \
                        output.argmax(dim=1).numpy()
                    for idx in range(len(output.argmax(dim=1))):
                        self.res_df.loc[self.res_df.index[idx+self.batch_sz*i], "probabilities"] = \
                            output.numpy()[idx][output.argmax(dim=1)[idx]]
            print(self.res_df["class_prediction"], self.res_df["probabilities"])
            if export_file:
                self.res_df.to_csv(f"./{export_file}")
            self.res_df = test_df
            if analyse:
                print(self.analyse_results(self.res_df))
            return self.res_df
        else:
            print("Чтобы провести тестирование необходимо загрузить или обучить модель")
