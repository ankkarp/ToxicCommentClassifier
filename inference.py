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
    def __init__(self, model=None, vocab='DeepPavlov/rubert-base-cased-conversational', token_len=128):
        """
        Конструктор тестировщика

        Параметры:
            model
                модель, которую нужно протестировать
            vocab: str (default: 'DeepPavlov/rubert-base-cased-conversational')
                название модели/словаря bert
            token_len: int
                длина токенов, в которые преобразуется текст токенайзеров
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {self.device}')
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(vocab)
        self.model = model
        self.token_len = token_len
        self.res_df = None

    def get_mistakes(self, export_file=None, print_accs=False):
        """
        Метод для получения датафрейм данных инференса, которые были класифицированы неправильно

        Параметры:
            export_file: str (default: None)
                путь к csv файлу, в который нужно сохранить результат
                при None, не будет сохранять результаты
            print_accs: bool (default: False)
                вывести ли в консоль точность модели по каждому классу поотдельности

        Возвращает:
            pd.DataFrame
                таблица данных, на которых модель ошиблась, с колонками:
                    text : str
                        входной текст
                    class_true: str
                        истинный лейбл
                    class_prediction: str
                        предсказанный лейбл
                    probabilities: float
                        вероятность предсказанного класса
        """
        mistakes_df = self.res_df.loc[self.res_df["class_prediction"] != self.res_df["class_true"]]
        if export_file:
            mistakes_df.to_csv(export_file)
        if print_accs:
            print("accuracy\t toxic \tnon-toxic", end='\n\t\t')
            print(f'{1 - sum(mistakes_df["class_true"] == "toxic") / len(self.res_df):0.4f}', end='\t')
            print(f'{1 - sum(mistakes_df["class_true"] == "non-toxic") / len(self.res_df):0.4f}', end='\t')
        return mistakes_df

    def load_model(self, path: str):
        """
        Метод для загрузки модели BERT из файла

        Параметры:
            path: str
                путь к файлу модели

        Ничего не возвращает
        """
        self.model = torch.load(path, map_location=self.device)

    def test(self, test_csv: str, batch_sz=16, export_file=None):
        """
        Метод для инференса модели

        Параметры:
            df: pd.DataFrame
                таблица исходных данных с колонками:
                    text : str - входной текст
                    class_true: str - истинный лейбл
            batch_sz: int
                размер батча
            export_file: str (default: None)
                путь к csv файлу, в который нужно сохранить результат
                при None, не будет сохранять результаты


        Возвращает:
            pd.DataFrame
                таблица результата инференса:
                    text : str
                        входной текст
                    class_true: str
                        истинный лейбл
                    class_prediction: str
                        предсказанный лейбл
                    probabilities: float
                        вероятность предсказанного класса
        """
        labels = {0: "non-toxic", 1: "toxic"}
        if self.model:
            test_df = load_csv_as_df(test_csv)
            if "comment" in test_df.columns:
                self.res_df = test_df.copy().rename(columns={"comment": "text", "toxic": "class_true"})
            self.res_df["class_prediction"] = np.NaN
            self.res_df["probabilities"] = np.NaN
            test = Dataset(self.tokenizer, test_df,  self.token_len)
            test_dataloader = DataLoader(test, batch_size=batch_sz)
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

            print(f'F1 score: {f1_score(self.res_df["class_true"], self.res_df["class_prediction"])}')
            self.res_df["class_true"].replace(labels, inplace=True)
            self.res_df["class_prediction"].replace(labels, inplace=True)
            if export_file:
                self.res_df.to_csv(export_file)
            return self.res_df
        else:
            print("Нет модели")


# if __name__ == "__main__":
#     df1 = pd.DataFrame(np.random.randint(0, high=2, size=(16, 2)))
#     print(df1[df1[0] == 0])
