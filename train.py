import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer, logging
import gc

from BertClassifier import BertClassifier
from Dataset import Dataset
from inference import Tester
from metrics import f1_score
from utils import create_folder, load_csv_as_df


class Trainer(Tester):
    """Класс для обучения модели, также наследует функционал для инференса """

    def __init__(self, model_class=BertForSequenceClassification, vocab='DeepPavlov/rubert-base-cased-conversational',
                 token_len=128, batch_sz=16):
        """
        Конструктор тренировщика модели

        Параметры:
            model_class: class (default: BertForSequenceClassification)
                класс модели (BertForSequenceClassification или BertClassifier)
            vocab: str (default: 'DeepPavlov/rubert-base-cased-conversational')
                название модели/словаря bert
            token_len: int
                длина токенов, в которые преобразуется текст токенайзеров
            batch_sz: int
                размер батча
        """
        self.model_class = model_class
        Tester.__init__(self, model=None, vocab=vocab, token_len=token_len, batch_sz=batch_sz)
        self.history = None

    def plot_history(self):
        """Стоит график обучения"""
        for i, metric in enumerate(("loss", "acc", "f1_score")):
            ax = plt.subplot(1, 3, i+1)
            ax.title.set_text(metric)
            ax.plot([i for i in range(1, len(self.history.index) + 1)], self.history[('_'.join(("train", metric)))],
                    label="train")
            ax.plot([i for i in range(1, len(self.history.index) + 1)], self.history[('_'.join(("train", metric)))],
                    label="val")
            ax.legend()
        plt.show()

    def __epoch(self, dataloader, loss_fn, ep, opt=None):
        """
        Проводит одну эпохи обучения/валидации, логирует историю обучения

        Параметры:
            dataloader: torch.utils.data.dataloader.DataLoader
                даталоадер обучения/валидации
            loss_fn
                функция потерь
            ep: int
                номер текущей эпохи
            opt (default=None)
                оптимизатор для обучения модели, на эпохе валидации не передавать

        Ничего не возвращает
        """
        batch_loss, batch_acc, batch_f1_score = [], [], []
        for x_batch, y_batch in dataloader:
            y_batch = y_batch.to(self.device)
            mask = x_batch['attention_mask'].to(self.device)
            input_id = x_batch['input_ids'].squeeze(1).to(self.device)
            output = self.model(input_id, mask) if isinstance(self.model, BertClassifier) else \
                self.model(input_id, mask, return_dict=False)[0]
            # print(output, y_batch)
            loss = loss_fn(output, y_batch.to(dtype=torch.long))
            batch_loss.append(loss.item())

            acc = (output.argmax(dim=1) == y_batch).sum().item() / len(y_batch)
            batch_acc.append(acc)
            batch_f1_score.append(f1_score(output.argmax(dim=1).cpu().detach(), y_batch.cpu().detach()))

            if opt:
                self.model.zero_grad()
                loss.backward()
                opt.step()
        mode = "train" if opt else "val"
        self.history.iloc[ep][f"{mode}_acc"] = np.mean(batch_acc)
        self.history.iloc[ep][f"{mode}_loss"] = np.mean(batch_loss)
        self.history.iloc[ep][f"{mode}_f1_score"] = np.mean(batch_f1_score)

    def __init_model(self):
        """
        Метод инициализации новой модели класса поля model_class

        Параметры:
            нет

        Ничего не возвращает
        """
        if self.model_class == BertForSequenceClassification:
            self.model = BertForSequenceClassification.from_pretrained(self.vocab, num_labels=2,
                                                                       output_attentions=False,
                                                                       output_hidden_states=False).to(self.device)
        else:
            self.model = self.model_class(self.vocab).to(self.device)

    def balance_data(self, df: pd.DataFrame, sz=None):
        """
        Функция для балансировки датасета.

        Параметры:
            df : pd.Dataframe
                Датасет с колонками:
                    comment: str
                        текст комментария
                    toxic: float
                        токсичность комментария (0 - нетоксичный, 1 - токсичный)
                    sz: int | None (default None)
                        желаемый размер датасета (если датасет нужно уменьшить),
                        если None - датасет принимает размер наименьшего класса * 2

        Ничего не возвращает
        """
        big_class, small_class = df["toxic"].value_counts().index
        nontoxic_sample = df[df["toxic"] == big_class].sample(sz // 2 if sz else len(df[df["toxic"] == small_class]))
        toxic_sample = df[df["toxic"] == small_class]
        if sz and sz < len(toxic_sample):
            toxic_sample = toxic_sample.sample(sz)
        return pd.concat((toxic_sample, nontoxic_sample))

    def get_model(self):
        return self.model

    def save_state_dict(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model(self, path):
        torch.save(self.model, path)

    def train(self, train_csv: str, val_csv: str, lr=5e-6, epochs=4, name='ToxicCommentClassifier',
              balance=False, balance_sz=None, get_best=True, wandb_logging=False):
        """
        Метод для обучения модели

        Параметры:
            train_csv: str
                путь к файлу с данными на обучение
            val_csv: str
                путь к файлу с данными на валидацию
            lr: float (default: 5e-6)
                learning rate для оптимизатора Adam
            epochs: int
                кол-во эпох
            name: str (default "ToxicCommentClassifier")
                название папки с промежуточными лучшими моделями обучения (при get_best=True),
                имя проекта в wandb (при wandb_logging=True), иначе не используется
            balance: bool (default: False)
                балансировать ли датасет на обучение (выровнять ли количество экземпляров целевых классов в датасете)
            balance_sz: int (default: None)
                размер, до которого нужно уменьшить датасеты (при balance=True)
                если None, то целевые классы будут приведены к размеру меньшего из них
            get_best: bool (default: True)
                нужно ли вернуть лучшую модель, если False вернет модель м последней эпохи и не будет локально
                сохранять сохранять промежуточные лучшие модели
            wandb_logging: bool (default: False)
                нужно ли логировать в wandb (требует установка через pip install wandb и
                аутентификация в wandb аккаунт через wandb login {ключ аутентификации})

        Возвращает обученную модель
        """

        train_df = load_csv_as_df(train_csv, {"comment": str, "toxic": int})
        val_df = load_csv_as_df(val_csv, {"comment": str, "toxic": int})

        if balance:
            train_df = self.balance_data(train_df, balance_sz)

        self.__init_model()
        self.history = pd.DataFrame(index=np.arange(1, epochs+1), columns=["train_loss", "train_acc", "train_f1_score",
                                                                           "val_loss", "val_acc", "val_f1_score"])
        self.history.index.name = 'epoch'
        loss_fn = nn.CrossEntropyLoss()
        opt = Adam(self.model.parameters(), lr=lr)
        tokenizer = BertTokenizer.from_pretrained(self.vocab)

        train = Dataset(tokenizer, train_df, self.token_len)
        val = Dataset(tokenizer, val_df, self.token_len)

        train_dataloader = DataLoader(train, batch_size=self.batch_sz, shuffle=True)
        val_dataloader = DataLoader(val, batch_size=self.batch_sz)

        log_template = "\nEpoch {}/{}:\n\ttrain_loss: {:0.4f}\t train_acc: {:0.4f}\t train_f1_score: {:0.4f}\n" \
                       "\tval_loss: {:0.4f}\t val_acc: {:0.4f}\t val_f1_score: {:0.4f}"
        name_template = "./{name}/e{ep}_loss{loss:0.4f}.pth"
        try:
            create_folder(name)
            if wandb_logging:
                run = wandb.init(project=name, config={"learning_rate": lr, "epochs": epochs,
                                                       "batch_size": self.batch_sz, "token_length": self.token_len,
                                                       "vocabulary": self.vocab, "model_type": type(self.model)})

            for ep in tqdm(range(epochs)):
                self.model.train()

                self.__epoch(dataloader=train_dataloader, loss_fn=loss_fn, ep=ep, opt=opt)
                self.model.eval()

                with torch.no_grad():
                    self.__epoch(dataloader=val_dataloader, loss_fn=loss_fn, ep=ep)

                tqdm.write(log_template.format(ep + 1, epochs, *self.history.iloc[ep].values))

                if wandb_logging:
                    run.log(dict(self.history.iloc[ep]))
                if get_best and self.history.iloc[ep]["val_loss"] == self.history["val_loss"].min():
                    model_name = name_template.format(name=name, ep=ep + 1, loss=self.history.iloc[ep]["val_loss"])
                    self.save_state_dict(model_name)

        except (Exception, KeyboardInterrupt):
            print(traceback.format_exc())
        finally:
            if get_best:
                self.load_state_dict(model_name)
            if wandb_logging:
                run.finish(quiet=True)
            return self.model
