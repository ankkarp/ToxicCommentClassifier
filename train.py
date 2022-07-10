# import argparse
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-n', '--model_name', type=str, default='bert-base-multilingual-uncased',
#                         help='Pretrained BERT model name (from https://huggingface.co/models?other=bert)')
#     parser.add_argument('-s', '--save_folder', type=str, default='.')
#     parser.add_argument('-s', '--save_folder', type=str, default='.')

import os
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
from test import Tester
from metrics import f1_score
from utils import create_folder, load_csv_as_df


class Model(Tester):
    def __init__(self, model_class=BertForSequenceClassification, vocab='DeepPavlov/rubert-base-cased-conversational',
                 token_len=64, batch_sz=16):
        self.model_class = model_class
        Tester.__init__(self, model=None, vocab=vocab, token_len=token_len, batch_sz=batch_sz)
        self.history = None

    def get_history(self):
        return self.history

    def plot_history(self):
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

    def init_model(self):
        if self.model_class == BertForSequenceClassification:
            self.model = BertForSequenceClassification.from_pretrained(self.vocab, num_labels=2,
                                                                       output_attentions=False,
                                                                       output_hidden_states=False).to(self.device)
        else:
            self.model = self.model_class(self.vocab).to(self.device)

    def balance_data(self, df: pd.DataFrame, sz):
        nontoxic_sample = df[df["toxic"] == 0].sample(sz // 2 if sz else len(df[df["toxic"] == 1]))
        toxic_sample = df[df["toxic"] == 1]
        if sz or sz < len(toxic_sample):
            toxic_sample = toxic_sample.sample(sz)
        return pd.concat((toxic_sample, nontoxic_sample))

    def train(self, train_csv: str, val_csv: str, lr=5e-6, epochs=4, name='ToxicCommentClassifier',
              balance=False, size=None, get_best=True, wandb_logging=False):

        train_df = load_csv_as_df(train_csv, {"comment": str, "toxic": int})
        val_df = load_csv_as_df(val_csv, {"comment": str, "toxic": int})

        if balance:
            train_df = self.balance_data(train_df, size)
            val_df = self.balance_data(val_df, size)

        self.init_model()
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
        name_template = "./{name}/e{ep}_loss{loss:0.4f}.h5"
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
                    torch.save(self.model, model_name)

        except (Exception, KeyboardInterrupt):
            print(traceback.format_exc())
        finally:
            if get_best and wandb_logging:
                run.log_artifact(f'./{name}', name=f'{name}', type='model')
            if get_best:
                self.model = torch.load(model_name)
            if wandb_logging:
                run.finish(quiet=True)
            return self.model


if __name__ == "__main__":
    gc.collect()
    # model = Model(token_len=2, batch_sz=2)
    # model.train(train_csv='data_train[493].csv', val_csv='data_test_public[494].csv', balance=True, size=2)
    # print(model.test(test_csv='data_test_public[494].csv', export_file='res.csv'))
    tester = Tester(None, batch_sz=2, token_len=2)
    tester.load_model('ToxicCommentClassifier/e4_loss0.7007.h5')

    print(tester.test(test_csv='data_test_public[494].csv', export_file='res.csv', analyse=True))
