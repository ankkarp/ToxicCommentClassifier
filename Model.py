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

from BertClassifier import BertClassifier
from Dataset import Dataset
from Tester import Tester
from metrics import f1_score
from utils import create_folder, read_csv_as_dtypes


class Trainer(Tester):
    def __init__(self, model_class=BertForSequenceClassification, vocab='DeepPavlov/rubert-base-cased-conversational',
                 token_len=64, batch_sz=16):
        logging.set_verbosity_warning()
        self.model_class = model_class
        Tester.__init__(self, model=None, vocab=vocab, token_len=token_len, batch_sz=batch_sz)
        self.history = pd.DataFrame(columns=["train_loss", "train_acc", "train_f1_score"
                                                                        "val_loss", "val_acc", "val_f1_score"])

    def get_history(self):
        return self.history

    def plot_history(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        fig = plt.figure(figsize=(30, 12))
        for i, metric in ("loss", "acc", "f1_score"):
            ax = fig.add_subplot(1, 3, i)
            ax.title.set_text(metric)
            ax.plot([i for i in range(1, len(self.history.index) + 1)], self.history[('_'.join("train", metric))],
                    label="train")
            ax.plot([i for i in range(1, len(self.history.index) + 1)], self.history[('_'.join("train", metric))],
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
            print(output, y_batch)
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

    def balance(self, df, sz):
        nontoxic_sample = df[df["toxic"] == 0].sample(sz // 2 if sz else len(df[df["toxic"] == 1]))
        toxic_sample = df[df["toxic"] == 1]
        if sz or sz < len(toxic_sample):
            toxic_sample = toxic_sample.sample(sz)
        return pd.concat((toxic_sample, nontoxic_sample))

    def train(self, train_csv: str, val_csv: str, balance=False, size=None,
              return_best=True, lr=5e-6, epochs=4, name='ToxicCommentClassifier',
              display_plots=False, wandb_logging=False):

        train_df = read_csv_as_dtypes(train_csv, {"comment": str, "toxic": int})
        val_df = read_csv_as_dtypes(val_csv, {"comment": str, "toxic": int})
        print(val_df.info())

        if balance:
            train_df = self.balance(train_df, size)
            val_df = self.balance(val_df, size)

        self.init_model()
        if len(self.history):
            self.history = pd.DataFrame(columns=self.history.columns)
        loss_fn = nn.CrossEntropyLoss()
        opt = Adam(self.model.parameters(), lr=lr)
        tokenizer = BertTokenizer.from_pretrained(self.vocab)

        train = Dataset(tokenizer, train_df, self.token_len)
        val = Dataset(tokenizer, val_df, self.token_len)

        train_dataloader = DataLoader(train, batch_size=self.batch_sz, shuffle=True)
        val_dataloader = DataLoader(val, batch_size=self.batch_sz)

        log_template = "\nEpoch {}/{}:\n\ttrain_loss: {:0.4f}\t train_acc: {:0.4f}\t train_f1_score: {:0.4f}\n" \
                       "\t\tval_loss: {:0.4f}\t val_acc: {:0.4f}\t val_f1_score: {:0.4f}"
        name_template = "./{name}/e{ep}_loss{loss:0.4f}.h5"
        try:
            create_folder(name)
            if wandb_logging:
                run = wandb.init(project=name)
                run.name = name

            for ep in tqdm(range(epochs)):
                self.model.train()

                self.__epoch(dataloader=train_dataloader, loss_fn=loss_fn, ep=ep, opt=opt)
                self.model.eval()

                with torch.no_grad():
                    self.__epoch(dataloader=val_dataloader, loss_fn=loss_fn, ep=ep, opt=opt)

                if display_plots:
                    self.plot_history()
                tqdm.write(log_template.format(ep + 1, epochs, *self.history[ep].values))
                print(self.history[ep])

                if wandb_logging:
                    run.log(dict(self.history[ep]["train_loss"]))
                if self.history.iloc[ep]["val_loss"] == self.history["val_loss"].min():
                    model_name = name_template.format(name=name, ep=ep + 1, loss=self.history.iloc[ep]["val_loss"])
                    torch.save(self.model, model_name)

        except (Exception, KeyboardInterrupt):
            print(traceback.format_exc())
        finally:
            if return_best:
                if wandb_logging:
                    run.log_artifact(f'./{name}', name=f'{name}', type='model')
                    run.finish(quiet=True)
                self.model = torch.load(model_name)
            return self.model


if __name__ == "__main__":
    model = Trainer(token_len=2, batch_sz=2)
    model.train(train_csv='data_train[493].csv', val_csv='data_test_public[494].csv', balance=True, size=16)
