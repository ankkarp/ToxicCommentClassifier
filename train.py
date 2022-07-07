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
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import torch
import numpy as np


def train(model, loss_fn, opt, train_dataloader, val_dataloader, lr, epochs):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    name = f'bert_multilang_uncased_lr{lr}'

    if os.path.exists(f'./{name}'):
        os.chdir(f'./{name}')
        for f in os.listdir(f'.'):
            os.remove(f)
        os.chdir("..")
    else:
        os.mkdir(f'./{name}')

    run = wandb.init(project='CISM_NLP')
    run.name = name
    log_template = "\nEpoch {ep} train_loss: {t_loss:0.4f} val_loss: {v_loss:0.4f} train_acc: {t_acc:0.4f} val_acc: {v_acc:0.4f}"
    name_template = "./{name}/e{ep}_loss{loss:0.4f}.h5"
    train_acc, train_loss, val_acc, val_loss = [], [], [], []

    for ep in tqdm(range(epochs)):

        batch_acc, batch_loss = [], []

        model.train()
        for X_batch, Y_batch in train_dataloader:
            Y_batch = Y_batch.to(DEVICE)
            mask = X_batch['attention_mask'].to(DEVICE)
            input_id = X_batch['input_ids'].squeeze(1).to(DEVICE)

            output = model(input_id, mask)

            loss = loss_fn(output, Y_batch)
            batch_loss.append(loss.item())

            acc = (output.argmax(dim=1) == Y_batch).sum().item() / len(Y_batch)
            batch_acc.append(acc)

            model.zero_grad()
            loss.backward()
            opt.step()

        train_acc.append(np.mean(batch_acc))
        train_loss.append(np.mean(batch_loss))

        batch_acc, batch_loss = [], []
        model.eval()
        with torch.no_grad():

            for X_batch, Y_batch in val_dataloader:
                Y_batch = Y_batch.to(DEVICE)
                mask = X_batch['attention_mask'].to(DEVICE)
                input_id = X_batch['input_ids'].squeeze(1).to(DEVICE)

                output = model(input_id, mask)

                loss = loss_fn(output, Y_batch)
                batch_loss.append(loss.item())

                acc = (output.argmax(dim=1) == Y_batch).sum().item() / len(Y_batch)
                batch_acc.append(acc)

        val_acc.append(np.mean(batch_acc))
        val_loss.append(np.mean(batch_loss))

        if os.name == 'posix':
            os.system('clear')
        else:
            os.system('cls')
        fig = plt.figure()
        loss_ax = fig.add_subplot(121)
        loss_ax.plot([i for i in range(1, len(train_loss) + 1)], train_loss,
                     label="train")
        loss_ax.plot([i for i in range(1, len(val_loss) + 1)], val_loss,
                     label="val")
        loss_ax.legend()
        acc_ax = fig.add_subplot(122)
        acc_ax.plot([i for i in range(1, len(train_acc) + 1)], train_acc,
                    label="train")
        acc_ax.plot([i for i in range(1, len(val_acc) + 1)], val_acc,
                    label="val")
        acc_ax.legend()
        plt.show()
        run.log({'train_loss': train_loss[-1], 'val_loss': val_loss[-1],
                 'train_acc': train_acc[-1], 'val_acc': val_acc[-1]})
        tqdm.write(log_template.format(ep=ep + 1, t_loss=train_loss[-1], t_acc=train_acc[-1],
                                       v_loss=val_loss[-1], v_acc=val_acc[-1]))
        if val_loss[-1] == min(val_loss):
            model_name = name_template.format(name=name, ep=ep + 1, loss=val_loss[-1])
            torch.save(model, model_name)
        run.log_artifact(f'./{name}', name=f'bert_multilang_uncased_lr{lr}type='model')
        run.finish(quiet=True)
        # return model
        return torch.load(model_name)