import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from BertClassifier import BertClassifier
from Dataset import Dataset


class Tester:
    def __init__(self, model, vocab, token_len=64, batch_sz=16):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Используется {self.device}')
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(vocab)
        self.model = model
        self.token_len = token_len
        self.batch_sz = batch_sz

    def test(self, test_csv: str, export_to_csv=None):
        test_df = pd.read_csv(test_csv)
        if "text" not in test_df.columns:
            test_df.rename(columns={"comment": "text", "toxic": "class_true"}, inplace=True)
            test_df["class_true"].replace({0: "non-toxic", 1: "toxic"}, inplace=True)
        test = Dataset(test_df, self.token_len)
        test_dataloader = DataLoader(test, batch_size=self.batch_sz)
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, _) in test_dataloader:
                mask = x_batch['attention_mask'].to(self.device)
                input_id = x_batch['input_ids'].squeeze(1).to(self.device)
                output = self.model(input_id, mask) if isinstance(self.model, BertClassifier) else \
                    self.model(input_id, mask, return_dict=False)[0]
                test_df[i * self.batch_sz: (i + 1) * self.batch_sz]["class_prediction"] = output.argmax(dim=1)
                test_df[i * self.batch_sz: (i + 1) * self.batch_sz]["probabilities"] = output.argmax(dim=0)
        if export_to_csv:
            test_df.to_csv(f"./{export_to_csv}")
        return test_df
