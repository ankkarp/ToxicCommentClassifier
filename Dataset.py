import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    """Датасет для модели BERT"""

    def __init__(self, tokenizer, data, token_len: int):
        """Конструктор датасета, сохраняет целевые классы, разбивает тексты на токены и сохраняет их"""
        self.labels = [float(label) for label in data['toxic']]
        self.texts = [tokenizer(text, padding='max_length', max_length=token_len, truncation=True, return_tensors="pt")
                      for text in data['comment']]

    def classes(self):
        """Метод получения всех истинных классов"""
        return self.labels

    def __len__(self):
        """Метод получения длины датасета"""
        return len(self.labels)

    def get_batch_labels(self, idx):
        """Метод получения батча целевых классов"""
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        """Метод получения батча токенизированных текстов"""
        return self.texts[idx]

    def __getitem__(self, idx):
        """
        Метод итерирования датасета
        Возвращает батч токенизированных текстов и батч их истинных классов
        """
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y