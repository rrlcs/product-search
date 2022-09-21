import torch
from torch.utils.data import DataLoader, Dataset
from model import BERT_Cat
from transformers import AutoTokenizer

import pandas as pd

# honestly not sure if that is the best way to go, but it works :)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = BERT_Cat.from_pretrained(
    "sebastian-hofstaetter/distilbert-cat-margin_mse-T2-msmarco")

# print(model)
data = pd.read_csv('triples.tsv', sep='\t')
# print(data.head())
query = data['query']
positive = data['positive']
negative = data['negative']


class TrainDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.query = df['query'].values
        self.positive = df['positive'].values
        self.negative = df['negative'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        query = self.query[idx]
        positive = self.positive[idx]
        negative = self.negative[idx]

        return query, positive, negative


# query_loader = DataLoader(query, batch_size=4)
# positive_loader = DataLoader(positive, batch_size=4)
# negative_loader = DataLoader(negative, batch_size=4)
train_data = TrainDataset(data)
# print(type(train_data))
train_loader = DataLoader(train_data, batch_size=4)
print(train_loader.batch_size)

p1_score = 0
p2_score = 0
for batch_idx, sample in enumerate(train_loader):
    p1 = tokenizer(sample[0], sample[1], return_tensors='pt', padding=True)
    p2 = tokenizer(sample[0], sample[2], return_tensors='pt', padding=True)

    p1_score += model(p1)
    p2_score += model(p2)

print(p1_score, p2_score)
