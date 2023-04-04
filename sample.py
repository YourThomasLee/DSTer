# -*- encoding: utf-8 -*-
import transormers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

daaset = load_dataset("glue", "mrpc", split="train")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.unique_no_split_tokens += ["<gogogo>"] # 定以单词

model = AutoModelSequenceClassification.from_pretrained("bert-base-cased")

def encode(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

dataset.map(encode, batched=True)
print(dataset[0])
# rename the label colum to labels
dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4)
dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)



