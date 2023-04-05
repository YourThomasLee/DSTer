# -*- encoding: utf-8 -*-
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AddedToken
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

# dataset = load_dataset("glue", "mrpc", split="train")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.add_tokens(["<gogogo>"]) # 定义特殊单词
tokenizer.add_tokens(AddedToken(content="[You are good]", single_word=False))
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
s1 = "you are successful in doing something while you fail to some other things"
s2 = "you are right, but there is nothing useful!"


# def encode(examples):
#     return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")
print("\n", tokenizer([s1, s2], truncation=True, padding=True, return_tensors="pt"))
# dataset.map(encode, batched=True)
# print(dataset[0])
# rename the label colum to labels
# dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
# dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

# from torch.utils.data.sampler import BatchSampler, RandomSampler
# batch_sampler = BatchSampler(RandomSampler(dataset), batch_size=32, drop_last=False)
# dataloader = DataLoader(dataset, batch_sampler=batch_sampler)



