#!/usr/bin python
# -*- encoding: utf-8 -*-
"""
The objective of this file is provide an unified way of reading dialogue data into program \
and given a simple/easy-using access to get the information to train an dialogue state tracking model
"""
from datasets import load_dataset
import json
import sys
import pdb
from transformers import AutoTokenizer, AddedToken
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
# dataset1 = load_dataset("adamlin/multiwoz_dst")
# dataset2 = load_dataset("pietrolesci/multiwoz_all_versions")

class DialogueData:
    """
    pass
    """
    def __init__(self, ):
        pass

  
def write_to_file(name, d):
    for k in d:
        d[k].set_format("pandas")
        d[k].to_csv(name + k + ".csv")

if __name__ == "__main__":
    d1 = load_dataset(path="./multiwoz/multiwoz_dst", split="validation")
    print(d1)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    print(" \n", len(tokenizer.get_vocab()))
    # d2 = load_dataset("pietrolesci/multiwoz_all_versions", split=None)
    # write_to_file("./all_version/", d2)