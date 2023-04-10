# -*- encoding: utf-8 -*-
from transformers import AutoModel, AutoTokenizer, AddedToken, BertForTokenClassification
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

# dataset = load_dataset("./data/multiwoz",split="train") # local dataset
dataset = load_dataset("pietrolesci/multiwoz_all_versions",split="train")
print(dataset)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_tokens(["<gogogo>"]) # 定义特殊单词
tokenizer.add_tokens(AddedToken(content="[You are good]", single_word=False))
# model = AutoModel.from_pretrained("bert-base-uncased")
# s1 = "you are successful in doing something while you fail to some other things"
# s2 = "you are right, but there is nothing useful!"
# def encode(examples):
#     return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")
# print("\n", tokenizer(s1, truncation=True, padding=True, return_tensors="pt"))
# input = tokenizer([s1, s2], truncation=True, padding=True, return_tensors="pt")
# print(input)
# print(model(input['input_ids']).last_hidden_state.shape, input['input_ids'].shape)
# dataset.map(encode, batched=True)
# print(dataset[0])
# rename the label colum to labels
# dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
# dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
def collate_woz(batch):
    # dialogue_id, turn_id, context, sys_utt, usr_utt, 
    # prev_states, states
    #str data
    dial_id = [item['dialogue_id'] for item in batch]
    turn_id = [item['turn_id'] for item in batch]
    f = lambda l: tokenizer(l, truncation=False, padding=True, return_tensors = "pt")
    with torch.no_grad():
        context_tokens = f([item["context"] for item in batch])
        usr_utt_tokens = f([item["usr_utt"] for item in batch])
        sys_utt_tokens = f([item["sys_utt"] for item in batch])
        states_text = f([k.replace("-", " ") for k in batch[0]["states"].keys()]) # state_text
        cur_states = {} # label -> label embedding
        for turn in batch:
            for k,v in zip(turn["states"]["slot_name"], turn["states"]["slot_value"]):
                if k not in cur_states:
                    cur_states[k] = list()
                cur_states[k].append("none" if len(v) == 0 else v[0])
            
        prev_states = {}# label -> label embedding
        for turn in batch:
            for k,v in zip(turn["prev_states"]["slot_name"], turn["prev_states"]["slot_value"]):
                if k not in prev_states:
                    prev_states[k] = list()
                prev_states[k].append("none" if len(v) == 0 else v[0])
    return {
        "dialogue_id": dial_id,
        "turn_id": turn_id,
        "context_ids": context_tokens["input_ids"],
        "context_mask": context_tokens["attention_mask"],
        "usr_utt_ids": usr_utt_tokens["input_ids"],
        "usr_utt_mask": usr_utt_tokens["attention_mask"],
        "sys_utt_ids": sys_utt_tokens["input_ids"],
        "sys_utt_mask": sys_utt_tokens["attention_mask"],
        "state_text": states_text["input_ids"],
        "state_mask": states_text["attention_mask"],
        "cur_state_tokens": cur_states,
        "pre_state_values": prev_states
    }
from torch.utils.data.sampler import BatchSampler, RandomSampler
batch_sampler = BatchSampler(RandomSampler(dataset), batch_size=32, drop_last=False)
dataloader = DataLoader(dataset, batch_sampler=batch_sampler)#, collate_fn = collate_woz)
max_len = 0
for idx, data in enumerate(dataloader):
    #max_len = max(max_len, data["context_ids"].shape[1])
    import pdb
    pdb.set_trace()

print(max_len)



