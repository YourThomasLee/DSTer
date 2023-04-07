#!/usr/bin/env python
#-*- encoding: utf-8 -*-
from typing import overload
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random

from base import BaseDataLoader
from copy import deepcopy as dc

# user package
from utils import util
from trainer.dataset import MultiWOZ

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def collate_oracle(batch):
    #str data
    dial_id = [item['dial_id'] for item in batch]
    turn_id = [item['turn_id'] for item in batch]
    # text/number to tensor
    with torch.no_grad():
        history_text = pad_sequence([torch.tensor(item['history_text'], dtype=int) for item in batch],batch_first=True, padding_value=0)
        history_text_mask = (history_text == 0).unsqueeze(-2)
        history_roles = pad_sequence([torch.tensor(item['history_roles'], dtype=int) for item in batch],batch_first=True, padding_value=0)
        current_text = pad_sequence([torch.tensor(item['current_text'], dtype=int) for item in batch],batch_first=True, padding_value=0)
        current_text_mask = (current_text == 0).unsqueeze(-2)
        current_roles = pad_sequence([torch.tensor(item['current_roles'], dtype=int) for item in batch],batch_first=True, padding_value=0)
        #nums
        previous_state = torch.tensor([item["previous_state"] for item in batch])
        current_state = torch.tensor([item["current_state"] for item in batch])
    
    return {
        "dial_id": dial_id,
        "turn_id": turn_id,
        "previous_state": previous_state,
        "current_state": current_state,
        "history_text": history_text,
        "history_text_mask": history_text_mask,
        "history_roles": history_roles,
        "current_text": current_text,
        "current_text_mask": current_text_mask,
        "current_roles": current_roles,
    }

class MultiWOZDataLoader(DataLoader):
    """
    MultiWOZ(8 domains) data loading using BaseDataLoader
    Description: Multi-Domain Wizard-of-Oz dataset (MultiWOZ), a fully-labeled collection of human-human written conversations spanning over multiple domains and topics. At a size of 10k dialogues, it is at least one order of magnitude larger than all previous annotated task-oriented corpora.
        - There are 3,406 single-domain dialogues that include booking if the domain allows for that and 7,032 multi-domain dialogues consisting of at least 2 up to 5 domains. To enforce reproducibility of results, the corpus was randomly split into a train, test and development set. The test and development sets contain 1k examples each. Even though all dialogues are coherent, some of them were not finished in terms of task description. Therefore, the validation and test sets only contain fully successful dialogues thus enabling a fair comparison of models. There are no dialogues from hospital and police domains in validation and testing sets.
        - 2.0为多领域的对话数据集, 人-人对话数据集, 对话和标注都由人完成, 存在延迟标注(标注轮次较晚), 多重标注(一个值被标注在多个槽位中), 错误标注(值和槽并不匹配), 语法错误, 标注错漏.
        - 2.1: 观察到2.0版本存在对话状态和对话话语中存在错误标注/噪声, 另外统一了已有的前人对数据集的扩充(例如对话动作扩充等等); 
            1. 对于话语上的自由性, 许多值并没有和ontology中的设定值完全一样, 对于槽值进行了规整;
            2. 使用Levenshtein distance小于3的值作为候选槽值, 然后构建完槽值后对句子进行匹配替换, 生成新的对话语句
            3. 对每个槽增加了简短的说明, 考虑支持小样本/迁移(冷启动)等状况下模型设计和评估
            4. 对话动作的标注(主要就是合并了前人工作的成果)
        - 2.2: 进一步修正17.3%的对话的状态; 修订了ontology去除大量非合法的值. 另外对一些槽的值进行话语上的区间标注
            1. 修复以下错误: 过早标注, 数据库标注(话语中未被提及, 只是包含在数据库中), 书写错误, 隐含的时间处理; 
    woz data url: https://github.com/budzianowski/multiwoz.git
    dialogue state tracking: 
        formulation: $State_n = f(State_{n-1}, history_{n-1}, utterance_{n}, action_{n-1})$
    data formulation: previous_state, dialogue history, utterance, previous_action, cur_state
    torchtext usage example: 
        1. https://wangjiosw.github.io/2020/02/29/deep-learning/torchtext_use/index.html
        2. https://zhuanlan.zhihu.com/p/147019255
    task description: 
        数据: 领域, 槽位, 领域候选数据(正排数据如何组织, 依赖关系如何建模)
        机制设计: 对话状态系统机制的设计(数据交融, 模型设计和推理, 线上如何使用, 在线如何训练)
    此处只考虑文本对话数据: 对话, 状态
    """
    def __init__(self, 
        original_data_dir,
        data_dir, 
        batch_size, 
        shuffle=True, 
        validation_split=0.0, 
        num_workers=1, 
        state_domain_card=256,
        collate_fn=collate_oracle):

        self.original_data_dir = original_data_dir
        self.data_dir = data_dir
        self.datasets = MultiWOZ(original_data_dir, data_dir, state_domain_card)
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(dataset = self.datasets.train_data,
                    **self.init_kwargs,
            )

        # max history text length: 791
        # max history + current text: 831
        # max current text length:  129
    
    def split_validation(self):
        return DataLoader(dataset = self.datasets.vali_data, **self.init_kwargs)


import datasets
from transformers import AutoTokenizer, AddedToken
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
class WOZDataLoader(BaseDataLoader):
    """
    original data schema:
        dialogue_id,string
        turn_id,int
        sys_utt,string
        usr_utt,string
        states_21,dict
        states_22,dict
        states_23,dict
        states_24,dict
    preprocess:
        remove_last_turn
        remove_empty_state
        generate_dialogue_history => context
        generate_previous_state => prev_states
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split=0.1, num_workers=1, dialogue_history_pool= 5, version=23, training=True):
        dataset = self.preprocess(datasets.load_dataset(path=data_dir, split="train" if training else "test"), dialogue_history_pool, version=version) # train dataset
        validation = self.preprocess(datasets.load_dataset(path=data_dir, split="validation"), dialogue_history_pool, version=version) # validation dataset
        self.add_tokens(['[end of text]'])
        self.train_sampler = BatchSampler(RandomSampler(dataset), batch_size=batch_size, drop_last=False)
        self.valid_sampler = BatchSampler(SequentialSampler(validation), batch_size=batch_size, drop_last=False)
        super().__init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.collate_woz)

    def add_tokens(self, token_list):
        for it in token_list:
            tokenizer.add_tokens(AddedToken(content=it, single_world=False))
            # self.tokenizer.add_tokens("[special_token]")

    def _split_sampler(self, split):
        return self.train_sampler, self.valid_sampler
    
    def preprocess(data, max_hist_len = 5, version = 22):
        data_idx = dict()
        for idx, turn in enumerate(data):
            # remove last turn that usr_utt is none
            if turn['usr_utt'] == 'none': 
                continue
            # remove empty state
            if turn['states_dict21'] == None:
                continue
            k = turn["dialogue_id"]
            if k not in data_idx: data_idx[k] = list()
            assert(turn['turn_id'] > data_idx[k][-1])
            data_idx[k].append(idx)
        
        data_ret = list()
        for d in data_idx:
            context = list()
            prev = None
            for t, idx in enumerate(data_idx[d]):
                turn = data[idx]
                sys_utt = "none"
                assert(turn["usr_utt"] != "none")
                data_ret.append({
                    "dialogue_id": d,
                    "turn_id": t,
                    "context": "".join(context[-max_hist_len:]),
                    "sys_utt": sys_utt,
                    "usr_utt": turn["usr_utt"],
                    "prev_states": prev if prev is not None else {k: "none" for k in turn["states_" + str(version)]},
                    "states": turn["states_" + str(version)],
                })
                prev = turn["states_" + str(version)]
                context.append(sys_utt + " [end of text] " + turn["usr_utt"] +" [end of text]")
        return data_ret


    def collate_woz(batch):
        # dialogue_id, turn_id, context, sys_utt, usr_utt, 
        # prev_states, states
        #str data
        dial_id = [item['dial_id'] for item in batch]
        turn_id = [item['turn_id'] for item in batch]
        f = lambda l: tokenizer(l, truncation=True, padding=True, return_tensor = "pt")
        with torch.no_grad():
            context_tokens = f([item["context"] for item in batch])
            usr_utt_tokens = f([item["usr_utt"] for item in batch])
            sys_utt_tokens = f([item["sys_utt"] for item in batch])
            states = f([list(item["states"].keys()) for item in batch])
            
            
        
    
        return {
            "dial_id": dial_id,
            "turn_id": turn_id,
            "previous_state": previous_state,
            "current_state": current_state,
            "history_text": history_text,
            "history_text_mask": history_text_mask,
            "history_roles": history_roles,
            "current_text": current_text,
            "current_text_mask": current_text_mask,
            "current_roles": current_roles,
        }
