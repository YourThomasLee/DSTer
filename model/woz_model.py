# -*- encoding: utf-8 -*-
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as c

from base import BaseModel
from trainer.data_loaders import tokenizer
from model.attention import MultiHeadedAttention

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([c(module) for _ in range(N)])

class WozEmbedding(nn.Module):

    def __init__(self, bert_name, d_model, dropout):
        super(WozEmbedding, self).__init__()
        self.bert_layer = AutoModel.from_pretrained(bert_name)
        self.bert_layer.resize_token_embeddings(len(tokenizer))
        
    def forward(self, context_ids, usr_utt_ids, pre_states_ids):
        context_embed = self.bert_layer(context_ids)
        usr_utt_embed = self.bert_layer(usr_utt_ids)
        pre_states_embed = {k: self.bert_layer(v).pooler_output for k, v in pre_states_ids.items()}
        return context_embed, usr_utt_embed, pre_states_embed # batch_size * sentence_len * 768

class CrossLayer(nn.Module):
    '''
    input:
        - context: user_utterrance + system_utterrance ... of previous turns => self_attention
        - usr_utt: user_utterance in current turn => attention(usr_utt, context, )
        - previous_states: belief_state in previous turn
        - slot_gate_logit
        - slot_value_logit
    '''

    def __init__(self, d_model, dropout):
        super(CrossLayer, self).__init__()
        self.squeezer = clones(nn.Linear(768, d_model), 3)
        self.normalize_layer = clones(nn.Sequential(nn.LayerNorm(d_model),nn.Dropout(dropout)), 3)
        

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ):
        out = []
        for idx, item in enumerate([1, 2, 3]):
            out.append(self.normalize_layer[idx](self.squeezer[idx](item)))

class TaskLayer(nn.Module):
    '''
    input: 
        - context: utterance history
        - usr_utt: user_utterance
        - domains - slots: 
        - previous_states:
        - cur_states:
    '''
    def __init__(self, ):
        super(TaskLayer, self).__init__()
        pass

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, ):
        pass


class WozModel(BaseModel):
    def __init__(self, bert_name,
            state_label_num, 
            state_domain_card, 
            d_model,
            dropout,
            attn_heads,
            **kargs
            ):
        super(WozModel, self).__init__()
        self.embedding = WozEmbedding(bert_name, d_model, dropout)
        # self.model = CrossLayer(attn_heads, d_model, dropout, self.embedding)
        # self.task_layer = TaskLayer(state_label_num, state_domain_card, self.embedding, dropout)

    def forward(self, batch):
        context_tensor, usr_utt_tensor, pre_states_tensor = self.embedding(batch['context_ids'], 
                                   batch['usr_utt_ids'], 
                                   batch['pre_states_ids']
                                )
        # feature = self.model(embedding)
        # pred = self.task_layer(feature)
        # return pred
        pass
