# -*- encoding: utf-8 -*-
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class BertEmbedding(nn.Module):

    def __init__(self, bert_name, d_model, dropout):
        self.bert_layer = AutoModel.from_pretrained(bert_name)
        self.squeezer = nn.Linear(768, d_model)
        self.normalize_layer = nn.Sequential(nn.LayerNorm(d_model),nn.Dropout(dropout))

    def forward(self, text):
        bert_out = self.bert_layer(text).last_hidden_state
        mid_tensor = self.squeezer(bert_out) 
        return self.normalize_layer(mid_tensor) # batch_size * sentence_len * d_model

class CrossLayer(nn.Module):

    def __init__(self, d_model, dropout):
        pass

    def forward(self, ):
        pass

class TaskLayer(nn.Module):

    def __init__(self, ):
        pass

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
        self.embedding = BertEmbedding(bert_name, d_model, dropout)
        
        self.model = CrossLayer(attn_heads, d_model, dropout, self.embedding)
        self.task_layer = TaskLayer(state_label_num, state_domain_card, self.embedding, dropout)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch):
        embedding = self.embedding(batch)
        feature = self.model(embedding)
        pred = self.task_layer(feature)
        return pred
