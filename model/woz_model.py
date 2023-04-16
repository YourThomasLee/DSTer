# -*- encoding: utf-8 -*-
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as c

from base import BaseModel
from trainer.data_loaders import tokenizer
from model.attention import MultiHeadedAttention
from model.feed_forward_net import FeedForwardNet

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([c(module) for _ in range(N)])

class WozEmbedding(nn.Module):

    def __init__(self, bert_name, d_model, dropout):
        super(WozEmbedding, self).__init__()
        self.bert_layer = AutoModel.from_pretrained(bert_name)
        self.bert_layer.resize_token_embeddings(len(tokenizer))
        
    def forward(self, context_ids, cur_utt_ids, pre_states_ids):
        context_embed = self.bert_layer(context_ids)
        cur_utt_embed = self.bert_layer(cur_utt_ids)
        pre_states_embed = {k: self.bert_layer(v).pooler_output for k, v in pre_states_ids.items()}
        embeddings = {"context": context_embed,
                 "cur_utt": cur_utt_embed,
                 "pre_states": pre_states_embed
                }
        return embeddings # batch_size * sentence_len * 768

class CrossLayer(nn.Module):
    '''
    input:
        - context: user_utterrance + system_utterrance ... of previous turns => self_attention
        - usr_utt: user_utterance in current turn => attention(usr_utt, context, )
        - previous_states: belief_state in previous turn
        - slot_gate_logit
        - slot_value_logit
    '''

    def __init__(self, attn_heads, d_model, dropout):
        super(CrossLayer, self).__init__()
        self.attns = clones(MultiHeadedAttention(attn_heads, d_model, dropout), 3)
        self.normalize_layer = clones(nn.Sequential(
                                                nn.Linear(768, d_model), 
                                                nn.LayerNorm(d_model),
                                                nn.Dropout(dropout)
                                            ), 
                                        3)
        self.norms = clones(nn.LayerNorm(d_model), 2)
        self.ffns = clones(FeedForwardNet(d_model, int(1.5 * d_model), dropout), 2)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, embeddings, masks):
        i = 0
        fea = dict()
        for k,v in embeddings.items():
            if "last_hidden_state" in v.keys(): # cur_utt, context
                fea[k] = self.normalize_layer[i](v['last_hidden_state'])
            elif len(v.keys()) > 29: # slots 
                if k not in fea: fea[k] = dict()
                for slot, embed in v.items():
                    fea[k][slot] = self.normalize_layer[i](embed)
            i += 1
        pred_input = dict()
        # preNorm + residual connection
        cur_utt_attn = fea['cur_utt'] + self.attns[0](
                    fea['cur_utt'], fea['context'], fea['context'], 
                    masks['context']
                )
        pred_input['cur_utt'] = cur_utt_attn + self.ffns[0](self.norms[0](cur_utt_attn))
        pred_input['slots_gates'] = dict()
        for k,v in fea['pre_states'].items():
            v_attn = v + self.attns[1](
                    v.unsqueeze(-2), pred_input['cur_utt'], pred_input['cur_utt'], 
                    masks['cur_utt']
                ).squeeze(-2)
            pred_input['slots_gates'][k] = v_attn + self.ffns[1](self.norms[1](v_attn))
        return pred_input

class TaskLayer(nn.Module):
    '''
    input: 
        - context: utterance history
        - usr_utt: user_utterance
        - domains - slots: 
        - previous_states:
        - cur_states:
    '''
    def __init__(self, slots, slots_classification, d_model, dropout):
        super(TaskLayer, self).__init__()
        self.is_classification = nn.Parameter(torch.tensor([1 if i in slots_classification else 0 for i in slots]), requires_grad=False)
        self.gates_layer = nn.Linear(d_model, 2) # slot_gates_prediction
        self.values_layer = nn.Linear(d_model, 15) # slot_value_prediction

        

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input):
        logits = dict()
        slots_fea = torch.stack([v for k, v in input['slots_gates'].items()], dim=1)
        logits['slots_gates'] = self.gates_layer(slots_fea)
        logits['slots_values'] = self.values_layer(slots_fea)#.masked_fill(self.is_classification.unsqueeze(0).unsqueeze(-1) == 0, 0.0)
        return logits

class WozModel(BaseModel):
    def __init__(self, bert_name,
            domain_slots, 
            slots_classification, 
            d_model,
            dropout,
            attn_heads,
            **kargs
            ):
        super(WozModel, self).__init__()
        self.embedding = WozEmbedding(bert_name, d_model, dropout)
        self.cross_layer = CrossLayer(attn_heads, d_model, dropout)
        self.task_layer = TaskLayer(domain_slots, slots_classification, d_model, dropout)

    def forward(self, batch):
        embeddings = self.embedding(batch['context_ids'], 
                                   batch['cur_utt_ids'], 
                                   batch['pre_states_ids']
                                )
        masks = {"context": batch["context_mask"],
                "cur_utt": batch["cur_utt_mask"],
                "pre_states": batch["pre_states_mask"], 
                }
        pred_input = self.cross_layer(embeddings, masks)
        logits = self.task_layer(pred_input)
        return logits
