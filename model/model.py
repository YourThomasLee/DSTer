import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as cp
import math
from base import BaseModel
from model.position_encoding import PositionalEncoding
from model.attention import MultiHeadedAttention
from model.feed_forward_net import FeedForwardNet

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model,padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class EmbeddingLayer(nn.Module):
    def __init__(self,text_vocab_size, 
                state_label_num=35,
                state_domain_card=[255],
                d_model=64, 
                padding_idx = 0,
                dropout=0.1):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        max_state_domain_card = max(state_domain_card) + 1
        
        self.text_embedding = nn.Embedding(text_vocab_size, d_model, padding_idx=padding_idx)
        self.position_embedding = PositionalEncoding(d_model)
        # state_label_num * state_domain_card * d_model
        state_embedding_matrix = torch.randn((state_label_num, max_state_domain_card, d_model), requires_grad=True)
        # 用embedding做预测
        state_mask = torch.tensor([[1 for j in range(state_domain_card[i] + 1)] + [0 for j in range(max(state_domain_card) - state_domain_card[i])]
            for i in range(state_label_num)], requires_grad=False)
        state_embedding_matrix =  state_embedding_matrix * state_mask.view(state_label_num, max(state_domain_card) + 1, 1)
        self.state_embedding = nn.Parameter(state_embedding_matrix) #label embedding
        self.register_parameter("state_embedding", self.state_embedding)
        
    def transform_state(self, state):
        batch_size, state_cardinity, _ = state.shape
        d_model = self.state_embedding.shape[-1] 
        embedding = torch.zeros((batch_size, state_cardinity, d_model), device=self.state_embedding.device)
        for i in range(self.state_embedding.shape[0]):
            # batch_size * 1 * d_model = batch_size * 1 * state_domain_card @ 1 * state_domain_card * d_model
            embedding[:, i, :] = state[:, i, :] @ self.state_embedding[i, :, :]
        return self.dropout(embedding)

    def transform_utterance(self, utterance):
        utterance_embedding = self.text_embedding(utterance)
        return self.position_embedding(utterance_embedding)

    def forward(self, batch):
        batch['pre_state_embed'] = self.transform_state(batch['previous_state'])
        batch['cur_state_embed'] = self.transform_state(batch['current_state'])
        batch['hist_text_embed'] = self.transform_utterance(batch['history_text'])
        batch['curr_text_embed'] = self.transform_utterance(batch['current_text'])
        # 加入话语的角色编码
        batch['hist_text_embed'] += self.transform_utterance(batch['history_roles'])
        batch['curr_text_embed'] += self.transform_utterance(batch['current_roles'])
        # dropout
        batch['hist_text_embed'] = self.dropout(batch['hist_text_embed'])
        batch['curr_text_embed'] = self.dropout(batch['curr_text_embed'])
        return batch

class AttentionLayer(nn.Module):
    def __init__(self, attn_heads, d_model, dropout):
        super(AttentionLayer, self).__init__()
        self.adapt_layer1 = nn.Sequential(nn.LayerNorm(d_model),nn.Dropout(dropout))
        self.adapt_layer2 = nn.Sequential(nn.LayerNorm(d_model),nn.Dropout(dropout))
        self.attention = MultiHeadedAttention(attn_heads, d_model, dropout)
        self.ffn = FeedForwardNet(d_model, 2 * d_model, dropout)
    
    def forward(self, query, key, value, mask):
        x = query + self.attention(query, key, value, mask)
        x = self.adapt_layer1(x)
        x = x + self.ffn(x)
        x = self.adapt_layer2(x)
        return x


class ModelLayer(nn.Module):
    """
    "previous_state": previous_state,
    "current_state": current_state,
    "history_text": history_text,
    "current_text": current_text,
    可能存在的问题：
    - 开始轮次的对话标签被后续轮次使用导致重复训练，预计会影响网络的性能
    - 每个轮次使用一次不能够充分利用样本（历史 -> 历史状态， 当前轮次话语 -> 当前状态）提升样本的利用率
    - 问题建模的合理性没有被实验评估

    需要做的事情
    0. 最后的任务设定：分类、检索
    1. 基于对话文本直接去做标签的预测 f(U_{1:t-1}) = S_{t-1} 
    2. 基于历史状态和当前对话更新状态 f(S_{t-1}, U_{t}) = S_{t}
    3. 在2的框架中使用历史对话和当前轮次对话的信息融合 U_{t-1} = f(U_{1:t-1}, U_{t})
    4. 在2/3的框架中使用历史状态对当前轮次对话进行信息提取 U_{t-1} = f(U_{t}, S_{t-1})
    5. Wide&deep 基于位置对句子的表示进行学习，并最后融入到最后的预测状态中
    6. 分析embedding各个维度的数值，考虑做个维度归一化/梯度过滤器增强embedding维度的利用率
    7. 对于非自回归的对话状态追踪网络，可以考虑使用CRF来缓解槽值部分预测正确的问题，因为槽值组合其实是比较有限的
    8. 对于多标签中不同标签中的类别不平衡问题，是否能考虑对于每个标签设置一种技术来提高一个标签下的类别预测效果
    """
    def __init__(self, 
        attn_heads,
        d_model,
        dropout,
        embedding = None,
        **kargs):
        super(ModelLayer, self).__init__()
        assert(embedding != None)
        self.embedding = embedding
        # 文本内容的建模考虑使用 注意力 + FFN
        attn = AttentionLayer(attn_heads, d_model, dropout)
        self.self_attn = nn.ModuleList([cp(attn) for _ in range(2)])
        self.inter_attns = nn.ModuleList([cp(attn) for _ in range(2)])
        self.state_fusion = cp(attn)

    def forward(self, embedding):
        feature = dict()
        hist_text = embedding['hist_text_embed']
        history_text_mask = embedding['history_text_mask']
        current_text = embedding['curr_text_embed']
        current_text_mask = embedding['current_text_mask']
        
        #self text attention layer
        for layer in self.self_attn:
            hist_text = layer(hist_text, hist_text, hist_text, history_text_mask)
            current_text = layer(current_text, current_text, current_text, current_text_mask)
        # 将所有学习的文字表示转换为状态的预测
        # batch_size * text_length * d_model
        # state_label * state_domain_card * d_model -> batch_size * state_label * d_model
        batch_size = hist_text.shape[0]
        label_num, domain_card, d_model = self.embedding.state_embedding.shape
        state_feature = self.embedding.state_embedding.mean(dim = 1).view(1,label_num, d_model).repeat(batch_size, 1, 1)
        # batch_size * state_label * d_model
        # *** naive fusion
        hist_state = self.state_fusion(state_feature, hist_text, hist_text, history_text_mask)
        update_state = self.state_fusion(state_feature, current_text, current_text, current_text_mask)
        feature['hist_state'] = hist_state
        feature['update_state'] = update_state

        # history, current turn interaction attention layer
        current_text = self.inter_attns[0](current_text, hist_text, hist_text, history_text_mask)
        # state, text interaction attention layer
        curr_state = self.inter_attns[1](embedding['pre_state_embed'], current_text, current_text, current_text_mask)
        feature['current_text'] = current_text
        feature['curr_state'] = curr_state
        return feature


class TaskLayer(nn.Module):
    def __init__(self, 
            state_label_num, 
            state_domain_card, 
            embedding,
            dropout
            ):
        super(TaskLayer, self).__init__()
        # naive prediction
        self.state_label_num = state_label_num
        self.state_domain_card = state_domain_card
        # batch_size * state_label_num * d_model
        self.embedding = embedding
        self.dp = nn.Dropout(p=dropout)

    def top_match(self, state_fea):
        state_label_num, value_num, d_model = self.embedding.state_embedding.shape
        # state_label_num * state_domain_card * d_model -> 1 * state_label_num * d_model * state_domain_card
        p = self.embedding.state_embedding.transpose(-2, -1).view(1, state_label_num, d_model, value_num)
        # result: batch_size * state_label_num, state_domain_card
        logit = (state_fea.unsqueeze(2) @ p).squeeze(2)
        norms = torch.norm(logit, p=2, dim=-1, keepdim=True) + 1e-7
        logit = torch.div(logit, norms)# 2范式打压极值,考虑收敛太慢了，下降为1范数
        logit = logit.masked_fill(logit == 0, -1e9)# label mask
        return logit

    def forward(self, feature):
        pred = dict()
        pred['hist_state'] = self.top_match(feature['hist_state'])
        pred['update_state'] = self.top_match(feature['update_state'])
        pred['curr_state'] = self.top_match(feature['curr_state'])
        # current_text = feature['current_text']
        return pred

class DSTModel(BaseModel):
    def __init__(self, text_vocab_size, 
            state_label_num, 
            state_domain_card, 
            d_model, 
            padding_idx,
            dropout,
            attn_heads,
            **kargs
            ):
        super(DSTModel, self).__init__()
        self.embedding = EmbeddingLayer(text_vocab_size, state_label_num, state_domain_card, d_model, padding_idx)
        self.model = ModelLayer(attn_heads, d_model, dropout, self.embedding)
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



