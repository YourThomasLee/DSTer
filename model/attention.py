import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy as c

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([c(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)  # query and key are d_k dimentional   query: batch*query_len*d_model   key: batch*key_len*d_model
    scores = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(d_k)#batch*query_len*key_len 
    #masked_fill： Fills elements of self tensor with value where mask is True.详见官方文档
    if mask is not None:
        try:
            scores = scores.masked_fill(
                mask == 0, -1e9
            )  #-1e9: 可以理解为负无穷，score为负无穷，则经由softmax后相应的概率为0，相应的value不起作用，达到mask的作用
        except:
            print("attention.py line 22")
            import pdb
            pdb.set_trace()
    p_attn = F.softmax(scores, dim=-1)  #attention的概率分布
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value), p_attn  #返回attention向量的结果和attention的概率分布

class MultiHeadedAttention(nn.Module):
    def __init__(self, heads_num, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert(d_model % heads_num == 0) #head数要能够被输入的维度所整除
        self.d_k = d_model // heads_num
        self.heads_num = heads_num
        self.linears = clones(nn.Linear(d_model, d_model),3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
    
        query, key, value = \
            [l(x).view(nbatches, -1, self.heads_num, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]
        
        
        x, self.attn = attention(
            query, key, value, mask=mask,
            dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.heads_num * self.d_k)
        return x