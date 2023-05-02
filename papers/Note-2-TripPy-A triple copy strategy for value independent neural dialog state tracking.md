---
title: Note-2-TripPy:A triple copy strategy for value independent neural dialog state tracking collaborative selection for dialogue state tracking
date: 2023-05-03 22:26:53
tags: 
- Natural language processing
- dialogue state tracking
- copy mechanism
---

**Problem**: Multi-domain and open-vocabulary settings 

**Work**:  A slot is filled by (1) span prediction may extract values from the user input; (2) slot value can be copied from a system inform memory; (3) a value may be copied over from a different slot that is already contained in the dialog state to resolve correferences within and across domains.  In experiments on MultiWOZ2.1, the TripPy model achieve a joint goal accuracy beyond 55%.

**Experiment**

**Detail Information**: 

**Context encoder**: use BERT as front-end to encode ar each turn $t$ the dialog context as 
$$
R_t = \text{BERT}([CLS] \oplus U_t \oplus [SEP] \oplus M_t \oplus [SEP] \oplus H_t \oplus [SEP])
$$
where $U_t, M_t$ represents utterance of user and system. $H_t$ denotes the dialog context.

**Slot gates**: For ordinary class domains: $C = \{none, dontcare, span, inform, refer\}$ , the predication can be represented as $ p_{t,s}^{gate}(r_t^{CLS}) = \text{softmax}(W_s^{gate}\cdot r_t^{CLS} + b_s^{gate}) \in R^{5}$

Boolean slots, i.e., slots that only take binary values are treated separately. $C_{bool} = \{none, dontcare, true, false\}$ and the slot gate probability is $p_{t,s}^{bgate}(r_t^{CLS}) = \text{softmax}(W_s^{gate}\cdot r_t^{CLS} + b_s^{gate}) \in R^{4}$

**Span-based value prediction**: For each slot $s$ that is to be filled via span prediction, taking the token representations $[r_t^1,\cdots, r_t^{seq_{max}}]$ of entire dialog context for turn $t$ as input and projects them as follows:
$$
[\alpha_{t,i}^s, \beta_{t,i}^s] = W_s^{span}\cdot r_t^i + b_s^{span}\in R^2\\
start_{t}^s = \arg\max(\text{softmax}(\alpha_t^s))\\
end_t^s = \arg\max(\text{softmax}(\beta_t^s))
$$
**System inform memory for value prediction**: this module tracking of all slot values that were informed by the system in dialog turn $t$. 

**DS memory for coreference resolution**: The third copy mechanism utilizes the $DS$ as a memory to resolve corefeences. If a slot gate predicts that the user refers to a value that has already been assigned to a different slot during the conversation, then the probability distribution over all possible slots that can be referenced is 
$$
p_{t,s}^{refer}(r_t^{CLS}) = \text{softmax}(W_s^{refer}\cdot r_t^{CLS} + b_s^{refer}) \in R^{N+1}
$$
for each slot, a linear layer classification head either predicts the slot which contains the referenced value, or none for no reference. 

**Partial masking**: The author partially mask the dialog history $H_t$ by replacing values with BERT's generic [UNK] token.

**Training Loss**: 

The loss schema of training is as following (every item is cross entropy):
$$
L = 0.8\cdot L_{gate} + 0.1 \cdot L_{span} + 0.1 \cdot L_{refer}
$$
As there is no coreferencing in the evaluated single-domain datasets, the refer loss is not computed in those cases and the loss function is 
$$
L = 0.8 \cdot L_{gate} + 0.2 \cdot L_{span}
$$
Ablation experiments for model

| Model                             | JGA    |
| --------------------------------- | ------ |
| Span prediction only(entire turn) | 42.63% |
| + triple copy mechanism           | 49.23% |
| + dialogue history                | 52.58% |
| + auxiliary features              | 54.08% |
| + masking                         | 54.29% |
| TripPy(full sequency width)       | 55.29% |

