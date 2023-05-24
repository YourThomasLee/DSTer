---
title: Note-1-Advancing Federated Learning with Granular Computing
date: 2023-05-23 22:26:53
tags: 
- granular computing
- federate learning

---
## background
**Problem**:  Most time the numerical value does not provide semantic meaning in perspective of human being. It hurts the explain-ability and robustness of model. Granular computing can be an alternative way to cope with such a problem and it can be useful to elicit the concept generation on data.


## Information granule
**Information granule**: it was built with realizing a suitable level of abstraction which can be a good pragmatic problem-oriented trade-off among precision of results, easiness of interpretation, value and stability.

well-established formal frameworks of information granules:

- **Sets (intervals).** They realize a concept of abstraction by introducing a notion of dichotomy. Along with the set theory comes a well-developed discipline of interval analysis[9-11]
- **Fuzzy sets.** By admitting a notion of partial membership they deliver a conceptual and algorithmic generalization of set[12-14]
- **Shadowed sets.** They offer an interesting description of information granules by distinguishing among three categories of elements[15, 16] by discriminating among elements, which fully belong to the concept, which are excluded from it, and those elements whose belongingness is completely unknown.
- **Rough sets** are concerned with a roughness phenomenon, which arises when an object (pattern) is described in terms of a limited vocabulary of certain granularity. The description of this nature gives rise to
    so-called lower and upper bounds forming the essence of a rough set.

**The principle of justifiable granularity**: The underlying rationale behind the principle is to deliver a concise and abstract characterization of the data such that 

- the produced granule is justified in light of the available experimental data (representative - how much data supports the granule)
- the granule comes with a well-defined semantics meaning that it can be
    easily interpreted and becomes distinguishable from the others(good abstraction - in perspective of semantic meaning understood by human)

The first item was taken as coverage and the second item was taken as specificity. As an illustration, consider an interval form of information granule $A$. In case of intervals built on a basis of one-dimensional numeric data(evidence) $x_1, x_2, \cdots, x_N$, 
- the coverage measure is associated with a count of the number of data embraced by $A$, namely $cov(A) = card\{x_k|x_k \in A\}/N$. If the data are weighted by the corresponding weights $w_1, w_2, \cdots, w_N$, then the coverage is modified to be a weighted sum in the form $cov(A) = \sum_{x_i\in [a,b]}\frac{w_i}{\sum_{i=1}^N w_i}$
- the specificity of $A$, i.e., $sp(A)$ is regarded as a decreasing funciton $g$ of the size (length) of information granule. $sp(A) = g(length(A)) = 1-\frac{|b-a|}{range}=\exp(-\varepsilon|b - a|), \varepsilon > 0$

The criteria is the product $V = cov(A) sp(A)$. The design of information granule is accomplished by maximizing the above product of coverage and specificity. Thus, the desired solution (optimal values of $a$ and $b$) is the ones where the value of $V$ attains its maximum. 

## Credibility of ML Constructs
Two main chanllenges related to the construction an d an efficient deployment of ML architectures.
- Development of ML models by optimizing some loss function: different learning schemas at the minimization of the loss function, such as structural optimization and parameteric optimization
- Quantification of credibility of the model and its result: applications address the need to express how much confidence could be associated with the constructed ML model. A numeric result of prediction or classification does not carry any associated credibility measure.

From the architectural perspective, we can think of a granular embedding the original numeric ML model as illustrated in Fig. 2. The embedding mechanism is endowed with a level of information granularity $\epsilon$ which can be thought as a design asset. From the algorithmic perspective, the embedding is realized by optimizing a certain performance index characterizing the quality of granular results when being confronted
with the data.

![Fig. 2.](_resource/granule%20creditability.png)

Taking a numeric model $M$ expressed as $y = M(x; w)$ as an example, where $M$ is designed in a supervised mode on he basis pairs of input-output data $(x_k, target_k), k=1,2,...,N$.  