---
title: Note-1-Advancing Federated Learning with Granular Computing
date: 2023-05-23 22:26:53
tags: 
- granular computing
- federate learning

---

**Problem**:  Most time the numerical value does not provide semantic meaning in perspective of human being. It hurts the explain-ability and robustness of model. Granular computing can be an alternative way to cope with such a problem and it can be useful to elicit the concept generation on data.



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

The author call the first item as coverage and the second item as specificity.
