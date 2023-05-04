---
title: Note - Multi-task learning as multi-objective optimization
date: 2023-05-04 22:26:53
tags: 
- optimization
- multi-task learning
---



reference link page: [summary of multi-task methods](https://c-harlin.github.io/%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/2022/07/30/%E5%A4%9A%E4%BB%BB%E5%8A%A1%E5%AD%A6%E4%B9%A0%E6%A6%82%E8%BF%B0.html)



## Problem

Multi-task learning is inherently a multi-objective problem. A classical weighted linear combination of per-task losses is only valid when the tasks do not compete. 



## Work

This paper cast multi-task learning as multi-objective optimization, with the overall objective of finding a Pareto optimal solution. 





codes: https://github.com/isl-org/MultiObjectiveOptimization/tree/master/multi_task
