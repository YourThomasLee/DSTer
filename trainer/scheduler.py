#!/usr/bin python
#-*- encoding: utf8 -*-
import torch
from torch.optim.lr_scheduler import _LRScheduler
 
 
class NoamLR(_LRScheduler):
    """This scheduler is almost same as WarmupLR Scheduler except for following difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, model_size, factor, warmup_steps, last_epoch=-1):
        
        self.model_size = model_size
        self.factor = factor
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
 
    def get_lr(self):
        """we will use the first m batches, and set the learning rate to base_lr * m / total_iters
        """
        step = self._step_count
        if step == 0:
            step = 1
        adjust = lambda lr: lr * self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
        return [adjust(base_lr) for base_lr in self.base_lrs]