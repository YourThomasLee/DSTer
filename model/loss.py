import torch
import torch.nn.functional as F


def cross_entropy(output, target):
    loss = torch.tensor([0.], device=target["previous_state"].device)
    #前一个状态的预测
    # hist_state_true = target["previous_state"]
    # hist_state_pred = output['hist_state']
    # loss += F.cross_entropy(hist_state_pred, hist_state_true)

    #当前轮次更新的槽预测
    # state_change = torch.max(target['current_state'] - target['previous_state'], 
    #     torch.zeros_like(target['previous_state'], device=target['previous_state'].device, requires_grad=False)) #计算增加的状态
    # update_state_true = torch.max(torch.zeros(state_change.shape, device=state_change.device, requires_grad=False), state_change)
    # update_state_pred = output['update_state']
    # loss += F.cross_entropy(update_state_pred, update_state_true)
    #当前轮次的状态预测
    curr_state_true = target["current_state"]
    curr_state_pred = output['curr_state']
    loss += F.cross_entropy(curr_state_pred, curr_state_true)
    return loss

def nll_loss(output, target):
    return F.nll_loss(output, target)

