import torch
import numpy as np

def joint_accuracy(output, target):
    def comput_joint_acc(pred, label):
        #TODO: 计算真正的joint_acc
        #batch_size * state_label_num * state_domain
        batch_size, state_label_num = pred.shape[:2]
        pred = torch.argmax(pred, dim=-1)#batch_size * state_label
        label = torch.argmax(label, dim=-1)
        ret = (pred == label).sum(dim=-1)# batch_size
        return ret.sum().item() / (batch_size * state_label_num)
    
    device = output['hist_state'].device
    with torch.no_grad():
        # hist_acc = comput_joint_acc(output['hist_state'], target["previous_state"])
        ## 当前轮次更新的槽预测
        # state_change = torch.max(target['current_state'] - target['previous_state'], torch.zeros_like(target['previous_state'], device=device, requires_grad=False),) #计算增加的状态
        # update_state_true = torch.max(torch.zeros(state_change.shape, device=device, requires_grad=False), state_change)
        # update_state_pred = output['update_state']
        # update_acc = comput_joint_acc(update_state_pred, update_state_true)
        ## 当前轮次的状态预测
        cur_state_acc = comput_joint_acc(output['curr_state'], target["current_state"])
    return cur_state_acc#np.mean([hist_acc, cur_state_acc]) # np.mean([hist_acc, update_acc, cur_state_acc])
        

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
