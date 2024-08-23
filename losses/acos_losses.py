import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from losses.SupCL import SupConLoss


def normalize_size(tensor):
    # 将输入的张量重新调整为适合计算的形状。 张量维度 3->2 2->1
    if len(tensor.size()) == 3:
        tensor = tensor.contiguous().view(-1, tensor.size(2))
    elif len(tensor.size()) == 2:
        tensor = tensor.contiguous().view(-1)

    return tensor


def calculate_entity_loss(pred_start, pred_end, gold_start, gold_end):
    # 计算实体识别任务的损失 模型预测的其实和结束位置的概率分布 真实的起始和结束位置标签
    pred_start = normalize_size(pred_start)
    pred_end = normalize_size(pred_end)
    gold_start = normalize_size(gold_start)
    gold_end = normalize_size(gold_end)

    # 类别权重，用于处理类别不平衡
    weight = torch.tensor([1, 3]).float().to('cpu')

    # 使用加权交叉熵损失计算预测和真实标签之间的差距
    loss_start = F.cross_entropy(pred_start, gold_start.long(), reduction='sum', weight=weight, ignore_index=-1)
    loss_end = F.cross_entropy(pred_end, gold_end.long(), reduction='sum', weight=weight, ignore_index=-1)

    # 将起始和结束位置的损失取平均
    return 0.5 * loss_start + 0.5 * loss_end


def calculate_category_loss(pred_category, gold_category):
    # 使用交叉熵损失函数计算预测类别和真实标签之间的差距
    return F.cross_entropy(pred_category, gold_category.long(), reduction='sum', ignore_index=-1)


def calculate_sentiment_loss(pred_sentiment, gold_sentiment):
    # 计算情感分析任务的损失
    return F.cross_entropy(pred_sentiment, gold_sentiment.long(), reduction='sum', ignore_index=-1)


def calculate_SCL_loss(gold, pred_scores):
    # 计算了监督对比学习的损失
    SCL = SupConLoss(contrast_mode='all', temperature=0.9)  # 0.9

    answer = gold
    idxs = torch.nonzero(answer != -1).squeeze()

    answers, score_list = [], []
    for i in idxs:
        answers.append(answer[i])
        score_list.append(pred_scores[i])

    # label 维度变为:[trans_dim]
    answers = torch.stack(answers)
    # 进行维度重构 category维度:[category_nums, 1, trans_dim]
    scores = torch.stack(score_list)

    scores = F.softmax(scores, dim=1)
    scores = scores.unsqueeze(1)

    scl_loss = SCL(scores, answers)
    scl_loss /= len(scores)

    return scl_loss


class FocalLoss(nn.Module):
    # 实现了焦点损失，用于处理类别不平衡的问题
    """
    Multi-class Focal loss implementation
    """

    def __init__(self, gamma=1, weight=None, ignore_index=-1):
        super(FocalLoss, self).__init__()
        # alpha
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, gold):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = F.log_softmax(pred, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = F.nll_loss(log_pt, gold, self.weight, ignore_index=self.ignore_index)
        return loss
