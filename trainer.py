import json
import os
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm  # 显示进度条

from labels import get_aspect_category, get_sentiment
from losses.acos_losses import calculate_entity_loss, calculate_category_loss, calculate_sentiment_loss, \
    calculate_SCL_loss, FocalLoss
from metrics import ACOSScore, ACOXSScore
from question_template import get_English_Template, get_Chinese_Template
from tools import filter_unpaired, pair_combine, triplet_combine, FGM, PGD, batch_pair_combine


class ACOSTrainer:
    def __init__(self, logger, model, optimizer, scheduler, tokenizer, args):
        self.logger = logger                             # 日志
        self.model = model                               # 深度学习模型
        self.optimizer = optimizer                       # 优化器
        self.scheduler = scheduler                       # 学习率调度器，根据预定的计划调整优化器的学习率
        self.tokenizer = tokenizer                       # 分词器 分词并转为id
        self.args = args                                 # Terminal传入的参数(有部分模型的超参数)
        self.fgm = FGM(self.model)                       # 对抗训练方法1
        self.pgd = PGD(self.model)                       # 对抗训练方法2
        self.focalLoss = FocalLoss(self.args.flp_gamma)  # 损失函数，处理类别不平衡的问题
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 处理单元

    def train(self, train_dataloader, epoch):
        with tqdm(total=len(train_dataloader), desc="train") as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                # 清空梯度 每次反向传播之前，以免积累上一次计算的梯度，影响更新
                self.optimizer.zero_grad()

                # 查看当前数据形态
                # print(batch.forward_asp_query[0])  # 二维 每个example forward只有一个方面问句
                # print(batch.forward_opi_query[0])  # 三维 每个example forward会有多个opi问句
                # print(batch.forward_adv_query[0])  # 三维 每个example forward会有多个adv问句

                # 计算损失并反向传播
                loss_sum = self.get_train_loss(batch)
                loss_sum.backward()

                # 使用FGM对抗训练
                if self.args.use_FGM:
                    # 在embedding层上添加对抗扰动
                    self.fgm.attack()
                    FGM_loss_sum = self.get_train_loss(batch)

                    # 恢复embedding参数
                    FGM_loss_sum.backward()
                    self.fgm.restore()

                # 使用PGD对抗训练
                if self.args.use_PGD:
                    self.pgd.backup_grad()
                    for t in range(self.args.pgd_k):
                        # 在embedding上添加对抗扰动, first attack时备份param.data
                        self.pgd.attack(is_first_attack=(t == 0))
                        if t != self.args.pgd_k - 1:
                            self.model.zero_grad()
                        else:
                            self.pgd.restore_grad()

                        PGD_loss_sum = self.get_train_loss(batch)
                        # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                        PGD_loss_sum.backward()
                        # 恢复embedding参数
                    self.pgd.restore()

                # 梯度下降 更新参数
                self.optimizer.step()  # 更新模型的参数
                self.scheduler.step()  # 更新学习率
                self.model.zero_grad()  # 再次清空梯度

                pbar.set_description(f'Epoch [{epoch}/{self.args.epoch_num}]')
                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss_sum)})
                pbar.update(1)

    # def eval(self, eval_dataloader):
    #     json_res = []
    #     acos_score = ACOSScore(self.logger)
    #     self.model.eval()
    #
    #     Forward_Q1, Backward_Q1, Forward_Q2, Backward_Q2, Q3, Q4 = get_English_Template()
    #     f_asp_imp_start = 5
    #     b_opi_imp_start = 5
    #     for batch in tqdm(eval_dataloader):
    #         asp_predict, opi_predict, asp_opi_predict, triplets_predict, aocs_predict, quadruples_predict = [], [], [], [], [], []
    #
    #         forward_pair_list, forward_pair_prob, forward_pair_ind_list = [], [], []
    #
    #         backward_pair_list, backward_pair_prob, backward_pair_ind_list = [], [], []
    #
    #         # forward q_1 nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
    #         passenge_index = batch.forward_asp_answer_start[0].gt(-1).float().nonzero()
    #         passenge = batch.forward_asp_query[0][passenge_index].squeeze(1)
    #
    #         f_asp_start_scores, f_asp_end_scores = self.model(batch.forward_asp_query.to('cpu'),
    #                                                           batch.forward_asp_query_mask.to('cpu'),
    #                                                           batch.forward_asp_query_seg.to('cpu'), 0)
    #         f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
    #         f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
    #         f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
    #         f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)
    #
    #         f_asp_start_prob_temp = []
    #         f_asp_end_prob_temp = []
    #         f_asp_start_index_temp = []
    #         f_asp_end_index_temp = []
    #         for i in range(f_asp_start_ind.size(0)):
    #             if batch.forward_asp_answer_start[0, i] != -1:
    #                 if f_asp_start_ind[i].item() == 1:
    #                     f_asp_start_index_temp.append(i)
    #                     f_asp_start_prob_temp.append(f_asp_start_prob[i].item())
    #                 if f_asp_end_ind[i].item() == 1:
    #                     f_asp_end_index_temp.append(i)
    #                     f_asp_end_prob_temp.append(f_asp_end_prob[i].item())
    #
    #         f_asp_start_index, f_asp_end_index, f_asp_prob = filter_unpaired(
    #             f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp,
    #             f_asp_imp_start)
    #
    #         for i in range(len(f_asp_start_index)):
    #             if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                 opinion_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[:7]])
    #             else:
    #                 opinion_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[:6]])
    #             for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1):
    #                 opinion_query.append(batch.forward_asp_query[0][j].item())
    #             if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                 opinion_query.extend(self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[7:]]))
    #             else:
    #                 opinion_query.extend(self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[6:]]))
    #             imp_start = len(opinion_query)
    #
    #             opinion_query_seg = [0] * len(opinion_query)
    #             f_opi_length = len(opinion_query)
    #
    #             opinion_query = torch.tensor(opinion_query).long()
    #             opinion_query = torch.cat([opinion_query, passenge], -1).to('cpu').unsqueeze(0)
    #             opinion_query_seg += [1] * passenge.size(0)
    #             opinion_query_mask = torch.ones(opinion_query.size(1)).float().to('cpu').unsqueeze(0)
    #             opinion_query_seg = torch.tensor(opinion_query_seg).long().to('cpu').unsqueeze(0)
    #
    #             f_opi_start_scores, f_opi_end_scores = self.model(opinion_query, opinion_query_mask, opinion_query_seg,
    #                                                               0)
    #
    #             f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
    #             f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
    #             f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
    #             f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)
    #
    #             f_opi_start_prob_temp = []
    #             f_opi_end_prob_temp = []
    #             f_opi_start_index_temp = []
    #             f_opi_end_index_temp = []
    #             for k in range(f_opi_start_ind.size(0)):
    #                 if opinion_query_seg[0, k] == 1:
    #                     if f_opi_start_ind[k].item() == 1:
    #                         f_opi_start_index_temp.append(k)
    #                         f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
    #                     if f_opi_end_ind[k].item() == 1:
    #                         f_opi_end_index_temp.append(k)
    #                         f_opi_end_prob_temp.append(f_opi_end_prob[k].item())
    #
    #             f_opi_start_index, f_opi_end_index, f_opi_prob = filter_unpaired(
    #                 f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp, imp_start)
    #
    #             for idx in range(len(f_opi_start_index)):
    #                 asp = [batch.forward_asp_query[0][j].item() for j in
    #                        range(f_asp_start_index[i], f_asp_end_index[i] + 1)]
    #                 opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
    #                 # null -> -1, -1
    #                 if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                     asp_ind = [f_asp_start_index[i] - 8, f_asp_end_index[i] - 8]
    #                 else:
    #                     asp_ind = [f_asp_start_index[i] - 6, f_asp_end_index[i] - 6]
    #                 opi_ind = [f_opi_start_index[idx] - f_opi_length - 1, f_opi_end_index[idx] - f_opi_length - 1]
    #                 temp_prob = f_asp_prob[i] * f_opi_prob[idx]
    #                 if asp_ind + opi_ind not in forward_pair_list:
    #                     forward_pair_list.append([asp] + [opi])
    #                     forward_pair_prob.append(temp_prob)
    #                     forward_pair_ind_list.append(asp_ind + opi_ind)
    #
    #         # backward q_1
    #         b_opi_start_scores, b_opi_end_scores = self.model(batch.backward_opi_query.to('cpu'),
    #                                                           batch.backward_opi_query_mask.to('cpu'),
    #                                                           batch.backward_opi_query_seg.to('cpu'), 0)
    #         b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
    #         b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
    #         b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
    #         b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)
    #
    #         b_opi_start_prob_temp = []
    #         b_opi_end_prob_temp = []
    #         b_opi_start_index_temp = []
    #         b_opi_end_index_temp = []
    #         for i in range(b_opi_start_ind.size(0)):
    #             if batch.backward_opi_answer_start[0, i] != -1:
    #                 if b_opi_start_ind[i].item() == 1:
    #                     b_opi_start_index_temp.append(i)
    #                     b_opi_start_prob_temp.append(b_opi_start_prob[i].item())
    #                 if b_opi_end_ind[i].item() == 1:
    #                     b_opi_end_index_temp.append(i)
    #                     b_opi_end_prob_temp.append(b_opi_end_prob[i].item())
    #
    #         b_opi_start_index, b_opi_end_index, b_opi_prob = filter_unpaired(
    #             b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp,
    #             b_opi_imp_start)
    #
    #         # backward q_2
    #         for i in range(len(b_opi_start_index)):
    #             if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                 aspect_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[:7]])
    #             else:
    #                 aspect_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[:6]])
    #             for j in range(b_opi_start_index[i], b_opi_end_index[i] + 1):
    #                 aspect_query.append(batch.backward_opi_query[0][j].item())
    #
    #             if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                 aspect_query.extend(self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[7:]]))
    #             else:
    #                 aspect_query.extend(self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[6:]]))
    #             imp_start = len(aspect_query)
    #
    #             aspect_query_seg = [0] * len(aspect_query)
    #             b_asp_length = len(aspect_query)
    #             aspect_query = torch.tensor(aspect_query).long()
    #             aspect_query = torch.cat([aspect_query, passenge], -1).to('cpu').unsqueeze(0)
    #             aspect_query_seg += [1] * passenge.size(0)
    #             aspect_query_mask = torch.ones(aspect_query.size(1)).float().to('cpu').unsqueeze(0)
    #             aspect_query_seg = torch.tensor(aspect_query_seg).long().to('cpu').unsqueeze(0)
    #
    #             b_asp_start_scores, b_asp_end_scores = self.model(aspect_query, aspect_query_mask, aspect_query_seg, 0)
    #
    #             b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
    #             b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
    #             b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
    #             b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)
    #
    #             b_asp_start_prob_temp = []
    #             b_asp_end_prob_temp = []
    #             b_asp_start_index_temp = []
    #             b_asp_end_index_temp = []
    #             for k in range(b_asp_start_ind.size(0)):
    #                 if aspect_query_seg[0, k] == 1:
    #                     if b_asp_start_ind[k].item() == 1:
    #                         b_asp_start_index_temp.append(k)
    #                         b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
    #                     if b_asp_end_ind[k].item() == 1:
    #                         b_asp_end_index_temp.append(k)
    #                         b_asp_end_prob_temp.append(b_asp_end_prob[k].item())
    #
    #             b_asp_start_index, b_asp_end_index, b_asp_prob = filter_unpaired(
    #                 b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp, imp_start)
    #
    #             for idx in range(len(b_asp_start_index)):
    #                 opi = [batch.backward_opi_query[0][j].item() for j in
    #                        range(b_opi_start_index[i], b_opi_end_index[i] + 1)]
    #                 asp = [aspect_query[0][j].item() for j in
    #                        range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
    #                 # null -> -1, -1
    #                 asp_ind = [b_asp_start_index[idx] - b_asp_length - 1, b_asp_end_index[idx] - b_asp_length - 1]
    #                 if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                     opi_ind = [b_opi_start_index[i] - 8, b_opi_end_index[i] - 8]
    #                 else:
    #                     opi_ind = [b_opi_start_index[i] - 6, b_opi_end_index[i] - 6]
    #                 temp_prob = b_asp_prob[idx] * b_opi_prob[i]
    #                 if asp_ind + opi_ind not in backward_pair_ind_list:
    #                     backward_pair_list.append([asp] + [opi])
    #                     backward_pair_prob.append(temp_prob)
    #                     backward_pair_ind_list.append(asp_ind + opi_ind)
    #
    #         if self.args.use_Forward:
    #             final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
    #             for idx in range(len(forward_pair_list)):
    #                 if forward_pair_list[idx][0] not in final_asp_list:
    #                     final_asp_list.append(forward_pair_list[idx][0])
    #                     final_opi_list.append([forward_pair_list[idx][1]])
    #                     final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
    #                     final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
    #                 else:
    #                     asp_index = final_asp_list.index(forward_pair_list[idx][0])
    #                     if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
    #                         final_opi_list[asp_index].append(forward_pair_list[idx][1])
    #                         final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
    #         elif self.args.use_Backward:
    #             final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
    #             for idx in range(len(backward_pair_list)):
    #                 if backward_pair_list[idx][0] not in final_asp_list:
    #                     final_asp_list.append(backward_pair_list[idx][0])
    #                     final_opi_list.append([backward_pair_list[idx][1]])
    #                     final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
    #                     final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
    #                 else:
    #                     asp_index = final_asp_list.index(backward_pair_list[idx][0])
    #                     if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
    #                         final_opi_list[asp_index].append(backward_pair_list[idx][1])
    #                         final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])
    #         else:
    #             # combine forward and backward pairs
    #             final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = pair_combine(forward_pair_list,
    #                                                                                                   forward_pair_prob,
    #                                                                                                   forward_pair_ind_list,
    #                                                                                                   backward_pair_list,
    #                                                                                                   backward_pair_prob,
    #                                                                                                   backward_pair_ind_list,
    #                                                                                                   self.args.alpha,
    #                                                                                                   self.args.beta)
    #
    #         # category sentiment
    #         for idx in range(len(final_asp_list)):
    #             predict_opinion_num = len(final_opi_list[idx])
    #             # category sentiment
    #             if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                 category_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[:7]])
    #                 sentiment_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:7]])
    #             else:
    #                 category_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[:6]])
    #                 sentiment_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:6]])
    #             category_query += final_asp_list[idx]
    #             sentiment_query += final_asp_list[idx]
    #             if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                 [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:9]]
    #                 category_query += self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[7:8]])
    #                 sentiment_query += self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[7:8]])
    #             else:
    #                 category_query += self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[6:9]])
    #                 sentiment_query += self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:9]])
    #
    #             # 拼接opinion
    #             for idy in range(predict_opinion_num):
    #                 category_query2 = category_query + final_opi_list[idx][idy]
    #                 sentiment_query2 = sentiment_query + final_opi_list[idx][idy]
    #                 if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                     category_query2.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[8:]]))
    #                     sentiment_query2.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[8:]]))
    #                 else:
    #                     category_query2.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[9:]]))
    #                     sentiment_query2.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[9:]]))
    #
    #                 category_query_seg = [0] * len(category_query2)
    #                 category_query2 = torch.tensor(category_query2).long().to('cpu')
    #                 category_query2 = torch.cat([category_query2, passenge.to('cpu')], -1).unsqueeze(0)
    #                 category_query_seg += [1] * passenge.size(0)
    #                 category_query_mask = torch.ones(category_query2.size(1)).float().to('cpu').unsqueeze(0)
    #                 category_query_seg = torch.tensor(category_query_seg).long().to('cpu').unsqueeze(0)
    #
    #                 sentiment_query_seg = [0] * len(sentiment_query2)
    #                 sentiment_query2 = torch.tensor(sentiment_query2).long().to('cpu')
    #                 sentiment_query2 = torch.cat([sentiment_query2, passenge.to('cpu')], -1).unsqueeze(0)
    #                 sentiment_query_seg += [1] * passenge.size(0)
    #                 sentiment_query_mask = torch.ones(sentiment_query2.size(1)).float().to('cpu').unsqueeze(0)
    #                 sentiment_query_seg = torch.tensor(sentiment_query_seg).long().to('cpu').unsqueeze(0)
    #
    #                 category_scores = self.model(category_query2, category_query_mask, category_query_seg, 1)
    #                 category_scores = F.softmax(category_scores, dim=1)
    #                 category_predicted = torch.argmax(category_scores[0], dim=0).item()
    #
    #                 sentiment_scores = self.model(sentiment_query2, sentiment_query_mask, sentiment_query_seg, 2)
    #                 sentiment_scores = F.softmax(sentiment_scores, dim=1)
    #                 sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()
    #
    #                 # 三元组、四元组组合
    #                 asp_f, opi_f = [], []
    #                 asp_f.append(final_asp_ind_list[idx][0])
    #                 asp_f.append(final_asp_ind_list[idx][1])
    #                 opi_f.append(final_opi_ind_list[idx][idy][0])
    #                 opi_f.append(final_opi_ind_list[idx][idy][1])
    #                 triplet_predict = [asp_f, opi_f, sentiment_predicted]
    #                 aoc_predict = [asp_f, opi_f, category_predicted]
    #                 quadruple_predict = [asp_f, category_predicted, opi_f, sentiment_predicted]
    #
    #                 if asp_f not in asp_predict:
    #                     asp_predict.append(asp_f)
    #                 if opi_f not in opi_predict:
    #                     opi_predict.append(opi_f)
    #                 if [asp_f, opi_f] not in asp_opi_predict:
    #                     asp_opi_predict.append([asp_f, opi_f])
    #                 if triplet_predict not in triplets_predict:
    #                     triplets_predict.append(triplet_predict)
    #                 if aoc_predict not in aocs_predict:
    #                     aocs_predict.append(aoc_predict)
    #                 if quadruple_predict not in quadruples_predict:
    #                     quadruples_predict.append(quadruple_predict)
    #
    #         acos_score.update(batch.aspects[0], batch.opinions[0], batch.pairs[0], batch.aste_triplets[0],
    #                           batch.aoc_triplets[0], batch.quadruples[0],
    #                           asp_predict, opi_predict, asp_opi_predict, triplets_predict, aocs_predict,
    #                           quadruples_predict)
    #
    #         # sentences_list.append(' '.join(batch.sentence_token[0]))
    #         # pred_quads.append(quadruples_predict)
    #         # gold_quads.append(batch.quadruples[0])
    #         one_json = {'sentence': ' '.join(batch.sentence_token[0]), 'pred': str(quadruples_predict),
    #                     'gold': str(batch.quadruples[0])}
    #         json_res.append(one_json)
    #     with open(os.path.join(self.args.output_dir, self.args.task, self.args.data_type, 'pred.json'), 'w', encoding='utf-8') as fP:
    #         json.dump(json_res, fP, ensure_ascii=False, indent=4)
    #     return acos_score.compute()

    def my_eval(self, eval_dataloader):
        self.model.eval()
        json_res = []
        acoxs_score = ACOXSScore(self.logger)

        q1, q2, q3, q4, q5, q6, q7, q8 = get_Chinese_Template()
        for batch in tqdm(eval_dataloader):
            # 查看当前数据形态   dev的batch_size = 1 所以形式上只是多了一个维度而已
            # batch(size = 1)/dataloader中的所有模型输入系列，除了第一个问句不需要推测信息之外，其他都是直接参照答案获取的
            # 所以这里eval中不可以直接套用大部分的dataloader，只能使用batch.forward_asp_query 所以整体风格偏向inference

            asp_predict, opi_predict, adv_predict, asp_opi_predict, \
                aste_triplets_predict, aoc_triplets_predict, aoa_triplets_predict, \
                quintuples_predict = [], [], [], [], [], [], [], []

            forward_pair_list, forward_pair_prob, forward_pair_idx_list = [], [], []
            forward_triplet_list, forward_triplet_prob, forward_triplet_idx_list = [], [], []
            backward_pair_list, backward_pair_prob, backward_pair_idx_list = [], [], []
            backward_triplet_list, backward_triplet_prob, backward_triplet_idx_list = [], [], []

            # ====================Forward Q1: Aspect=========================
            # forward q_1 nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
            # 返回 null + review 在temp_text中的索引
            passenge_index = batch.forward_asp_answer_start[0].gt(-1).float().nonzero()
            passenge = batch.forward_asp_query[0][passenge_index].squeeze(1)  # 去除一个维度 中间的

            f_asp_start_scores, f_asp_end_scores = self.model(batch.forward_asp_query.to(self.device),
                                                              batch.forward_asp_query_mask.to(self.device),
                                                              batch.forward_asp_query_seg.to(self.device), 0)
            f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
            f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
            f_asp_start_prob, f_asp_start_idx = torch.max(f_asp_start_scores, dim=1)
            f_asp_end_prob, f_asp_end_idx = torch.max(f_asp_end_scores, dim=1)

            f_asp_start_prob_temp = []
            f_asp_end_prob_temp = []
            f_asp_start_index_temp = []
            f_asp_end_index_temp = []
            for i in range(f_asp_start_idx.size(0)):
                if batch.forward_asp_answer_start[0, i] != -1:  # 忽略答案1  只要不是-1就行，效果就和idx_list一样
                    if f_asp_start_idx[i].item() == 1:
                        f_asp_start_index_temp.append(i)
                        f_asp_start_prob_temp.append(f_asp_start_prob[i].item())
                    if f_asp_end_idx[i].item() == 1:
                        f_asp_end_index_temp.append(i)
                        f_asp_end_prob_temp.append(f_asp_end_prob[i].item())

            f_asp_start_idx, f_asp_end_idx, f_asp_prob = filter_unpaired(
                f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp,
                8
                )
            # =================Forward Q2: Aspect->Opinion================
            for a in range(len(f_asp_start_idx)):
                f_opi_query = self.tokenizer.convert_tokens_to_ids(q2)
                for j in range(f_asp_start_idx[a], f_asp_end_idx[a] + 1):
                    f_opi_query.insert(5, batch.forward_asp_query[0][j].item())

                f_opi_query_length = len(f_opi_query)
                f_opi_query_seg = [0] * len(f_opi_query)
                imp_start = len(f_opi_query)
                f_opi_query = torch.tensor(f_opi_query).long()
                f_opi_query = torch.cat([f_opi_query, passenge], -1).to(self.device).unsqueeze(0)

                f_opi_query_mask = torch.ones(f_opi_query.size(1)).float().to(self.device).unsqueeze(0)
                f_opi_query_seg += [1] * passenge.size(0)
                f_opi_query_seg = torch.tensor(f_opi_query_seg).long().to(self.device).unsqueeze(0)

                f_opi_start_scores, f_opi_end_scores = self.model(f_opi_query, f_opi_query_mask, f_opi_query_seg, 0)

                f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
                f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
                f_opi_start_prob, f_opi_start_idx = torch.max(f_opi_start_scores, dim=1)
                f_opi_end_prob, f_opi_end_idx = torch.max(f_opi_end_scores, dim=1)

                f_opi_start_prob_temp, f_opi_end_prob_temp = [], []
                f_opi_start_idx_temp, f_opi_end_idx_temp = [], []

                for k in range(f_opi_start_idx.size(0)):
                    if f_opi_query_seg[0, k] == 1:  # 把判定到query部分的id全部过滤掉
                        if f_opi_start_idx[k].item() == 1:
                            f_opi_start_idx_temp.append(k)
                            f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                        if f_opi_end_idx[k].item() == 1:
                            f_opi_end_idx_temp.append(k)
                            f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

                f_opi_start_idx, f_opi_end_idx, f_opi_prob = filter_unpaired(
                    f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_idx_temp, f_opi_end_idx_temp,
                    imp_start=imp_start
                )
                # 组合 形成 a-o pair
                for k in range(len(f_opi_start_idx)):
                    asp = [batch.forward_asp_query[0][j].item() for j in range(f_asp_start_idx[a], f_asp_end_idx[a] + 1)]
                    opi = [f_opi_query[0][j].item() for j in range(f_opi_start_idx[k], f_opi_end_idx[k] + 1)]

                    asp_idx = [f_asp_start_idx[a] - len(q1) - 2, f_asp_end_idx[a] - len(q1) - 2]
                    opi_idx = [f_opi_start_idx[k] - f_opi_query_length - 1, f_opi_end_idx[k] - f_opi_query_length - 1]

                    # 计算配对概率
                    temp_prob = f_asp_prob[a] * f_opi_prob[k]

                    if asp_idx + opi_idx not in forward_pair_idx_list:
                        forward_pair_list.append([asp] + [opi])
                        forward_pair_prob.append(temp_prob)
                        forward_pair_idx_list.append(asp_idx + opi_idx)

                # ==================Q3: Aspect + Opinion -> Adverb ======================
                for b in range(len(f_opi_start_idx)):
                    f_adv_query = self.tokenizer.convert_tokens_to_ids(q3)
                    # insert aspect
                    for j in range(f_asp_start_idx[a], f_asp_end_idx[a] + 1):
                        f_adv_query.insert(5, batch.forward_asp_query[0][j].item())
                    # insert opinion
                    for j in range(f_opi_start_idx[b], f_opi_end_idx[b] + 1):
                        f_adv_query.insert(-8, f_opi_query[0][j].item())

                    f_adv_query_length = len(f_adv_query)

                    f_adv_query_seg = [0] * len(f_adv_query)
                    imp_start = len(f_adv_query)
                    f_adv_query = torch.tensor(f_adv_query).long()
                    f_adv_query = torch.cat([f_adv_query, passenge], -1).to(self.device).unsqueeze(0)

                    f_adv_query_mask = torch.ones(f_adv_query.size(1)).float().to(self.device).unsqueeze(0)
                    f_adv_query_seg += [1] * passenge.size(0)
                    f_adv_query_seg = torch.tensor(f_adv_query_seg).long().to(self.device).unsqueeze(0)

                    f_adv_start_scores, f_adv_end_scores = self.model(f_adv_query, f_adv_query_mask,
                                                                      f_adv_query_seg, 0)

                    f_adv_start_scores = F.softmax(f_adv_start_scores[0], dim=1)
                    f_adv_end_scores = F.softmax(f_adv_end_scores[0], dim=1)
                    f_adv_start_prob, f_adv_start_idx = torch.max(f_adv_start_scores, dim=1)
                    f_adv_end_prob, f_adv_end_idx = torch.max(f_adv_end_scores, dim=1)

                    f_adv_start_prob_temp, f_adv_end_prob_temp = [], []
                    f_adv_start_idx_temp, f_adv_end_idx_temp = [], []

                    for k in range(f_adv_start_idx.size(0)):
                        if f_adv_query_seg[0, k] == 1:
                            if f_adv_start_idx[k].item() == 1:
                                f_adv_start_idx_temp.append(k)
                                f_adv_start_prob_temp.append(f_adv_start_prob[k].item())
                            if f_adv_end_idx[k].item() == 1:
                                f_adv_end_idx_temp.append(k)
                                f_adv_end_prob_temp.append(f_adv_end_prob[k].item())

                    f_adv_start_idx, f_adv_end_idx, f_adv_prob = filter_unpaired(
                        f_adv_start_prob_temp, f_adv_end_prob_temp, f_adv_start_idx_temp, f_adv_end_idx_temp,
                        imp_start=imp_start
                    )

                    for k in range(len(f_adv_start_idx)):
                        asp = [batch.forward_asp_query[0][j].item() for j in
                               range(f_asp_start_idx[a], f_asp_end_idx[a] + 1)]
                        opi = [f_opi_query[0][j].item() for j in
                               range(f_opi_start_idx[b], f_opi_end_idx[b] + 1)]
                        adv = [f_adv_query[0][j].item() for j in
                               range(f_adv_start_idx[k], f_adv_end_idx[k] + 1)]

                        # 问题 + null     没有null(0,0) -> (-1, -1)   回归最原始评论中关键元素的下标
                        asp_idx = [f_asp_start_idx[a] - len(q1) - 2, f_asp_end_idx[a] - len(q1) - 2]
                        opi_idx = [f_opi_start_idx[b] - f_opi_query_length - 1,
                                   f_opi_end_idx[b] - f_opi_query_length - 1]
                        adv_idx = [f_adv_start_idx[k] - f_adv_query_length - 1,
                                   f_adv_end_idx[k] - f_adv_query_length - 1]

                        # 计算配对概率
                        temp_prob = f_asp_prob[a] * f_opi_prob[b] * f_adv_prob[k]

                        if asp_idx + opi_idx + adv_idx not in forward_triplet_idx_list:
                            forward_triplet_list.append([asp] + [opi] + [adv])
                            forward_triplet_prob.append(temp_prob)
                            forward_triplet_idx_list.append(asp_idx + opi_idx + adv_idx)

            # =================Backward Q4: Adverb -> Opinion========================
            b_adv_start_scores, b_adv_end_scores = self.model(batch.backward_adv_query.to(self.device),
                                                              batch.backward_adv_query_mask.to(self.device),
                                                              batch.backward_adv_query_seg.to(self.device), 0)
            b_adv_start_scores = F.softmax(b_adv_start_scores[0], dim=1)
            b_adv_end_scores = F.softmax(b_adv_end_scores[0], dim=1)
            b_adv_start_prob, b_adv_start_idx = torch.max(b_adv_start_scores, dim=1)
            b_adv_end_prob, b_adv_end_idx = torch.max(b_adv_end_scores, dim=1)

            b_adv_start_prob_temp = []
            b_adv_end_prob_temp = []
            b_adv_start_index_temp = []
            b_adv_end_index_temp = []
            for i in range(b_adv_start_idx.size(0)):
                if batch.backward_adv_answer_start[0, i] != -1:
                    if b_adv_start_idx[i].item() == 1:
                        b_adv_start_index_temp.append(i)
                        b_adv_start_prob_temp.append(b_adv_start_prob[i].item())
                    if b_adv_end_idx[i].item() == 1:
                        b_adv_end_index_temp.append(i)
                        b_adv_end_prob_temp.append(b_adv_end_prob[i].item())

            b_adv_start_idx, b_adv_end_idx, b_adv_prob = filter_unpaired(
                b_adv_start_prob_temp, b_adv_end_prob_temp, b_adv_start_index_temp, b_adv_end_index_temp,
                8)

            # ==================Backward Q5: Adverb -> Opinion =======================
            for a in range(len(b_adv_start_idx)):
                b_opi_query = self.tokenizer.convert_tokens_to_ids(q5)
                for j in range(b_adv_start_idx[a], b_adv_end_idx[a] + 1):
                    b_opi_query.insert(5, batch.backward_adv_query[0][j].item())

                b_opi_query_length = len(b_opi_query)

                b_opi_query_seg = [0] * len(b_opi_query)
                imp_start = len(b_opi_query)
                b_opi_query = torch.tensor(b_opi_query).long()
                b_opi_query = torch.cat([b_opi_query, passenge], -1).to(self.device).unsqueeze(0)

                b_opi_query_mask = torch.ones(b_opi_query.size(1)).float().to(self.device).unsqueeze(0)
                b_opi_query_seg += [1] * passenge.size(0)
                b_opi_query_seg = torch.tensor(b_opi_query_seg).long().to(self.device).unsqueeze(0)

                b_opi_start_scores, b_opi_end_scores = self.model(b_opi_query, b_opi_query_mask,
                                                                  b_opi_query_seg, 0)

                b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
                b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
                b_opi_start_prob, b_opi_start_idx = torch.max(b_opi_start_scores, dim=1)
                b_opi_end_prob, b_opi_end_idx = torch.max(b_opi_end_scores, dim=1)

                b_opi_start_prob_temp, b_opi_end_prob_temp = [], []
                b_opi_start_idx_temp, b_opi_end_idx_temp = [], []

                for k in range(b_opi_start_idx.size(0)):
                    if b_opi_query_seg[0, k] == 1:
                        if b_opi_start_idx[k].item() == 1:
                            b_opi_start_idx_temp.append(k)
                            b_opi_start_prob_temp.append(b_opi_start_prob[k].item())
                        if b_opi_end_idx[k].item() == 1:
                            b_opi_end_idx_temp.append(k)
                            b_opi_end_prob_temp.append(b_opi_end_prob[k].item())

                b_opi_start_idx, b_opi_end_idx, b_opi_prob = filter_unpaired(
                    b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_idx_temp, b_opi_end_idx_temp,
                    imp_start=imp_start
                )

                for k in range(len(b_opi_start_idx)):
                    adv = [batch.backward_adv_query[0][j].item() for j in range(b_adv_start_idx[a], b_adv_end_idx[a] + 1)]
                    opi = [b_opi_query[0][j].item() for j in range(b_opi_start_idx[k], b_opi_end_idx[k] + 1)]

                    adv_idx = [b_adv_start_idx[a] - len(q4) - 2, b_adv_end_idx[a] - len(q4) - 2]
                    opi_idx = [b_opi_start_idx[k] - b_opi_query_length - 1,
                               b_opi_end_idx[k] - b_opi_query_length - 1]

                    temp_prob = b_opi_prob[k] * b_adv_prob[a]

                    if opi_idx + adv_idx not in backward_pair_idx_list:
                        backward_pair_list.append([opi] + [adv])
                        backward_pair_prob.append(temp_prob)
                        backward_pair_idx_list.append(opi_idx + adv_idx)

                # =============================Backward Q6========================================
                for b in range(len(b_opi_start_idx)):
                    b_asp_query = self.tokenizer.convert_tokens_to_ids(q6)
                    # insert adverb
                    for j in range(b_adv_start_idx[a], b_adv_end_idx[a] + 1):
                        b_asp_query.insert(5, batch.backward_adv_query[0][j].item())
                    # insert opinion
                    for j in range(b_opi_start_idx[b], b_opi_end_idx[b] + 1):
                        b_asp_query.insert(-10, b_opi_query[0][j].item())

                    b_asp_query_length = len(b_asp_query)

                    b_asp_query_seg = [0] * len(b_asp_query)
                    imp_start = len(b_asp_query)
                    b_asp_query = torch.tensor(b_asp_query)
                    b_asp_query = torch.cat([b_asp_query, passenge], -1).to(self.device).unsqueeze(0)

                    b_asp_query_mask = torch.ones(b_asp_query.size(1)).float().to(self.device).unsqueeze(0)
                    b_asp_query_seg += [1] * passenge.size(0)
                    b_asp_query_seg = torch.tensor(b_asp_query_seg).long().to(self.device).unsqueeze(0)

                    b_asp_start_scores, b_asp_end_scores = self.model(b_asp_query, b_asp_query_mask,
                                                                      b_asp_query_seg, 0)

                    b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
                    b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
                    b_asp_start_prob, b_asp_start_idx = torch.max(b_asp_start_scores, dim=1)
                    b_asp_end_prob, b_asp_end_idx = torch.max(b_asp_end_scores, dim=1)

                    b_asp_start_prob_temp, b_asp_end_prob_temp = [], []
                    b_asp_start_idx_temp, b_asp_end_idx_temp = [], []

                    for k in range(b_asp_start_idx.size(0)):
                        if b_asp_query_seg[0, k] == 1:
                            if b_asp_start_idx[k].item() == 1:
                                b_asp_start_idx_temp.append(k)
                                b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                            if b_asp_end_idx[k].item() == 1:
                                b_asp_end_idx_temp.append(k)
                                b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

                    b_asp_start_idx, b_asp_end_idx, b_asp_prob = filter_unpaired(
                        b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_idx_temp, b_asp_end_idx_temp,
                        imp_start=imp_start
                    )

                    for k in range(len(b_asp_start_idx)):
                        adv = [batch.backward_adv_query[0][j].item() for j in
                               range(b_adv_start_idx[a], b_adv_end_idx[a] + 1)]
                        opi = [b_opi_query[0][j].item() for j in
                               range(b_opi_start_idx[b], b_opi_end_idx[b] + 1)]
                        asp = [b_asp_query[0][j].item() for j in
                               range(b_asp_start_idx[k], b_asp_end_idx[k] + 1)]

                        adv_idx = [b_adv_start_idx[a] - len(q4) - 2, b_adv_end_idx[a] - len(q4) - 2]
                        opi_idx = [b_opi_start_idx[b] - b_opi_query_length - 1,
                                   b_opi_end_idx[b] - b_opi_query_length - 1]
                        asp_idx = [b_asp_start_idx[k] - b_asp_query_length - 1,
                                   b_asp_end_idx[k] - b_asp_query_length - 1]

                        temp_prob = b_asp_prob[k] * b_opi_prob[b] * b_adv_prob[a]
                        if asp_idx + opi_idx + adv_idx not in backward_triplet_idx_list:
                            backward_triplet_list.append([asp] + [opi] + [adv])
                            backward_triplet_prob.append(temp_prob)
                            backward_triplet_idx_list.append(asp_idx + opi_idx + adv_idx)

            # ===================Q7: Category + Q8: Sentiment ======================
            if self.args.use_Forward:
                pass
            elif self.args.use_Backward:
                pass
            else:
                final_asp_list, final_opi_list, final_adv_list, \
                     final_asp_idx_list, final_opi_idx_list, final_adv_idx_list = triplet_combine(
                        forward_triplet_list,
                        forward_triplet_prob,
                        forward_triplet_idx_list,
                        backward_triplet_list,
                        backward_triplet_prob,
                        backward_triplet_idx_list,
                        self.args.alpha,
                        self.args.beta
                        )
                for a in range(len(final_asp_list)):
                    predict_opinion_num = len(final_opi_list[a])  # asp 对应的 opi数
                    category_query = self.tokenizer.convert_tokens_to_ids(q7)
                    sentiment_query = self.tokenizer.convert_tokens_to_ids(q8)
                    # insert aspect in query
                    for j in range(len(final_asp_list[a])):
                        category_query.insert(5, final_asp_list[a][j])
                        sentiment_query.insert(5, final_asp_list[a][j])
                    temp_category = category_query
                    temp_sentiment = sentiment_query
                    for b in range(predict_opinion_num):
                        predict_adverb_num = len(final_adv_list[a][b])
                        # 循环状态回溯
                        category_query = temp_category
                        sentiment_query = temp_sentiment
                        # insert opinion in query
                        for j in range(len(final_opi_list[a][b])):
                            category_query.insert(-8, final_opi_list[a][b][j])
                            sentiment_query.insert(-8, final_opi_list[a][b][j])

                        # category
                        category_query_seg = [0] * len(category_query)
                        category_query = torch.tensor(category_query).long().to(self.device)
                        category_query = torch.cat([category_query, passenge], -1).to(self.device).unsqueeze(0)
                        category_query_seg += [1] * passenge.size(0)
                        category_query_mask = torch.ones(category_query.size(1)).float().to(self.device).unsqueeze(0)
                        category_query_seg = torch.tensor(category_query_seg).long().to(self.device).unsqueeze(0)
                        # sentiment
                        sentiment_query_seg = [0] * len(sentiment_query)
                        sentiment_query = torch.tensor(sentiment_query).long().to(self.device)
                        sentiment_query = torch.cat([sentiment_query, passenge], -1).to(self.device).unsqueeze(0)
                        sentiment_query_seg += [1] * passenge.size(0)
                        sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().to(self.device).unsqueeze(0)
                        sentiment_query_seg = torch.tensor(sentiment_query_seg).long().to(self.device).unsqueeze(0)

                        # inference results of category
                        category_scores = self.model(category_query, category_query_mask, category_query_seg, 1)
                        category_scores = F.softmax(category_scores, dim=1)
                        category_predicted = torch.argmax(category_scores[0], dim=0).item()

                        # inference results of sentiment
                        sentiment_scores = self.model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 2)
                        sentiment_scores = F.softmax(sentiment_scores, dim=1)
                        sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()
                        # opinion对应的adverb
                        for c in range(predict_adverb_num):
                            # 三元组、五元组组合
                            asp, opi, adv = [], [], []
                            asp.append(final_asp_idx_list[a][0])  # asp 的start index
                            asp.append(final_asp_idx_list[a][1])  # asp 的end index
                            opi.append(final_opi_idx_list[a][b][0])
                            opi.append(final_opi_idx_list[a][b][1])
                            adv.append(final_adv_idx_list[a][b][c][0])
                            adv.append(final_adv_idx_list[a][b][c][1])

                            asp_opi = [asp, opi]
                            aste_triplet_predict = [asp, opi, sentiment_predicted]
                            aoc_triplet_predict = [asp, opi, category_predicted]
                            aoa_triplet_predict = [asp, opi, adv]
                            quintuple_predict = [asp, category_predicted, opi, adv, sentiment_predicted]

                            if asp not in asp_predict:
                                asp_predict.append(asp)
                            if opi not in opi_predict:
                                opi_predict.append(opi)
                            if adv not in adv_predict:
                                adv_predict.append(adv)
                            if asp_opi not in asp_opi_predict:
                                asp_opi_predict.append(asp_opi)
                            if aste_triplet_predict not in aste_triplets_predict:
                                aste_triplets_predict.append(aste_triplet_predict)
                            if aoc_triplet_predict not in aoc_triplets_predict:
                                aoc_triplets_predict.append(aoc_triplet_predict)
                            if aoa_triplet_predict not in aoa_triplets_predict:
                                aoa_triplets_predict.append(aoa_triplet_predict)
                            if quintuple_predict not in quintuples_predict:
                                quintuples_predict.append(quintuple_predict)

                acoxs_score.update(batch.aspects[0], batch.opinions[0], batch.adverbs[0], batch.pairs[0],
                                   batch.aste_triplets[0], batch.aoc_triplets[0], batch.aoa_triplets[0],
                                   batch.quintuples[0],
                                   asp_predict, opi_predict, adv_predict, asp_opi_predict,
                                   aste_triplets_predict, aoc_triplets_predict, aoa_triplets_predict,
                                   quintuples_predict)
                one_json = {'sentence': ' '.join(batch.sentence_token[0]), 'pred': str(quintuples_predict),
                            'gold': str(batch.quintuples[0])}
                json_res.append(one_json)
                with open(os.path.join(self.args.output_dir, self.args.task, self.args.data_type, 'predict.json'), 'w',
                          encoding='utf-8') as fP:
                    json.dump(json_res, fP, ensure_ascii=False, indent=4)
                return acoxs_score.compute()

    # def batch_eval(self, eval_dataloader):
    #     start_time = time.time()
    #     json_res = []
    #     acos_score = ACOSScore(self.logger)
    #     self.model.eval()
    #     Forward_Q1, Backward_Q1, Forward_Q2, Backward_Q2, Q3, Q4 = get_English_Template()
    #     f_asp_imp_start = 5
    #     b_opi_imp_start = 5
    #     for batch in tqdm(eval_dataloader):
    #         forward_pair_list, forward_pair_prob, forward_pair_ind_list = [], [], []
    #
    #         backward_pair_list, backward_pair_prob, backward_pair_ind_list = [], [], []
    #
    #         # forward q_1 nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
    #         passenges = []
    #         for p in range(len(batch.forward_asp_answer_start)):
    #             passenge_index = batch.forward_asp_answer_start[p].gt(-1).float().nonzero()
    #             passenge = batch.forward_asp_query[p][passenge_index].squeeze(1)
    #             passenges.append(passenge)
    #         batch_size = len(passenges)
    #         # 进行第一轮
    #         forward_len = len(batch.forward_asp_query)
    #         turn1_query = torch.cat((batch.forward_asp_query, batch.backward_opi_query), dim=0)
    #         turn1_mask = torch.cat((batch.forward_asp_query_mask, batch.backward_opi_query_mask), dim=0)
    #         turn1_seg = torch.cat((batch.forward_asp_query_seg, batch.backward_opi_query_seg), dim=0)
    #         turn1_start_scores, turn1_end_scores = self.model(turn1_query.to('cpu'),
    #                                                           turn1_mask.to('cpu'),
    #                                                           turn1_seg.to('cpu'), 0)
    #         f_asp_start_scores, f_asp_end_scores = turn1_start_scores[:forward_len], turn1_end_scores[:forward_len]
    #         b_opi_start_scores, b_opi_end_scores = turn1_start_scores[forward_len:], turn1_end_scores[forward_len:]
    #
    #         f_asp_start_scores = F.softmax(f_asp_start_scores, dim=-1)
    #         f_asp_end_scores = F.softmax(f_asp_end_scores, dim=-1)
    #         f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=-1)
    #         f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=-1)
    #
    #         b_opi_start_scores = F.softmax(b_opi_start_scores, dim=-1)
    #         b_opi_end_scores = F.softmax(b_opi_end_scores, dim=-1)
    #         b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=-1)
    #         b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=-1)
    #
    #         f_asp_start_indexs, f_asp_end_indexs, f_asp_probs = [], [], []
    #         for b in range(f_asp_end_prob.size(0)):
    #             f_asp_start_prob_temp = []
    #             f_asp_end_prob_temp = []
    #             f_asp_start_index_temp = []
    #             f_asp_end_index_temp = []
    #             for i in range(f_asp_start_ind[b].size(0)):
    #                 # 填充部分不需要考虑
    #                 if batch.sentence_len[b] + f_asp_imp_start < i:
    #                     break
    #                 if batch.forward_asp_answer_start[b, i] != -1:
    #                     if f_asp_start_ind[b][i].item() == 1:
    #                         f_asp_start_index_temp.append(i)
    #                         f_asp_start_prob_temp.append(f_asp_start_prob[b][i].item())
    #                     if f_asp_end_ind[b][i].item() == 1:
    #                         f_asp_end_index_temp.append(i)
    #                         f_asp_end_prob_temp.append(f_asp_end_prob[b][i].item())
    #
    #             f_asp_start_index, f_asp_end_index, f_asp_prob = filter_unpaired(
    #                 f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp,
    #                 f_asp_imp_start)
    #             f_asp_start_indexs.append(f_asp_start_index)
    #             f_asp_end_indexs.append(f_asp_end_index)
    #             f_asp_probs.append(f_asp_prob)
    #
    #         # f_asp_start_indexs, f_asp_end_indexs, f_asp_probs
    #         f_asp_nums = []
    #         imp_starts = []
    #         f_opinion_querys, f_opinion_segs, f_opinion_masks = [], [], []
    #         f_opinion_lens = []
    #         for b in range(len(f_asp_start_indexs)):
    #             f_asp_nums.append(len(f_asp_start_indexs[b]))
    #             for i in range(len(f_asp_start_indexs[b])):
    #                 if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                     opinion_query = self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[:7]])
    #                 else:
    #                     opinion_query = self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[:6]])
    #                 for j in range(f_asp_start_indexs[b][i], f_asp_end_indexs[b][i] + 1):
    #                     opinion_query.append(batch.forward_asp_query[b][j].item())
    #                 if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                     opinion_query.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[7:]]))
    #                 else:
    #                     opinion_query.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Forward_Q2[6:]]))
    #                 imp_start = len(opinion_query)
    #                 imp_starts.append(imp_start)
    #
    #                 opinion_query_seg = [0] * len(opinion_query)
    #
    #                 opinion_query = torch.tensor(opinion_query).long()
    #                 opinion_query = torch.cat([opinion_query, passenges[b]], -1)
    #                 opinion_query_seg += [1] * passenges[b].size(0)
    #                 opinion_query_mask = torch.ones(opinion_query.size(0)).float()
    #                 opinion_query_seg = torch.tensor(opinion_query_seg).long()
    #
    #                 f_opinion_querys.append(opinion_query)
    #                 f_opinion_segs.append(opinion_query_seg)
    #                 f_opinion_masks.append(opinion_query_mask)
    #                 f_opinion_lens.append(len(opinion_query))
    #
    #         batch_f_ao_idxs = []
    #         if f_opinion_querys:
    #             # 进行padding
    #             f_opinion_querys = pad_sequence(f_opinion_querys, batch_first=True, padding_value=0).to('cpu')
    #             f_opinion_segs = pad_sequence(f_opinion_segs, batch_first=True, padding_value=1).to('cpu')
    #             f_opinion_masks = pad_sequence(f_opinion_masks, batch_first=True, padding_value=0).to('cpu')
    #
    #             f_opi_start_scores, f_opi_end_scores = self.model(f_opinion_querys, f_opinion_masks, f_opinion_segs, 0)
    #
    #             f_opi_start_scores = F.softmax(f_opi_start_scores, dim=-1)
    #             f_opi_end_scores = F.softmax(f_opi_end_scores, dim=-1)
    #             f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=-1)
    #             f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=-1)
    #
    #             # 对asp进行batch处理
    #             b_f_asp_start_indexs = [asp for asps in f_asp_start_indexs for asp in asps]
    #             b_f_asp_end_indexs = [asp for asps in f_asp_end_indexs for asp in asps]
    #             b_f_asp_probs = [asp for asps in f_asp_probs for asp in asps]
    #             batch_map_list = [d for c in [[a] * f_asp_nums[a] for a in range(len(f_asp_nums))] for d in c]
    #             for b in range(f_opi_end_prob.size(0)):
    #                 temp_forward_pair_list, temp_forward_pair_prob, temp_forward_pair_ind_list = [], [], []
    #                 f_opi_start_prob_temp = []
    #                 f_opi_end_prob_temp = []
    #                 f_opi_start_index_temp = []
    #                 f_opi_end_index_temp = []
    #                 for k in range(f_opi_start_ind[b].size(0)):
    #                     # 填充部分不需要考虑
    #                     if f_opinion_lens[b] - 1 < k:
    #                         break
    #                     if f_opinion_segs[b, k].item() == 1:
    #                         if f_opi_start_ind[b][k].item() == 1:
    #                             f_opi_start_index_temp.append(k)
    #                             f_opi_start_prob_temp.append(f_opi_start_prob[b][k].item())
    #                         if f_opi_end_ind[b][k].item() == 1:
    #                             f_opi_end_index_temp.append(k)
    #                             f_opi_end_prob_temp.append(f_opi_end_prob[b][k].item())
    #                 f_opi_start_index, f_opi_end_index, f_opi_prob = filter_unpaired(
    #                     f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp,
    #                     imp_starts[b])
    #                 # 进行结果抽取
    #                 for idx in range(len(f_opi_start_index)):
    #                     asp = [batch.forward_asp_query[batch_map_list[b]][j].item() for j in
    #                            range(b_f_asp_start_indexs[b], b_f_asp_end_indexs[b] + 1)]
    #                     opi = [f_opinion_querys[b][j].item() for j in
    #                            range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
    #                     # null -> -1, -1
    #                     if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                         asp_ind = [b_f_asp_start_indexs[b] - 8, b_f_asp_end_indexs[b] - 8]
    #                     else:
    #                         asp_ind = [b_f_asp_start_indexs[b] - 6, b_f_asp_end_indexs[b] - 6]
    #                     opi_ind = [f_opi_start_index[idx] - imp_starts[b] - 1, f_opi_end_index[idx] - imp_starts[b] - 1]
    #                     temp_prob = b_f_asp_probs[b] * f_opi_prob[idx]
    #                     if asp_ind + opi_ind not in temp_forward_pair_list:
    #                         batch_f_ao_idxs.append(batch_map_list[b])
    #                         temp_forward_pair_list.append([asp] + [opi])
    #                         temp_forward_pair_prob.append(temp_prob)
    #                         temp_forward_pair_ind_list.append(asp_ind + opi_ind)
    #                 forward_pair_list.append(temp_forward_pair_list)
    #                 forward_pair_prob.append(temp_forward_pair_prob)
    #                 forward_pair_ind_list.append(temp_forward_pair_ind_list)
    #         # 进行合并
    #         forward_pair_list = [f_p for f_ps in forward_pair_list for f_p in f_ps]
    #         forward_pair_prob = [f_p for f_ps in forward_pair_prob for f_p in f_ps]
    #         forward_pair_ind_list = [f_p for f_ps in forward_pair_ind_list for f_p in f_ps]
    #         # backward q_1
    #         b_opi_start_indexs, b_opi_end_indexs, b_opi_probs = [], [], []
    #         for b in range(b_opi_end_prob.size(0)):
    #             b_opi_start_prob_temp = []
    #             b_opi_end_prob_temp = []
    #             b_opi_start_index_temp = []
    #             b_opi_end_index_temp = []
    #             for i in range(b_opi_start_ind[b].size(0)):
    #                 # 填充部分不需要考虑
    #                 if batch.sentence_len[b] + b_opi_imp_start < i:
    #                     break
    #                 if batch.backward_opi_answer_start[b, i] != -1:
    #                     if b_opi_start_ind[b][i].item() == 1:
    #                         b_opi_start_index_temp.append(i)
    #                         b_opi_start_prob_temp.append(b_opi_start_prob[b][i].item())
    #                     if b_opi_end_ind[b][i].item() == 1:
    #                         b_opi_end_index_temp.append(i)
    #                         b_opi_end_prob_temp.append(b_opi_end_prob[b][i].item())
    #
    #             b_opi_start_index, b_opi_end_index, b_opi_prob = filter_unpaired(
    #                 b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp,
    #                 b_opi_imp_start)
    #             b_opi_start_indexs.append(b_opi_start_index)
    #             b_opi_end_indexs.append(b_opi_end_index)
    #             b_opi_probs.append(b_opi_prob)
    #
    #         # backward q_2
    #         b_opi_nums = []
    #         imp_starts = []
    #         b_aspect_querys, b_aspect_segs, b_aspect_masks = [], [], []
    #         b_aspect_lens = []
    #         for b in range(len(b_opi_start_indexs)):
    #             b_opi_nums.append(len(b_opi_start_indexs[b]))
    #             for i in range(len(b_opi_start_indexs[b])):
    #                 if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                     aspect_query = self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[:7]])
    #                 else:
    #                     aspect_query = self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[:6]])
    #                 for j in range(b_opi_start_indexs[b][i], b_opi_end_indexs[b][i] + 1):
    #                     aspect_query.append(batch.backward_opi_query[b][j].item())
    #                 if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                     aspect_query.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[7:]]))
    #                 else:
    #                     aspect_query.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Backward_Q2[6:]]))
    #                 imp_start = len(aspect_query)
    #                 imp_starts.append(imp_start)
    #
    #                 aspect_query_seg = [0] * len(aspect_query)
    #
    #                 aspect_query = torch.tensor(aspect_query).long()
    #                 aspect_query = torch.cat([aspect_query, passenges[b]], -1)
    #                 aspect_query_seg += [1] * passenges[b].size(0)
    #                 aspect_query_mask = torch.ones(aspect_query.size(0)).float()
    #                 aspect_query_seg = torch.tensor(aspect_query_seg).long()
    #
    #                 b_aspect_querys.append(aspect_query)
    #                 b_aspect_segs.append(aspect_query_seg)
    #                 b_aspect_masks.append(aspect_query_mask)
    #                 b_aspect_lens.append(len(aspect_query_mask))
    #
    #         batch_b_ao_idxs = []
    #         if b_aspect_querys:
    #             # 进行padding
    #             b_aspect_querys = pad_sequence(b_aspect_querys, batch_first=True, padding_value=0).to('cpu')
    #             b_aspect_segs = pad_sequence(b_aspect_segs, batch_first=True, padding_value=1).to('cpu')
    #             b_aspect_masks = pad_sequence(b_aspect_masks, batch_first=True, padding_value=0).to('cpu')
    #
    #             b_asp_start_scores, b_asp_end_scores = self.model(b_aspect_querys, b_aspect_masks, b_aspect_segs, 0)
    #
    #             b_asp_start_scores = F.softmax(b_asp_start_scores, dim=-1)
    #             b_asp_end_scores = F.softmax(b_asp_end_scores, dim=-1)
    #             b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=-1)
    #             b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=-1)
    #
    #             # 对opi进行batch处理
    #             b_b_opi_start_indexs = [opi for opis in b_opi_start_indexs for opi in opis]
    #             b_b_opi_end_indexs = [opi for opis in b_opi_end_indexs for opi in opis]
    #             b_b_opi_probs = [opi for opis in b_opi_probs for opi in opis]
    #             f_batch_map_list = [d for c in [[a] * b_opi_nums[a] for a in range(len(b_opi_nums))] for d in c]
    #             for b in range(b_asp_end_prob.size(0)):
    #                 temp_backward_pair_list, temp_backward_pair_prob, temp_backward_pair_ind_list = [], [], []
    #                 b_asp_start_prob_temp = []
    #                 b_asp_end_prob_temp = []
    #                 b_asp_start_index_temp = []
    #                 b_asp_end_index_temp = []
    #                 for k in range(b_asp_start_ind[b].size(0)):
    #                     # 填充部分不需要考虑
    #                     if b_aspect_lens[b] - 1 < k:
    #                         break
    #                     if b_aspect_segs[b, k].item() == 1:
    #                         if b_asp_start_ind[b][k].item() == 1:
    #                             b_asp_start_index_temp.append(k)
    #                             b_asp_start_prob_temp.append(b_asp_start_prob[b][k].item())
    #                         if b_asp_end_ind[b][k].item() == 1:
    #                             b_asp_end_index_temp.append(k)
    #                             b_asp_end_prob_temp.append(b_asp_end_prob[b][k].item())
    #
    #                 b_asp_start_index, b_asp_end_index, b_asp_prob = filter_unpaired(
    #                     b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp,
    #                     imp_starts[b])
    #                 # 进行结果抽取
    #                 for idx in range(len(b_asp_start_index)):
    #                     opi = [batch.backward_opi_query[f_batch_map_list[b]][j].item() for j in
    #                            range(b_b_opi_start_indexs[b], b_b_opi_end_indexs[b] + 1)]
    #                     asp = [b_aspect_querys[b][j].item() for j in
    #                            range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
    #                     # null -> -1, -1
    #                     asp_ind = [b_asp_start_index[idx] - imp_starts[b] - 1, b_asp_end_index[idx] - imp_starts[b] - 1]
    #                     if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                         opi_ind = [b_b_opi_start_indexs[b] - 8, b_b_opi_end_indexs[b] - 8]
    #                     else:
    #                         opi_ind = [b_b_opi_start_indexs[b] - 6, b_b_opi_end_indexs[b] - 6]
    #                     temp_prob = b_asp_prob[idx] * b_b_opi_probs[b]
    #                     if asp_ind + opi_ind not in temp_backward_pair_ind_list:
    #                         batch_b_ao_idxs.append(f_batch_map_list[b])
    #                         temp_backward_pair_list.append([asp] + [opi])
    #                         temp_backward_pair_prob.append(temp_prob)
    #                         temp_backward_pair_ind_list.append(asp_ind + opi_ind)
    #                 backward_pair_list.append(temp_backward_pair_list)
    #                 backward_pair_prob.append(temp_backward_pair_prob)
    #                 backward_pair_ind_list.append(temp_backward_pair_ind_list)
    #         # 进行合并
    #         backward_pair_list = [b_p for b_ps in backward_pair_list for b_p in b_ps]
    #         backward_pair_prob = [b_p for b_ps in backward_pair_prob for b_p in b_ps]
    #         backward_pair_ind_list = [b_p for b_ps in backward_pair_ind_list for b_p in b_ps]
    #         if self.args.use_Forward:
    #             batch_final_idxs = []
    #             final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
    #             for idx in range(len(forward_pair_list)):
    #                 if forward_pair_list[idx][0] not in final_asp_list:
    #                     final_asp_list.append(forward_pair_list[idx][0])
    #                     final_opi_list.append([forward_pair_list[idx][1]])
    #                     final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
    #                     final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
    #                     batch_final_idxs.append(batch_f_ao_idxs[idx])
    #                 else:
    #                     asp_index = final_asp_list.index(forward_pair_list[idx][0])
    #                     if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
    #                         final_opi_list[asp_index].append(forward_pair_list[idx][1])
    #                         final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
    #         elif self.args.use_Backward:
    #             batch_final_idxs = []
    #             final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list = [], [], [], []
    #             for idx in range(len(backward_pair_list)):
    #                 if backward_pair_list[idx][0] not in final_asp_list:
    #                     final_asp_list.append(backward_pair_list[idx][0])
    #                     final_opi_list.append([backward_pair_list[idx][1]])
    #                     final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
    #                     final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
    #                     batch_final_idxs.append(batch_b_ao_idxs[idx])
    #                 else:
    #                     asp_index = final_asp_list.index(backward_pair_list[idx][0])
    #                     if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
    #                         final_opi_list[asp_index].append(backward_pair_list[idx][1])
    #                         final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])
    #         else:
    #             # combine forward and backward pairs
    #             final_asp_list, final_opi_list, final_asp_ind_list, final_opi_ind_list, batch_final_idxs = batch_pair_combine(
    #                 forward_pair_list,
    #                 forward_pair_prob,
    #                 forward_pair_ind_list,
    #                 backward_pair_list,
    #                 backward_pair_prob,
    #                 backward_pair_ind_list,
    #                 batch_f_ao_idxs,
    #                 batch_b_ao_idxs,
    #                 batch_size,
    #                 self.args.alpha,
    #                 self.args.beta)
    #
    #         # category sentiment
    #         batch_quad_idxs = []
    #         ao_category_querys, ao_category_segs, ao_category_masks = [], [], []
    #         ao_sentiment_querys, ao_sentiment_segs, ao_sentiment_masks = [], [], []
    #         for idx in range(len(final_asp_list)):
    #             predict_opinion_num = len(final_opi_list[idx])
    #             # category sentiment
    #             if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                 category_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[:7]])
    #                 sentiment_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:7]])
    #             else:
    #                 category_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[:6]])
    #                 sentiment_query = self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:6]])
    #             category_query += final_asp_list[idx]
    #             sentiment_query += final_asp_list[idx]
    #             if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                 [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:9]]
    #                 category_query += self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[7:8]])
    #                 sentiment_query += self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[7:8]])
    #             else:
    #                 category_query += self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[6:9]])
    #                 sentiment_query += self.tokenizer.convert_tokens_to_ids(
    #                     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:9]])
    #
    #             # 拼接opinion
    #             batch_quad_idx = []
    #             for idy in range(predict_opinion_num):
    #                 batch_quad_idxs.append(batch_final_idxs[idx])
    #                 category_query2 = category_query + final_opi_list[idx][idy]
    #                 sentiment_query2 = sentiment_query + final_opi_list[idx][idy]
    #                 if self.args.task.lower() == "asqe" or self.args.task.lower() == 'zh_quad':
    #                     category_query2.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[8:]]))
    #                     sentiment_query2.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[8:]]))
    #                 else:
    #                     category_query2.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q3[9:]]))
    #                     sentiment_query2.extend(self.tokenizer.convert_tokens_to_ids(
    #                         [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[9:]]))
    #
    #                 category_query_seg = [0] * len(category_query2)
    #                 category_query2 = torch.tensor(category_query2).long()
    #                 category_query2 = torch.cat([category_query2, passenges[batch_final_idxs[idx]]], -1)
    #                 category_query_seg += [1] * passenges[batch_final_idxs[idx]].size(0)
    #                 category_query_mask = torch.ones(category_query2.size(0)).float()
    #                 category_query_seg = torch.tensor(category_query_seg).long()
    #
    #                 sentiment_query_seg = [0] * len(sentiment_query2)
    #                 sentiment_query2 = torch.tensor(sentiment_query2).long()
    #                 sentiment_query2 = torch.cat([sentiment_query2, passenges[batch_final_idxs[idx]]], -1)
    #                 sentiment_query_seg += [1] * passenges[batch_final_idxs[idx]].size(0)
    #                 sentiment_query_mask = torch.ones(sentiment_query2.size(0)).float()
    #                 sentiment_query_seg = torch.tensor(sentiment_query_seg).long()
    #
    #                 ao_category_querys.append(category_query2)
    #                 ao_category_segs.append(category_query_seg)
    #                 ao_category_masks.append(category_query_mask)
    #
    #                 ao_sentiment_querys.append(sentiment_query2)
    #                 ao_sentiment_segs.append(sentiment_query_seg)
    #                 ao_sentiment_masks.append(sentiment_query_mask)
    #         if ao_category_querys:
    #             # 进行padding
    #             ao_category_querys = pad_sequence(ao_category_querys, batch_first=True, padding_value=0).to('cpu')
    #             ao_category_segs = pad_sequence(ao_category_segs, batch_first=True, padding_value=1).to('cpu')
    #             ao_category_masks = pad_sequence(ao_category_masks, batch_first=True, padding_value=0).to('cpu')
    #
    #             ao_sentiment_querys = pad_sequence(ao_sentiment_querys, batch_first=True, padding_value=0).to('cpu')
    #             ao_sentiment_segs = pad_sequence(ao_sentiment_segs, batch_first=True, padding_value=1).to('cpu')
    #             ao_sentiment_masks = pad_sequence(ao_sentiment_masks, batch_first=True, padding_value=0).to('cpu')
    #
    #             category_scores = self.model(ao_category_querys, ao_category_masks, ao_category_segs, 1)
    #             category_scores = F.softmax(category_scores, dim=-1)
    #             category_predicted = torch.argmax(category_scores, dim=-1)
    #
    #             sentiment_scores = self.model(ao_sentiment_querys, ao_sentiment_masks, ao_sentiment_segs, 2)
    #             sentiment_scores = F.softmax(sentiment_scores, dim=-1)
    #             sentiment_predicted = torch.argmax(sentiment_scores, dim=-1)
    #
    #             ao_nums = [len(fi) for fi in final_opi_list]
    #             ao_batch_map_list = [d for c in [[a] * ao_nums[a] for a in range(len(ao_nums))] for d in c]
    #             final_opi_ind_list = [opi for opis in final_opi_ind_list for opi in opis]
    #             # 三元组、四元组组合
    #             quadruples_predicts = [[] for _ in range(len(batch.quadruples))]
    #             for idx in range(len(final_opi_ind_list)):
    #                 asp_f, opi_f = [], []
    #                 asp_f.append(final_asp_ind_list[ao_batch_map_list[idx]][0])
    #                 asp_f.append(final_asp_ind_list[ao_batch_map_list[idx]][1])
    #                 opi_f.append(final_opi_ind_list[idx][0])
    #                 opi_f.append(final_opi_ind_list[idx][1])
    #                 quadruple_predict = [asp_f, category_predicted[idx].item(), opi_f, sentiment_predicted[idx].item()]
    #                 if quadruple_predict not in quadruples_predicts[batch_quad_idxs[idx]]:
    #                     quadruples_predicts[batch_quad_idxs[idx]].append(quadruple_predict)
    #         else:
    #             quadruples_predicts = [[] for _ in range(len(batch.quadruples))]
    #         acos_score.update2(batch.quadruples, quadruples_predicts)
    #
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"*************************执行总耗时：{elapsed_time}秒*************************")
    #     return acos_score.compute2()

    def inference(self, reviews):
        # 评估模式
        self.model.eval()

        # 是否支持gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 构建类别，情感字典
        category_id = get_aspect_category(self.args.task.lower(), self.args.data_type)[-1]
        sentiment_id = get_sentiment(self.args.task.lower())[-1]

        # 准备查询问题
        q1_asp, q2_asp2opi, q3_opi2adv, q4_adv, q5_adv2opi, q6_opi2asp, q7_c, q8_s = get_Chinese_Template()

        # 准备查询列表
        f_asp_query_list, f_asp_mask_list, f_asp_seg_list = [], [], []
        # b_opi_query_list, b_opi_mask_list, b_opi_seg_list = [], [], []
        b_adv_query_list, b_adv_mask_list, b_adv_seg_list = [], [], []

        # 准备索引列表 "null + review" 在 f_temp_text 中的下标列表 辅助用 适用于 a o adv 单独提取
        idx_list, vocab_idx_list = [], []

        # 前期准备工作 下标+id列表   前向后向两个开始的问题
        for review in reviews:
            review = review.split(' ')
            # ***forward***
            f_temp_text = q1_asp + ["null"] + review
            f_temp_text = list(map(self.tokenizer.tokenize, f_temp_text))
            f_temp_text = [elem for outer_list in f_temp_text for elem in outer_list]
            print(f"input_text: {f_temp_text}")

            # 生成 "null + review" 在 f_temp_text 中的下标列表
            idx_list.append([idx + len(q1_asp) for idx in range(len(f_temp_text) - len(q1_asp))])
            # 生成tokenizer词汇表中token对应的id
            vocab_idx_list.append(self.tokenizer.convert_tokens_to_ids(["null"] + review))

            # Bert输入三件套   1.id列表 2.掩码列表 3.标记序列（区分问题、查询）
            # 1.id列表
            f_asp_query = self.tokenizer.convert_tokens_to_ids(f_temp_text)
            f_asp_query_list.append(f_asp_query)
            print('.' * 100)
            print(f"f_asp_query: {f_asp_query}")
            # 2.掩码列表
            f_asp_mask = [1 for _ in range(len(f_asp_query))]
            f_asp_mask_list.append(f_asp_mask)
            # 3.标记序列
            q1_len = len(self.tokenizer.convert_tokens_to_ids(q1_asp))
            f_asp_seg = [0] * q1_len + [1] * (len(f_asp_query) - q1_len)
            f_asp_seg_list.append(f_asp_seg)

            # ***backward***
            # b_temp_text = q4_opi + ["null"] + review
            # b_temp_text = list(map(self.tokenizer.tokenize, b_temp_text))
            # b_temp_text = [elem for outer_list in b_temp_text for elem in outer_list]
            #
            # b_opi_query = self.tokenizer.convert_tokens_to_ids(b_temp_text)
            # b_opi_query_list.append(b_opi_query)
            # b_opi_mask = [1 for _ in range(len(b_opi_query))]
            # b_opi_mask_list.append(b_opi_mask)
            # q4_len = len(self.tokenizer.convert_tokens_to_ids(q4_opi))
            # b_opi_seg = [0] * q4_len + [1] * (len(b_opi_query) - q4_len)
            # b_opi_seg_list.append(b_opi_seg)

            # ***backward***
            b_temp_text = q4_adv + ["null"] + review
            b_temp_text = list(map(self.tokenizer.tokenize, b_temp_text))
            b_temp_text = [elem for outer_list in b_temp_text for elem in outer_list]

            b_adv_query = self.tokenizer.convert_tokens_to_ids(b_temp_text)
            b_adv_query_list.append(b_adv_query)
            b_adv_mask = [1 for _ in range(len(b_adv_query))]
            b_adv_mask_list.append(b_adv_mask)
            q4_len = len(self.tokenizer.convert_tokens_to_ids(q4_adv))
            b_adv_seg = [0] * q4_len + [1] * (len(b_adv_query) - q4_len)
            b_adv_seg_list.append(b_adv_seg)

        # 执行各个子任务
        for i in range(len(f_asp_query_list)):
            # ******************** Six SubTask ********************
            # 结果存放列表
            quadruples_predict, quintuples_predict = [], []
            # 数据容器准备  存入列表时这些数据的下标是以review为参考的 去除 query 和 null 分析道null 会以(-1,-1)代替
            forward_pair_list, forward_pair_prob, forward_pair_idx_list = [], [], []
            forward_triplet_list, forward_triplet_prob, forward_triplet_idx_list = [], [], []
            backward_pair_list, backward_pair_prob, backward_pair_idx_list = [], [], []
            backward_triplet_list, backward_triplet_prob, backward_triplet_idx_list = [], [], []

            # ********************Q1: Extract Aspect********************
            # transform type list -> tensor  dimension： 2
            f_asp_query = torch.tensor([f_asp_query_list[i]]).long()
            f_asp_query_mask = torch.tensor([f_asp_mask_list[i]]).long()
            f_asp_query_seg = torch.tensor([f_asp_seg_list[i]]).long()

            # 调用模型 获取 aspect 提取结果
            f_asp_start_scores, f_asp_end_scores = self.model(
                f_asp_query.to(device),
                f_asp_query_mask.to(device),
                f_asp_query_seg.to(device),
                0
            )
            # 将输出结果转为概率分布
            # 三维(只是为了符合输入的形式)->二维 [sequences_length, out_features] 对第2个维度进行概率分布
            f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
            f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)

            # torch.max() 返回张量指定维度上的最大值(值是概率)和最大索引
            f_asp_start_prob, f_asp_start_idx = torch.max(f_asp_start_scores, dim=1)
            f_asp_end_prob, f_asp_end_idx = torch.max(f_asp_end_scores, dim=1)

            f_asp_start_prob_temp, f_asp_end_prob_temp = [], []
            f_asp_start_idx_temp, f_asp_end_idx_temp = [], []

            # 初筛：选择符合条件的下标
            for idx in range(f_asp_start_idx.size(0)):
                if idx in idx_list[i]:  # 模型预测方面下标到问题部分 直接跳过
                    if f_asp_start_idx[idx].item() == 1:  # 遍历找到最大可能对应的下标索引
                        f_asp_start_idx_temp.append(idx)
                        f_asp_start_prob_temp.append(f_asp_start_prob[idx].item())
                    if f_asp_end_idx[idx].item() == 1:
                        f_asp_end_idx_temp.append(idx)
                        f_asp_end_prob_temp.append(f_asp_end_prob[idx].item())
            # 再筛 得到最终结果和一个整体的可能性
            f_asp_start_idx, f_asp_end_idx, f_asp_prob = filter_unpaired(
                f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_idx_temp, f_asp_end_idx_temp,
                imp_start=8
            )

            # ********************Q2: Aspect->Opinion ********************
            vocab_idx = torch.tensor(vocab_idx_list[i]).long()  # null + review 在vocab.txt中每个分词对应的id
            for a in range(len(f_asp_start_idx)):  # 已经提取了aspect，将aspect加入q2
                f_opi_query = self.tokenizer.convert_tokens_to_ids(q2_asp2opi)
                for j in range(f_asp_start_idx[a], f_asp_end_idx[a] + 1):
                    f_opi_query.insert(5, f_asp_query[0][j].item())

                f_opi_query_length = len(f_opi_query)
                f_opi_query_seg = [0] * len(f_opi_query)
                imp_start = len(f_opi_query)
                f_opi_query = torch.tensor(f_opi_query).long()
                f_opi_query = torch.cat([f_opi_query, vocab_idx], -1).to(device).unsqueeze(0)
                # print('.' * 100)
                # print(f"f_opi_query: {f_opi_query}")
                f_opi_query_mask = torch.ones(f_opi_query.size(1)).float().to(device).unsqueeze(0)
                f_opi_query_seg += [1] * vocab_idx.size(0)
                f_opi_query_seg = torch.tensor(f_opi_query_seg).long().to(device).unsqueeze(0)

                f_opi_start_scores, f_opi_end_scores = self.model(f_opi_query, f_opi_query_mask, f_opi_query_seg, 0)

                f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
                f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
                f_opi_start_prob, f_opi_start_idx = torch.max(f_opi_start_scores, dim=1)
                f_opi_end_prob, f_opi_end_idx = torch.max(f_opi_end_scores, dim=1)

                f_opi_start_prob_temp, f_opi_end_prob_temp = [], []
                f_opi_start_idx_temp, f_opi_end_idx_temp = [], []

                for k in range(f_opi_start_idx.size(0)):
                    if f_opi_query_seg[0, k] == 1:  # 把判定到query部分的id全部过滤掉
                        if f_opi_start_idx[k].item() == 1:
                            f_opi_start_idx_temp.append(k)
                            f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                        if f_opi_end_idx[k].item() == 1:
                            f_opi_end_idx_temp.append(k)
                            f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

                f_opi_start_idx, f_opi_end_idx, f_opi_prob = filter_unpaired(
                    f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_idx_temp, f_opi_end_idx_temp,
                    imp_start=imp_start
                )
                # 组合 形成 a-o pair
                for k in range(len(f_opi_start_idx)):
                    asp = [f_asp_query[0][j].item() for j in range(f_asp_start_idx[a], f_asp_end_idx[a] + 1)]
                    opi = [f_opi_query[0][j].item() for j in range(f_opi_start_idx[k], f_opi_end_idx[k] + 1)]

                    # 问题 + null     没有null(0,0) -> (-1, -1)   回归最原始评论中关键元素的下标
                    asp_idx = [f_asp_start_idx[a] - len(q1_asp) - 2, f_asp_end_idx[a] - len(q1_asp) - 2]
                    opi_idx = [f_opi_start_idx[k] - f_opi_query_length - 1, f_opi_end_idx[k] - f_opi_query_length - 1]

                    # 计算配对概率
                    temp_prob = f_asp_prob[a] * f_opi_prob[k]

                    if asp_idx + opi_idx not in forward_pair_idx_list:
                        forward_pair_list.append([asp] + [opi])
                        forward_pair_prob.append(temp_prob)
                        forward_pair_idx_list.append(asp_idx + opi_idx)

                # ********************Q3: Aspect + Opinion -> Adverb ********************
                # ["[CLS]", "这", "个", "方", "面", "5意", "见", "-8的", "-7副", "-6词", "-5有", "-4哪", "-3些", "？", "[SEP]"]
                for b in range(len(f_opi_start_idx)):
                    f_adv_query = self.tokenizer.convert_tokens_to_ids(q3_opi2adv)
                    # insert aspect
                    for j in range(f_asp_start_idx[a], f_asp_end_idx[a] + 1):
                        f_adv_query.insert(5, f_asp_query[0][j].item())
                    # insert opinion
                    for j in range(f_opi_start_idx[b], f_opi_end_idx[b] + 1):
                        f_adv_query.insert(-8, f_opi_query[0][j].item())

                    f_adv_query_length = len(f_adv_query)

                    f_adv_query_seg = [0] * len(f_adv_query)
                    imp_start = len(f_adv_query)  # 每个都是不一样的
                    f_adv_query = torch.tensor(f_adv_query).long()
                    f_adv_query = torch.cat([f_adv_query, vocab_idx], -1).to(device).unsqueeze(0)
                    # print('.' * 100)
                    # print(f"f_adv_query: {f_adv_query}")
                    f_adv_query_mask = torch.ones(f_adv_query.size(1)).float().to(device).unsqueeze(0)
                    f_adv_query_seg += [1] * vocab_idx.size(0)  # 还是list
                    f_adv_query_seg = torch.tensor(f_adv_query_seg).long().to(device).unsqueeze(0)

                    f_adv_start_scores, f_adv_end_scores = self.model(f_adv_query, f_adv_query_mask, f_adv_query_seg, 0)

                    f_adv_start_scores = F.softmax(f_adv_start_scores[0], dim=1)
                    f_adv_end_scores = F.softmax(f_adv_end_scores[0], dim=1)
                    f_adv_start_prob, f_adv_start_idx = torch.max(f_adv_start_scores, dim=1)
                    f_adv_end_prob, f_adv_end_idx = torch.max(f_adv_end_scores, dim=1)

                    f_adv_start_prob_temp, f_adv_end_prob_temp = [], []
                    f_adv_start_idx_temp, f_adv_end_idx_temp = [], []

                    for k in range(f_adv_start_idx.size(0)):
                        if f_adv_query_seg[0, k] == 1:
                            if f_adv_start_idx[k].item() == 1:
                                f_adv_start_idx_temp.append(k)
                                f_adv_start_prob_temp.append(f_adv_start_prob[k].item())
                            if f_adv_end_idx[k].item() == 1:
                                f_adv_end_idx_temp.append(k)
                                f_adv_end_prob_temp.append(f_adv_end_prob[k].item())

                    f_adv_start_idx, f_adv_end_idx, f_adv_prob = filter_unpaired(
                        f_adv_start_prob_temp, f_adv_end_prob_temp, f_adv_start_idx_temp, f_adv_end_idx_temp,
                        imp_start=imp_start
                    )

                    for k in range(len(f_adv_start_idx)):
                        asp = [f_asp_query[0][j].item() for j in range(f_asp_start_idx[a], f_asp_end_idx[a] + 1)]
                        opi = [f_opi_query[0][j].item() for j in range(f_opi_start_idx[b], f_opi_end_idx[b] + 1)]
                        adv = [f_adv_query[0][j].item() for j in range(f_adv_start_idx[k], f_adv_end_idx[k] + 1)]

                        # 问题 + null     没有null(0,0) -> (-1, -1)   回归最原始评论中关键元素的下标
                        asp_idx = [f_asp_start_idx[a] - len(q1_asp) - 2, f_asp_end_idx[a] - len(q1_asp) - 2]
                        opi_idx = [f_opi_start_idx[b] - f_opi_query_length - 1,
                                   f_opi_end_idx[b] - f_opi_query_length - 1]
                        adv_idx = [f_adv_start_idx[k] - f_adv_query_length - 1,
                                   f_adv_end_idx[k] - f_adv_query_length - 1]

                        # 计算配对概率
                        temp_prob = f_asp_prob[a] * f_opi_prob[b] * f_adv_prob[k]

                        if asp_idx + opi_idx + adv_idx not in forward_triplet_idx_list:
                            forward_triplet_list.append([asp] + [opi] + [adv])
                            forward_triplet_prob.append(temp_prob)
                            forward_triplet_idx_list.append(asp_idx + opi_idx + adv_idx)

            # ********************Q4: Extract Adverb ********************
            b_adv_query = torch.tensor([b_adv_query_list[i]]).long()
            b_adv_query_mask = torch.tensor([b_adv_mask_list[i]]).long()
            b_adv_query_seg = torch.tensor([b_adv_seg_list[i]]).long()

            b_adv_start_scores, b_adv_end_scores = self.model(
                b_adv_query.to(device),
                b_adv_query_mask.to(device),
                b_adv_query_seg.to(device),
                0
            )

            b_adv_start_scores = F.softmax(b_adv_start_scores[0], dim=1)
            b_adv_end_scores = F.softmax(b_adv_end_scores[0], dim=1)
            b_adv_start_prob, b_adv_start_idx = torch.max(b_adv_start_scores, dim=1)
            b_adv_end_prob, b_adv_end_idx = torch.max(b_adv_end_scores, dim=1)

            b_adv_start_prob_temp, b_adv_end_prob_temp = [], []
            b_adv_start_idx_temp, b_adv_end_idx_temp = [], []

            for k in range(b_adv_start_idx.size(0)):
                if k in idx_list[i]:  # q1、q4问题长度是一样的
                    if b_adv_start_idx[k].item() == 1:
                        b_adv_start_idx_temp.append(k)
                        b_adv_start_prob_temp.append(b_adv_start_prob[k].item())
                    if b_adv_end_idx[i].item() == 1:
                        b_adv_end_idx_temp.append(k)
                        b_adv_end_prob_temp.append(b_adv_end_prob[i].item())
            b_adv_start_idx, b_adv_end_idx, b_adv_prob = filter_unpaired(
                b_adv_start_prob_temp, b_adv_end_prob_temp, b_adv_start_idx_temp, b_adv_end_idx_temp,
                imp_start=8
            )

            # ********************Q5: Adverb -> Opinion ********************
            for a in range(len(b_adv_start_idx)):
                b_opi_query = self.tokenizer.convert_tokens_to_ids(q5_adv2opi)
                for j in range(b_adv_start_idx[a], b_adv_end_idx[a] + 1):
                    b_opi_query.insert(5, b_adv_query[0][j].item())

                b_opi_query_length = len(b_opi_query)

                b_opi_query_seg = [0] * len(b_opi_query)
                imp_start = len(b_opi_query)
                b_opi_query = torch.tensor(b_opi_query).long()
                b_opi_query = torch.cat([b_opi_query, vocab_idx], -1).to(device).unsqueeze(0)
                # print('.' * 100)
                # print(f"b_opi_query: {b_opi_query}")
                b_opi_query_mask = torch.ones(b_opi_query.size(1)).float().to(device).unsqueeze(0)
                b_opi_query_seg += [1] * vocab_idx.size(0)  # 还是list
                b_opi_query_seg = torch.tensor(b_opi_query_seg).long().to(device).unsqueeze(0)

                b_opi_start_scores, b_opi_end_scores = self.model(b_opi_query, b_opi_query_mask,
                                                                  b_opi_query_seg, 0)

                b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
                b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
                b_opi_start_prob, b_opi_start_idx = torch.max(b_opi_start_scores, dim=1)
                b_opi_end_prob, b_opi_end_idx = torch.max(b_opi_end_scores, dim=1)

                b_opi_start_prob_temp, b_opi_end_prob_temp = [], []
                b_opi_start_idx_temp, b_opi_end_idx_temp = [], []

                for k in range(b_opi_start_idx.size(0)):
                    if b_opi_query_seg[0, k] == 1:
                        if b_opi_start_idx[k].item() == 1:
                            b_opi_start_idx_temp.append(k)
                            b_opi_start_prob_temp.append(b_opi_start_prob[k].item())
                        if b_opi_end_idx[k].item() == 1:
                            b_opi_end_idx_temp.append(k)
                            b_opi_end_prob_temp.append(b_opi_end_prob[k].item())

                b_opi_start_idx, b_opi_end_idx, b_opi_prob = filter_unpaired(
                    b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_idx_temp, b_opi_end_idx_temp,
                    imp_start=imp_start
                )

                for k in range(len(b_opi_start_idx)):
                    adv = [b_adv_query[0][j].item() for j in range(b_adv_start_idx[a], b_adv_end_idx[a] + 1)]
                    opi = [b_opi_query[0][j].item() for j in range(b_opi_start_idx[k], b_opi_end_idx[k] + 1)]

                    # 问题 + null     没有null(0,0) -> (-1, -1)   回归最原始评论中关键元素的下标
                    adv_idx = [b_adv_start_idx[a] - len(q4_adv) - 2, b_adv_end_idx[a] - len(q4_adv) - 2]
                    opi_idx = [b_opi_start_idx[k] - b_opi_query_length - 1,
                               b_opi_end_idx[k] - b_opi_query_length - 1]

                    # 计算配对概率
                    temp_prob = b_opi_prob[k] * b_adv_prob[a]

                    if opi_idx + adv_idx not in backward_pair_idx_list:
                        backward_pair_list.append([opi] + [adv])
                        backward_pair_prob.append(temp_prob)
                        backward_pair_idx_list.append(opi_idx + adv_idx)

                # ********************Q6: Adverb + Opinion->Aspect ********************
                # ["[CLS]", "这", "个", "副", "词", "5和", "意", "见", "-10修", "饰", "的", "方", "面", "有", "哪", "些", "？", "[SEP]"]
                for b in range(len(b_opi_start_idx)):
                    b_asp_query = self.tokenizer.convert_tokens_to_ids(q6_opi2asp)
                    # insert adverb
                    for j in range(b_adv_start_idx[a], b_adv_end_idx[a] + 1):
                        b_asp_query.insert(5, b_adv_query[0][j].item())
                    # insert opinion
                    for j in range(b_opi_start_idx[b], b_opi_end_idx[b] + 1):
                        b_asp_query.insert(-10, b_opi_query[0][j].item())

                    b_asp_query_length = len(b_asp_query)

                    b_asp_query_seg = [0] * len(b_asp_query)
                    imp_start = len(b_asp_query)
                    b_asp_query = torch.tensor(b_asp_query)
                    b_asp_query = torch.cat([b_asp_query, vocab_idx], -1).to(device).unsqueeze(0)
                    # print('.' * 100)
                    # print(f"b_asp_query: {b_asp_query}")
                    b_asp_query_mask = torch.ones(b_asp_query.size(1)).float().to(device).unsqueeze(0)
                    b_asp_query_seg += [1] * vocab_idx.size(0)
                    b_asp_query_seg = torch.tensor(b_asp_query_seg).long().to(device).unsqueeze(0)

                    b_asp_start_scores, b_asp_end_scores = self.model(b_asp_query, b_asp_query_mask, b_asp_query_seg, 0)

                    b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
                    b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
                    b_asp_start_prob, b_asp_start_idx = torch.max(b_asp_start_scores, dim=1)
                    b_asp_end_prob, b_asp_end_idx = torch.max(b_asp_end_scores, dim=1)

                    b_asp_start_prob_temp, b_asp_end_prob_temp = [], []
                    b_asp_start_idx_temp, b_asp_end_idx_temp = [], []

                    for k in range(b_asp_start_idx.size(0)):
                        if b_asp_query_seg[0, k] == 1:
                            if b_asp_start_idx[k].item() == 1:
                                b_asp_start_idx_temp.append(k)
                                b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                            if b_asp_end_idx[k].item() == 1:
                                b_asp_end_idx_temp.append(k)
                                b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

                    b_asp_start_idx, b_asp_end_idx, b_asp_prob = filter_unpaired(
                        b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_idx_temp, b_asp_end_idx_temp,
                        imp_start=imp_start
                    )

                    for k in range(len(b_asp_start_idx)):
                        adv = [b_adv_query[0][j].item() for j in range(b_adv_start_idx[a], b_adv_end_idx[a] + 1)]
                        opi = [b_opi_query[0][j].item() for j in range(b_opi_start_idx[b], b_opi_end_idx[b] + 1)]
                        asp = [b_asp_query[0][j].item() for j in range(b_asp_start_idx[k], b_asp_end_idx[k] + 1)]

                        adv_idx = [b_adv_start_idx[a] - len(q4_adv) - 2, b_adv_end_idx[a] - len(q4_adv) - 2]
                        opi_idx = [b_opi_start_idx[b] - b_opi_query_length - 1,
                                   b_opi_end_idx[b] - b_opi_query_length - 1]
                        asp_idx = [b_asp_start_idx[k] - b_asp_query_length - 1,
                                   b_asp_end_idx[k] - b_asp_query_length - 1]

                        temp_prob = b_asp_prob[k] * b_opi_prob[b] * b_adv_prob[a]
                        if asp_idx + opi_idx + adv_idx not in backward_triplet_idx_list:
                            backward_triplet_list.append([asp] + [opi] + [adv])
                            backward_triplet_prob.append(temp_prob)
                            backward_triplet_idx_list.append(asp_idx + opi_idx + adv_idx)

            # ********************Q7: Category + Q8: Sentiment********************
            if self.args.use_Forward:
                pass
            elif self.args.use_Backward:
                pass
            else:
                # combine forward and backward
                # print(f"原始forward Triplet：{forward_triplet_list}")
                # print(f"原始backward Triplet：{backward_triplet_list}")
                # print(f"原始forward Triplet 下标：{forward_triplet_idx_list}")
                # print(f"原始backward Triplet 下标{backward_triplet_idx_list}")
                final_asp_list, final_opi_list, final_adv_list, \
                    final_asp_idx_list, final_opi_idx_list, final_adv_idx_list = triplet_combine(
                        forward_triplet_list,
                        forward_triplet_prob,
                        forward_triplet_idx_list,
                        backward_triplet_list,
                        backward_triplet_prob,
                        backward_triplet_idx_list,
                        self.args.alpha,
                        self.args.beta
                    )
                # print(f"final_asp_list:{final_asp_list}")
                # print(f"final_opi_list:{final_opi_list}")
                # print(f"final_adv_list:{final_adv_list}")
                # print(f"final_asp_idx_list:{final_asp_idx_list}")
                # print(f"final_opi_idx_list:{final_opi_idx_list}")
                # print(f"final_adv_idx_list:{final_adv_idx_list}")
                # final_aspect_list: 2_dimension list vocab_id
                for a in range(len(final_asp_list)):
                    predict_opinion_num = len(final_opi_list[a])  # asp 对应的 opi数
                    category_query = self.tokenizer.convert_tokens_to_ids(q7_c)
                    sentiment_query = self.tokenizer.convert_tokens_to_ids(q8_s)
                    # insert aspect in query
                    for j in range(len(final_asp_list[a])):
                        category_query.insert(5, final_asp_list[a][j])
                        sentiment_query.insert(5, final_asp_list[a][j])
                    temp_category = category_query
                    temp_sentiment = sentiment_query
                    for b in range(predict_opinion_num):
                        predict_adverb_num = len(final_adv_list[a][b])
                        # 循环状态回溯
                        category_query = temp_category
                        sentiment_query = temp_sentiment
                        # insert opinion in query
                        for j in range(len(final_opi_list[a][b])):
                            category_query.insert(-8, final_opi_list[a][b][j])
                            sentiment_query.insert(-8, final_opi_list[a][b][j])

                        # category
                        category_query_seg = [0] * len(category_query)
                        category_query = torch.tensor(category_query).long().to(device)
                        category_query = torch.cat([category_query, vocab_idx], -1).to(device).unsqueeze(0)
                        category_query_seg += [1] * vocab_idx.size(0)
                        category_query_mask = torch.ones(category_query.size(1)).float().to(device).unsqueeze(0)
                        category_query_seg = torch.tensor(category_query_seg).long().to(device).unsqueeze(0)
                        # sentiment
                        sentiment_query_seg = [0] * len(sentiment_query)
                        sentiment_query = torch.tensor(sentiment_query).long().to(device)
                        sentiment_query = torch.cat([sentiment_query, vocab_idx], -1).to(device).unsqueeze(0)
                        sentiment_query_seg += [1] * vocab_idx.size(0)
                        sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().to(device).unsqueeze(0)
                        sentiment_query_seg = torch.tensor(sentiment_query_seg).long().to(device).unsqueeze(0)

                        # inference results of category
                        category_scores = self.model(category_query, category_query_mask, category_query_seg, 1)
                        category_scores = F.softmax(category_scores, dim=1)
                        category_predicted = torch.argmax(category_scores[0], dim=0).item()

                        # inference results of sentiment
                        sentiment_scores = self.model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 2)
                        sentiment_scores = F.softmax(sentiment_scores, dim=1)
                        sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()
                        # opinion对应的adverb
                        for c in range(predict_adverb_num):
                            # 四元组、五元组组合
                            asp, opi, adv = [], [], []
                            asp.append(final_asp_idx_list[a][0])  # asp 的start index
                            asp.append(final_asp_idx_list[a][1])  # asp 的end index
                            opi.append(final_opi_idx_list[a][b][0])
                            opi.append(final_opi_idx_list[a][b][1])
                            adv.append(final_adv_idx_list[a][b][c][0])
                            adv.append(final_adv_idx_list[a][b][c][1])

                            quadruple_predict = [asp, category_predicted, opi, sentiment_predicted]
                            quintuple_predict = [asp, category_predicted, opi, adv, sentiment_predicted]

                            if quadruple_predict not in quadruples_predict:
                                quadruples_predict.append(quadruple_predict)
                            if quintuple_predict not in quintuples_predict:
                                quintuples_predict.append(quintuple_predict)

                # else 范围内
                print_quadruples_predict = []
                print_quintuples_predict = []
                review_list = reviews[i].split(' ')

                # 每个单词可能被拆分成多个子词
                tokenized_review = list(map(self.tokenizer.tokenize, review_list))
                subword_lengths = list(map(len, tokenized_review))
                token_start_idxs = np.cumsum([0] + subword_lengths[:-1])
                tokenized2word = {}
                for k in range(len(review_list)):
                    for t in range(token_start_idxs[k], token_start_idxs[k] + subword_lengths[k]):
                        tokenized2word[t] = k
                for q in quadruples_predict:
                    if q[0] == [-1, -1]:
                        asp = 'NULL'
                    else:
                        asp = ' '.join(review_list[tokenized2word[q[0][0]]: tokenized2word[q[0][-1]] + 1])
                    if q[2] == [-1, -1]:
                        opi = 'NULL'
                    else:
                        opi = ' '.join(review_list[tokenized2word[q[2][0]]:tokenized2word[q[2][-1]] + 1])
                    category, sentiment = category_id[q[1]], sentiment_id[q[-1]]
                    print_quadruples_predict.append([asp, category, opi, sentiment])
                print(f"`{reviews[i]}` 四元组抽取结果：`{print_quadruples_predict}`")

                for q in quintuples_predict:
                    if q[0] == [-1, -1]:
                        asp = 'NULL'
                    else:
                        asp = ' '.join(review_list[tokenized2word[q[0][0]]: tokenized2word[q[0][-1]] + 1])
                    if q[2] == [-1, -1]:
                        opi = 'NULL'
                    else:
                        opi = ' '.join(review_list[tokenized2word[q[2][0]]:tokenized2word[q[2][-1]] + 1])
                    if q[3] == [-1, -1]:
                        adv = 'NULL'
                    else:
                        adv = ' '.join(review_list[tokenized2word[q[3][0]]:tokenized2word[q[3][-1]] + 1])
                    category, sentiment = category_id[q[1]], sentiment_id[q[-1]]
                    print_quintuples_predict.append([asp, category, opi, adv, sentiment])
                print(f"`{reviews[i]}` 五元组抽取结果：`{print_quintuples_predict}`")

    def get_train_loss(self, batch):
        forward_opi_nums, forward_adv_nums, backward_opi_nums, backward_asp_nums, pairs_nums = \
            batch.forward_opi_nums, batch.forward_adv_nums, batch.backward_opi_nums, batch.backward_asp_nums, \
            batch.pairs_nums
        # 获取该批次中的最长输入
        max_f_asp_len, max_f_opi_lens, max_f_adv_lens, max_b_adv_len, max_b_opi_lens, max_b_asp_lens = \
            max(batch.forward_aspect_len), \
            max([max(batch.forward_opinion_lens[b]) for b in range(self.args.train_batch_size)]), \
            max([max(batch.forward_adverb_lens[b]) for b in range(self.args.train_batch_size)]), \
            max(batch.backward_adverb_len), \
            max([max(batch.backward_opinion_lens[b]) for b in range(self.args.train_batch_size)]), \
            max([max(batch.backward_aspect_lens[b]) for b in range(self.args.train_batch_size)])
        max_sent_cate_lens = max([max(batch.sentiment_category_lens[b]) for b in range(self.args.train_batch_size)])

        # 对第二个维度进行切片   forward_asp_query 是一个二维张量，在collate_fn中被处理的
        forward_asp_query = batch.forward_asp_query[:, :max_f_asp_len]
        forward_asp_query_mask = batch.forward_asp_query_mask[:, :max_f_asp_len]
        forward_asp_query_seg = batch.forward_asp_query_seg[:, :max_f_asp_len]
        forward_asp_answer_start = batch.forward_asp_answer_start[:, :max_f_asp_len]
        forward_asp_answer_end = batch.forward_asp_answer_end[:, :max_f_asp_len]

        backward_adv_query = batch.backward_adv_query[:, :max_b_adv_len]
        backward_adv_query_mask = batch.backward_adv_query_mask[:, :max_b_adv_len]
        backward_adv_query_seg = batch.backward_adv_query_seg[:, :max_b_adv_len]
        backward_adv_answer_start = batch.backward_adv_answer_start[:, :max_b_adv_len]
        backward_adv_answer_end = batch.backward_adv_answer_end[:, :max_b_adv_len]

        forward_opi_query, forward_opi_query_mask, forward_opi_query_seg = [], [], []
        forward_opi_answer_start, forward_opi_answer_end = [], []
        forward_adv_query, forward_adv_query_mask, forward_adv_query_seg = [], [], []
        forward_adv_answer_start, forward_adv_answer_end = [], []
        backward_opi_query, backward_opi_query_mask, backward_opi_query_seg = [], [], []
        backward_opi_answer_start, backward_opi_answer_end = [], []
        backward_asp_query, backward_asp_query_mask, backward_asp_query_seg = [], [], []
        backward_asp_answer_start, backward_asp_answer_end = [], []
        category_query, category_query_mask, category_query_seg, category_answer = [], [], [], []
        sentiment_query, sentiment_query_mask, sentiment_query_seg, sentiment_answer = [], [], [], []

        for b in range(self.args.train_batch_size):
            # 每个序列的长度中还是保留了padding  max_f_opi_lens 是这个批次中序列的最长长度
            forward_opi_query.append(batch.forward_opi_query[b][:forward_opi_nums[b], :max_f_opi_lens])
            forward_opi_query_mask.append(batch.forward_opi_query_mask[b][:forward_opi_nums[b], :max_f_opi_lens])
            forward_opi_query_seg.append(batch.forward_opi_query_seg[b][:forward_opi_nums[b], :max_f_opi_lens])
            forward_opi_answer_start.append(batch.forward_opi_answer_start[b][:forward_opi_nums[b], :max_f_opi_lens])
            forward_opi_answer_end.append(batch.forward_opi_answer_end[b][:forward_opi_nums[b], :max_f_opi_lens])

            forward_adv_query.append(batch.forward_adv_query[b][:forward_adv_nums[b], :max_f_adv_lens])
            forward_adv_query_mask.append(batch.forward_adv_query_mask[b][:forward_adv_nums[b], :max_f_adv_lens])
            forward_adv_query_seg.append(batch.forward_adv_query_seg[b][:forward_adv_nums[b], :max_f_adv_lens])
            forward_adv_answer_start.append(batch.forward_adv_answer_start[b][:forward_adv_nums[b], :max_f_adv_lens])
            forward_adv_answer_end.append(batch.forward_adv_answer_end[b][:forward_adv_nums[b], :max_f_adv_lens])

            backward_opi_query.append(batch.backward_opi_query[b][:backward_opi_nums[b], :max_b_opi_lens])
            backward_opi_query_mask.append(batch.backward_opi_query_mask[b][:backward_opi_nums[b], :max_b_opi_lens])
            backward_opi_query_seg.append(batch.backward_opi_query_seg[b][:backward_opi_nums[b], :max_b_opi_lens])
            backward_opi_answer_start.append(batch.backward_opi_answer_start[b][:backward_opi_nums[b], :max_b_opi_lens])
            backward_opi_answer_end.append(batch.backward_opi_answer_end[b][:backward_opi_nums[b], :max_b_opi_lens])

            backward_asp_query.append(batch.backward_asp_query[b][:backward_asp_nums[b], :max_b_asp_lens])
            backward_asp_query_mask.append(batch.backward_asp_query_mask[b][:backward_asp_nums[b], :max_b_asp_lens])
            backward_asp_query_seg.append(batch.backward_asp_query_seg[b][:backward_asp_nums[b], :max_b_asp_lens])
            backward_asp_answer_start.append(batch.backward_asp_answer_start[b][:backward_asp_nums[b], :max_b_asp_lens])
            backward_asp_answer_end.append(batch.backward_asp_answer_end[b][:backward_asp_nums[b], :max_b_asp_lens])

            category_query.append(batch.category_query[b][:pairs_nums[b], :max_sent_cate_lens])
            category_query_mask.append(batch.category_query_mask[b][:pairs_nums[b], :max_sent_cate_lens])
            category_query_seg.append(batch.category_query_seg[b][:pairs_nums[b], :max_sent_cate_lens])
            category_answer.append(batch.category_answer[b][:pairs_nums[b]])

            sentiment_query.append(batch.sentiment_query[b][:pairs_nums[b], :max_sent_cate_lens])
            sentiment_query_mask.append(batch.sentiment_query_mask[b][:pairs_nums[b], :max_sent_cate_lens])
            sentiment_query_seg.append(batch.sentiment_query_seg[b][:pairs_nums[b], :max_sent_cate_lens])
            sentiment_answer.append(batch.sentiment_answer[b][:pairs_nums[b]])

        # 列表拼接后 将列表转为三维向量 [batch_size, sequence_len, hidden_size]
        forward_opi_query = torch.cat(forward_opi_query, dim=0)
        forward_opi_query_mask = torch.cat(forward_opi_query_mask, dim=0)
        forward_opi_query_seg = torch.cat(forward_opi_query_seg, dim=0)
        forward_opi_answer_start = torch.cat(forward_opi_answer_start, dim=0)
        forward_opi_answer_end = torch.cat(forward_opi_answer_end, dim=0)

        forward_adv_query = torch.cat(forward_adv_query, dim=0)
        forward_adv_query_mask = torch.cat(forward_adv_query_mask, dim=0)
        forward_adv_query_seg = torch.cat(forward_adv_query_seg, dim=0)
        forward_adv_answer_start = torch.cat(forward_adv_answer_start, dim=0)
        forward_adv_answer_end = torch.cat(forward_adv_answer_end, dim=0)

        backward_opi_query = torch.cat(backward_opi_query, dim=0)
        backward_opi_query_mask = torch.cat(backward_opi_query_mask, dim=0)
        backward_opi_query_seg = torch.cat(backward_opi_query_seg, dim=0)
        backward_opi_answer_start = torch.cat(backward_opi_answer_start, dim=0)
        backward_opi_answer_end = torch.cat(backward_opi_answer_end, dim=0)

        backward_asp_query = torch.cat(backward_asp_query, dim=0)
        backward_asp_query_mask = torch.cat(backward_asp_query_mask, dim=0)
        backward_asp_query_seg = torch.cat(backward_asp_query_seg, dim=0)
        backward_asp_answer_start = torch.cat(backward_asp_answer_start, dim=0)
        backward_asp_answer_end = torch.cat(backward_asp_answer_end, dim=0)

        category_query = torch.cat(category_query, dim=0)
        category_query_mask = torch.cat(category_query_mask, dim=0)
        category_query_seg = torch.cat(category_query_seg, dim=0)
        category_answer = torch.cat(category_answer, dim=0)

        sentiment_query = torch.cat(sentiment_query, dim=0)
        sentiment_query_mask = torch.cat(sentiment_query_mask, dim=0)
        sentiment_query_seg = torch.cat(sentiment_query_seg, dim=0)
        sentiment_answer = torch.cat(sentiment_answer, dim=0)
        # 防止警告
        f_asp_loss, f_opi_loss, f_adv_loss, b_asp_loss, b_opi_loss, b_adv_loss = 0, 0, 0, 0, 0, 0
        if self.args.use_Forward:
            f_asp_start_scores, f_asp_end_scores = self.model(forward_asp_query.to(self.device),
                                                              forward_asp_query_mask.to(self.device),
                                                              forward_asp_query_seg.to(self.device), 0)
            f_opi_start_scores, f_opi_end_scores = self.model(forward_opi_query.to(self.device),
                                                              forward_opi_query_mask.to(self.device),
                                                              forward_opi_query_seg.to(self.device), 0)
            f_adv_start_scores, f_adv_end_scores = self.model(forward_adv_query.to(self.device),
                                                              forward_adv_query_mask.to(self.device),
                                                              forward_adv_query_seg.to(self.device), 0)
            f_asp_loss = calculate_entity_loss(f_asp_start_scores, f_asp_end_scores,
                                               forward_asp_answer_start.to(self.device),
                                               forward_asp_answer_end.to(self.device))
            f_opi_loss = calculate_entity_loss(f_opi_start_scores, f_opi_end_scores,
                                               forward_opi_answer_start.to(self.device),
                                               forward_opi_answer_end.to(self.device))
            f_adv_loss = calculate_entity_loss(f_adv_start_scores, f_adv_end_scores,
                                               forward_adv_answer_start.to(self.device),
                                               forward_adv_answer_end.to(self.device))
        elif self.args.use_Backward:
            b_adv_start_scores, b_adv_end_scores = self.model(backward_adv_query.to(self.device),
                                                              backward_adv_query_mask.to(self.device),
                                                              backward_adv_query_seg.to(self.device), 0)
            b_opi_start_scores, b_opi_end_scores = self.model(backward_opi_query.to(self.device),
                                                              backward_opi_query_mask.to(self.device),
                                                              backward_opi_query_seg.to(self.device), 0)
            b_asp_start_scores, b_asp_end_scores = self.model(backward_asp_query.to(self.device),
                                                              backward_asp_query_mask.to(self.device),
                                                              backward_asp_query_seg.to(self.device), 0)
            b_adv_loss = calculate_entity_loss(b_adv_start_scores, b_adv_end_scores,
                                               backward_adv_answer_start.to(self.device),
                                               backward_adv_answer_end.to(self.device))
            b_opi_loss = calculate_entity_loss(b_opi_start_scores, b_opi_end_scores,
                                               backward_opi_answer_start.to(self.device),
                                               backward_opi_answer_end.to(self.device))
            b_asp_loss = calculate_entity_loss(b_asp_start_scores, b_asp_end_scores,
                                               backward_asp_answer_start.to(self.device),
                                               backward_asp_answer_end.to(self.device))
        else:
            # =============================Run Model=====================================
            # forward
            f_asp_start_scores, f_asp_end_scores = self.model(forward_asp_query.to(self.device),
                                                              forward_asp_query_mask.to(self.device),
                                                              forward_asp_query_seg.to(self.device), 0)
            f_opi_start_scores, f_opi_end_scores = self.model(forward_opi_query.to(self.device),
                                                              forward_opi_query_mask.to(self.device),
                                                              forward_opi_query_seg.to(self.device), 0)
            f_adv_start_scores, f_adv_end_scores = self.model(forward_adv_query.to(self.device),
                                                              forward_adv_query_mask.to(self.device),
                                                              forward_adv_query_seg.to(self.device), 0)
            # backward
            b_adv_start_scores, b_adv_end_scores = self.model(backward_adv_query.to(self.device),
                                                              backward_adv_query_mask.to(self.device),
                                                              backward_adv_query_seg.to(self.device), 0)
            b_opi_start_scores, b_opi_end_scores = self.model(backward_opi_query.to(self.device),
                                                              backward_opi_query_mask.to(self.device),
                                                              backward_opi_query_seg.to(self.device), 0)
            b_asp_start_scores, b_asp_end_scores = self.model(backward_asp_query.to(self.device),
                                                              backward_asp_query_mask.to(self.device),
                                                              backward_asp_query_seg.to(self.device), 0)
            # ============================Calculate Loss===================================
            # forward
            f_asp_loss = calculate_entity_loss(f_asp_start_scores, f_asp_end_scores,
                                               forward_asp_answer_start.to(self.device),
                                               forward_asp_answer_end.to(self.device))
            f_opi_loss = calculate_entity_loss(f_opi_start_scores, f_opi_end_scores,
                                               forward_opi_answer_start.to(self.device),
                                               forward_opi_answer_end.to(self.device))
            f_adv_loss = calculate_entity_loss(f_adv_start_scores, f_adv_end_scores,
                                               forward_adv_answer_start.to(self.device),
                                               forward_adv_answer_end.to(self.device))
            # backward
            b_adv_loss = calculate_entity_loss(b_adv_start_scores, b_adv_end_scores,
                                               backward_adv_answer_start.to(self.device),
                                               backward_adv_answer_end.to(self.device))
            b_opi_loss = calculate_entity_loss(b_opi_start_scores, b_opi_end_scores,
                                               backward_opi_answer_start.to(self.device),
                                               backward_opi_answer_end.to(self.device))
            b_asp_loss = calculate_entity_loss(b_asp_start_scores, b_asp_end_scores,
                                               backward_asp_answer_start.to(self.device),
                                               backward_asp_answer_end.to(self.device))

        category_scores = self.model(category_query.to(self.device), category_query_mask.to(self.device),
                                     category_query_seg.to(self.device), 1)

        sentiment_scores = self.model(sentiment_query.to(self.device), sentiment_query_mask.to(self.device),
                                      sentiment_query_seg.to(self.device), 2)
        if self.args.use_FocalLoss:
            category_loss = self.focalLoss(category_scores, category_answer.to(self.device))
            sentiment_loss = self.focalLoss(sentiment_scores, sentiment_answer.to(self.device))
        else:
            # 交叉熵loss
            category_loss = calculate_category_loss(category_scores, category_answer.to(self.device))
            sentiment_loss = calculate_sentiment_loss(sentiment_scores, sentiment_answer.to(self.device))
            # 使用对比loss
        if self.args.use_category_SCL:
            scl_category_loss = calculate_SCL_loss(category_answer.to(self.device), category_scores)
            all_category_loss = ((1 - self.args.contrastive_lr1) * category_loss
                                 + self.args.contrastive_lr1 * scl_category_loss)
        else:
            all_category_loss = category_loss
        if self.args.use_sentiment_SCL:
            scl_sentiment_loss = calculate_SCL_loss(sentiment_answer.to('cpu'), sentiment_scores)
            all_sentiment_loss = ((1 - self.args.contrastive_lr2) * sentiment_loss
                                  + self.args.contrastive_lr2 * scl_sentiment_loss)
        else:
            all_sentiment_loss = sentiment_loss

        # 正常训练loss =======================汇总=============================
        if self.args.use_Forward:
            loss_sum = (f_asp_loss + f_opi_loss + f_adv_loss) + 2 * all_category_loss + 3 * all_sentiment_loss
        elif self.args.use_Backward:
            loss_sum = (b_adv_loss + b_opi_loss + b_asp_loss) + 2 * all_category_loss + 3 * all_sentiment_loss
        else:
            loss_sum = ((f_asp_loss + f_opi_loss + f_adv_loss) +
                        (b_adv_loss + b_opi_loss + b_asp_loss) +
                        2 * all_category_loss + 3 * all_sentiment_loss)
        return loss_sum
