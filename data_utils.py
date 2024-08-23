import random

# import numpy as np
from torch.utils.data import Dataset

from labels import get_aspect_category, get_sentiment
from question_template import get_Chinese_Template
from samples import DataSample, TokenizedSample


def get_jsonl(data_path):
    with open(data_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    data_list = []
    for line in lines:
        # 将{}解析为Python字典对象
        line = eval(line)
        data_list.append(line)
    return data_list


def get_quintuples(lines, tokenizer, task):
    sentence_token_list = []
    quintuples_list = []
    for line in lines:
        sentence, labels = line['sentence'], line['labels']
        # 处理成 分割后的 单级列表，元素是str  **纯中文不需要进行子词分割?**
        sentence_token = tokenizer.tokenize(sentence)
        quin = []
        for label in labels:
            # 将标注数据中的左闭右开 转成 左闭右闭
            if label[0] == (-1, -1):
                asp = (-1, -1)
            else:
                asp = (label[0][0], label[0][1] - 1)
            cate = label[1]
            if label[2] == (-1, -1):
                opi = (-1, -1)
            else:
                opi = (label[2][0], label[2][1] - 1)
            if label[3] == (-1, -1, 1):  # 目前adv的处理并不明确
                adv = (-1, -1, 1)
            else:
                adv = (label[3][0], label[3][1] - 1, 1)
            senti = label[4]
            # [tuple1, tuple2, ...]
            quin.append((asp, cate, opi, adv, senti))
        sentence_token_list.append(sentence_token)
        quintuples_list.append(quin)
    return sentence_token_list, quintuples_list


def deal_quintuple(quintuple, category_dict, sentiment_dict):
    # 对于每条评论的数据做转化并进行清洗
    # ASTE (Aspect Sentiment Triplet Extraction):
    # AOC (Aspect-Opinion Category):
    # AOA: Aspect-Opinion-Adverbial
    aspects = []
    opinions = []
    adverbs = []
    pairs = []  # 二元组
    aste_triplets = []  # 三元组
    aoc_triplets = []  # 三元组
    aoa_triplets = []  # 三元组
    # quadruples = []  # 四元组
    quintuples = []  # 五元组

    f_quin_asp = []
    f_quin_opi = []
    f_quin_adv = []
    b_quin_asp = []
    b_quin_opi = []
    b_quin_adv = []
    quin_category = []
    quin_sentiment = []
    # 原始数据不能出现完全一致的tuple
    for q in quintuple:
        # forward
        if q[0] not in f_quin_asp:
            f_quin_asp.append(q[0])
            f_quin_opi.append([q[2]])
            f_quin_adv.append([[q[3]]])
            quin_category.append([category_dict[q[1]]])  # str -> list
            quin_sentiment.append([sentiment_dict[q[-1]]])
        else:
            idx = f_quin_asp.index(q[0])
            if q[2] not in f_quin_opi[idx]:
                f_quin_opi[idx].append(q[2])
                f_quin_adv[idx].append([q[3]])
            else:
                idy = f_quin_opi[idx].index(q[2])
                f_quin_adv[idx][idy].append(q[3])
            quin_category[idx].append(category_dict[q[1]])
            quin_sentiment[idx].append(sentiment_dict[q[-1]])
        # backward
        if q[2] not in b_quin_adv:
            b_quin_adv.append(q[3])
            b_quin_opi.append([q[2]])
            b_quin_asp.append([[q[0]]])
        else:
            idz = b_quin_adv.index(q[3])
            if q[2] not in b_quin_opi[idz]:
                b_quin_opi[idz].append(q[2])
                b_quin_asp[idz].append([q[3]])
            else:
                idy = b_quin_opi[idz].index(q[2])
                b_quin_asp[idz][idy].append(q[0])

        asp = list(q[0])
        opi = list(q[2])
        adv = list(q[3])
        pair = [asp, opi]
        aste_triplet = [asp, opi, sentiment_dict[q[-1]]]
        aoc_triplet = [asp, opi, category_dict[q[1]]]
        aoa_triplet = [asp, opi, adv]
        # quad = [asp, category_dict[q[1]], opi, sentiment_dict[q[-1]]]
        quin = [asp, category_dict[q[1]], opi, adv, sentiment_dict[q[-1]]]
        if asp not in aspects:
            aspects.append(asp)
        if opi not in opinions:
            opinions.append(opi)
        if adv not in adverbs:
            adverbs.append(adv)
        if pair not in pairs:
            pair.append(pair)
        if aste_triplet not in aste_triplets:
            aste_triplets.append(aste_triplet)
        if aoc_triplet not in aoc_triplets:
            aoc_triplets.append(aoc_triplet)
        if aoa_triplet not in aoa_triplets:
            aoa_triplets.append(aoa_triplet)
        if quin not in quintuples:
            quintuples.append(quin)
    return f_quin_asp, f_quin_opi, f_quin_adv, b_quin_asp, b_quin_opi, b_quin_adv, quin_category, quin_sentiment, \
        aspects, opinions, adverbs, pairs, aste_triplets, aoc_triplets, aoa_triplets, quintuples


class ACOSDataset(Dataset):
    def __init__(self, tokenizer, args, dataset_type):
        """

        :param tokenizer: 模型的分词器
        :param args: 参数
        :param dataset_type: 数据集类型 [train训练, dev验证, test测试]
        """
        self.counter = 0
        self.tokenizer = tokenizer
        data_path = args.data_path
        task = args.task
        data_type = args.data_type

        self.max_forward_opi_nums, self.max_backward_opi_nums = 0, 0
        self.max_forward_adv_nums, self.max_backward_asp_nums, self.max_pair_nums = 0, 0, 0

        self.max_forward_asp_len, self.max_forward_opi_len, self.max_forward_adv_len, \
            self.max_backward_asp_len, self.max_backward_opi_len, self.max_backward_adv_len, \
            self.max_pair_len = 0, 0, 0, 0, 0, 0, 0

        low_resource = args.low_resource

        # 获取数据样本
        self.data_samples = self._build_examples(data_path, dataset_type, task, data_type)
        datas_len = len(self.data_samples)
        self.datas_len = int(low_resource * datas_len)
        if dataset_type == 'train' and low_resource != 1.0:
            sample_indices = random.sample(list(range(0, datas_len)), self.datas_len)
            temps = self._build_tokenized()
            self.tokenized_samples = [temps[i] for i in sample_indices]
        else:
            self.tokenized_samples = self._build_tokenized()

    def __getitem__(self, item):
        return self.tokenized_samples[item]

    def __len__(self):
        return len(self.tokenized_samples)

    def _build_examples(self, data_path, dataset_type, task, data_type):
        data_samples = []

        # category2id sentiment2id
        category2id, sentiment2id = get_aspect_category(task, data_type)[1], get_sentiment(task)[1]
        # get raw data
        lines = get_jsonl(data_path + dataset_type + '.jsonl')
        # get quadruples and labels
        sentence_token_list, quintuple_list = get_quintuples(lines, self.tokenizer, task)

        # 中文问题模板
        q1, q2, q3, q4, q5, q6, q7, q8 = get_Chinese_Template()

        # ================================merge question and review================================
        for k in range(len(sentence_token_list)):
            text = sentence_token_list[k]
            quintuple = quintuple_list[k]
            # print(quintuple)
            f_quin_asp, f_quin_opi, f_quin_adv, b_quin_asp, b_quin_opi, b_quin_adv, quin_category, quin_sentiment, \
                aspects, opinions, adverbs, pairs, aste_triplets, aoc_triplets, aoa_triplets, quintuples = deal_quintuple(
                    quintuple, category2id, sentiment2id
                )
            forward_query_list = []
            forward_answer_list = []
            backward_query_list = []
            backward_answer_list = []

            category_query_list = []
            category_answer_list = []
            sentiment_query_list = []
            sentiment_answer_list = []

            # =========================forward=========================
            # q1:aspect query
            forward_query_list.append(q1)
            forward_asp_len = len(forward_query_list[0]) + 1 + len(text)
            if forward_asp_len > self.max_forward_asp_len:
                self.max_forward_asp_len = forward_asp_len
            # null + text 的mask 下标
            start = [0] * (len(text) + 1)
            end = [0] * (len(text) + 1)
            for ta in f_quin_asp:
                # 可以只写else
                if ta == (-1, -1):
                    start[0] = 1
                    end[0] = 1
                else:
                    start[ta[0] + 1] = 1
                    end[ta[-1] + 1] = 1

            forward_answer_list.append([start, end])

            # q2:opinion query
            for idx in range(len(f_quin_asp)):
                # 问题
                ta = f_quin_asp[idx]
                if ta == (-1, -1):
                    query = q2[0:5] + ["null"] + q2[5:]
                else:
                    query = q2[0:5] + text[ta[0]:ta[-1] + 1] + q2[5:]  # 左闭右闭
                forward_query_list.append(query)
                # 最大长度
                forward_opi_len = len(query) + 1 + len(text)
                if forward_opi_len > self.max_forward_opi_len:
                    self.max_forward_opi_len = forward_opi_len
                # 答案
                start = [0] * (len(text) + 1)
                end = [0] * (len(text) + 1)
                for to in f_quin_opi[idx]:
                    if to == (-1, -1):
                        start[0] = 1
                        end[0] = 1
                    else:
                        start[to[0] + 1] = 1
                        end[to[-1] + 1] = 1
                forward_answer_list.append([start, end])

                # q7:category query && q8:sentiment query
                for idy in range(len(f_quin_opi[idx])):
                    to = f_quin_opi[idx][idy]
                    if ta == (-1, -1) and to == (-1, -1):
                        query1 = q7[0:5] + ["null"] + q7[5:8] + ["null"] + q7[8:]
                        query2 = q8[0:5] + ["null"] + q8[5:8] + ["null"] + q8[8:]
                    elif ta == (-1, -1):
                        query1 = q7[0:5] + ["null"] + q7[5:8] + text[to[0]:to[-1] + 1] + q7[8:]
                        query2 = q8[0:5] + ["null"] + q8[5:8] + text[to[0]:to[-1] + 1] + q8[8:]
                    elif to == (-1, -1):
                        query1 = q7[0:5] + text[ta[0]:ta[-1] + 1] + q7[5:8] + ["null"] + q7[8:]
                        query2 = q8[0:5] + text[ta[0]:ta[-1] + 1] + q8[5:8] + ["null"] + q8[8:]
                    else:
                        query1 = q7[0:5] + text[ta[0]:ta[-1] + 1] + q7[5:8] + text[to[0]:to[-1] + 1] + q7[8:]
                        query2 = q8[0:5] + text[ta[0]:ta[-1] + 1] + q8[5:8] + text[to[0]:to[-1] + 1] + q8[8:]

                    # query1 和 query2 的长度是一样的
                    pair_len = len(query1) + 1 + len(text)
                    if pair_len > self.max_pair_len:
                        self.max_pair_len = pair_len
                    # 答案
                    category_query_list.append(query1)
                    category_answer_list.append(quin_category[idx][idy])
                    sentiment_query_list.append(query2)
                    sentiment_answer_list.append(quin_sentiment[idx][idy])

            # q3: adverb query
            for idx in range(len(f_quin_asp)):
                ta = f_quin_asp[idx]
                for idy in range(len(f_quin_opi[idx])):
                    to = f_quin_opi[idx][idy]
                    if ta == (-1, -1) and to == (-1, -1):
                        query = q3[0:5] + ["null"] + q3[5:7] + ["null"] + q3[7:]
                    elif ta == (-1, -1):
                        query = q3[0:5] + ["null"] + q3[5:7] + text[to[0]:to[-1] + 1] + q3[7:]
                    elif to == (-1, -1):
                        query = q3[0:5] + text[ta[0]:ta[-1] + 1] + q3[5:7] + ["null"] + q3[7:]
                    else:
                        query = q3[0:5] + text[ta[0]:ta[-1] + 1] + q3[5:7] + text[to[0]:to[-1] + 1] + q3[7:]
                    forward_query_list.append(query)

                    forward_adv_len = len(query) + 1 + len(text)
                    if forward_adv_len > self.max_forward_adv_len:
                        self.max_forward_adv_len = forward_adv_len

                    start = [0] * (len(text) + 1)
                    end = [0] * (len(text) + 1)

                    for tadv in f_quin_adv[idx][idy]:
                        if tadv == (-1, -1, 1):
                            start[0] = 1
                            end[0] = 1
                        else:
                            start[tadv[0] + 1] = 1
                            end[tadv[1] + 1] = 1
                    forward_answer_list.append([start, end])

            # =========================backward=========================
            # q4:adverb query
            backward_query_list.append(q4)
            backward_adv_len = len(backward_query_list[0]) + 1 + len(text)
            if backward_adv_len > self.max_backward_adv_len:
                self.max_backward_adv_len = backward_adv_len
            start = [0] * (len(text) + 1)
            end = [0] * (len(text) + 1)
            for tadv in b_quin_adv:
                if tadv == (-1, -1, 1):
                    start[0] = 1
                    end[0] = 1
                else:
                    start[tadv[0] + 1] = 1
                    end[tadv[1] + 1] = 1
            backward_answer_list.append([start, end])

            # q5:opinion query
            for idx in range(len(b_quin_adv)):
                tadv = b_quin_adv[idx]
                if tadv == (-1, -1, 1):
                    query = q5[0:5] + ["null"] + q5[5:]
                else:
                    query = q5[0:5] + text[tadv[0]:tadv[1] + 1] + q5[5:]
                backward_query_list.append(query)
                backward_opi_len = len(query) + 1 + len(text)
                if backward_opi_len > self.max_backward_opi_len:
                    self.max_backward_opi_len = backward_opi_len
                start = [0] * (len(text) + 1)
                end = [0] * (len(text) + 1)
                for to in b_quin_opi[idx]:
                    if to == (-1, -1):
                        start[0] = 1
                        end[0] = 1
                    else:
                        start[to[0] + 1] = 1
                        end[to[-1] + 1] = 1
                backward_answer_list.append([start, end])

            # q6: adverb query
            for idx in range(len(b_quin_adv)):
                tadv = b_quin_adv[idx]
                for idy in range(len(b_quin_opi[idx])):
                    to = b_quin_opi[idx][idy]
                    if tadv == (-1, -1, 1) and to == (-1, -1):
                        query = q3[0:5] + ["null"] + q3[5:7] + ["null"] + q3[7:]
                    elif tadv == (-1, -1, 1):
                        query = q3[0:5] + ["null"] + q3[5:7] + text[to[0]:to[-1] + 1] + q3[7:]
                    elif to == (-1, -1):
                        query = q3[0:5] + text[tadv[0]:tadv[1] + 1] + q3[5:7] + ["null"] + q3[7:]
                    else:
                        query = q3[0:5] + text[tadv[0]:tadv[1] + 1] + q3[5:7] + text[to[0]:to[-1] + 1] + q3[7:]
                    backward_query_list.append(query)

                    backward_asp_len = len(query) + 1 + len(text)
                    if backward_asp_len > self.max_backward_asp_len:
                        self.max_backward_asp_len = backward_asp_len

                    start = [0] * (len(text) + 1)
                    end = [0] * (len(text) + 1)

                    for ta in b_quin_asp[idx][idy]:
                        if ta == (-1, -1):
                            start[0] = 1
                            end[0] = 1
                        else:
                            start[ta[0] + 1] = 1
                            end[ta[-1] + 1] = 1
                    backward_answer_list.append([start, end])

            # forward (max_adverb_nums)   ===> debug 方面数就是q2数
            if len(aspects) > self.max_forward_opi_nums:
                self.max_forward_opi_nums = len(aspects)
            if len(forward_query_list) - 1 - len(aspects) > self.max_forward_adv_nums:
                self.max_forward_adv_nums = len(forward_query_list) - 1 - len(aspects)
            # backward (max_aspect_nums)
            if len(adverbs) > self.max_backward_opi_nums:
                self.max_backward_opi_nums = len(adverbs)
            if len(backward_query_list) - 1 - len(adverbs) > self.max_backward_asp_nums:
                self.max_backward_asp_nums = len(backward_query_list) - 1 - len(adverbs)
            # max_pair_nums
            if len(category_query_list) > self.max_pair_nums:
                self.max_pair_nums = len(category_query_list)

            # self.counter += 1
            # print(f"第{self.counter}条评论的forward_query")
            # # print(len(aspects))
            # for tt in forward_query_list:
            #     print(tt)

            sample = DataSample(text, aspects, opinions, adverbs, pairs, aste_triplets, aoc_triplets, aoa_triplets, quintuples,
                                forward_query_list, forward_answer_list, backward_query_list, backward_answer_list,
                                category_query_list, category_answer_list, sentiment_query_list, sentiment_answer_list)
            data_samples.append(sample)

        return data_samples

    def _build_tokenized(self):
        tokenized_samples = []
        for item in range(len(self.data_samples)):
            # ======================进行token化处理======================
            _forward_asp_query, _forward_asp_answer_start, _forward_asp_answer_end, \
                _forward_asp_query_mask, _forward_asp_query_seg = [], [], [], [], []
            _forward_opi_query, _forward_opi_answer_start, _forward_opi_answer_end, \
                _forward_opi_query_mask, _forward_opi_query_seg = [], [], [], [], []
            _forward_adv_query, _forward_adv_answer_start, _forward_adv_answer_end, \
                _forward_adv_query_mask, _forward_adv_query_seg = [], [], [], [], []

            _backward_asp_query, _backward_asp_answer_start, _backward_asp_answer_end, \
                _backward_asp_query_mask, _backward_asp_query_seg = [], [], [], [], []
            _backward_opi_query, _backward_opi_answer_start, _backward_opi_answer_end, \
                _backward_opi_query_mask, _backward_opi_query_seg = [], [], [], [], []
            _backward_adv_query, _backward_adv_answer_start, _backward_adv_answer_end, \
                _backward_adv_query_mask, _backward_adv_query_seg = [], [], [], [], []

            _category_query, _category_answer, _category_query_mask, _category_query_seg = [], [], [], []
            _sentiment_query, _sentiment_answer, _sentiment_query_mask, _sentiment_query_seg = [], [], [], []

            sample = self.data_samples[item]
            sentence_token = sample.sentence_token
            forward_query_list, forward_answer_list = sample.forward_query_list, sample.forward_answer_list
            backward_query_list, backward_answer_list = sample.backward_query_list, sample.backward_answer_list

            category_query_list = sample.category_query_list
            category_answer_list = sample.category_answer_list
            sentiment_query_list = sample.sentiment_query_list
            sentiment_answer_list = sample.sentiment_answer_list

            # forward opi query nums
            forward_opi_nums = len(sample.aspects)
            forward_adv_nums = len(forward_query_list) - forward_opi_nums - 1
            # backward asp query nums
            backward_opi_nums = len(sample.adverbs)
            backward_asp_nums = len(backward_query_list) - backward_opi_nums - 1

            # =========================Forward=========================
            # 1.aspect query
            temp_text = forward_query_list[0] + ["null"] + sentence_token
            f_asp_pad_len = self.max_forward_asp_len - len(temp_text)
            forward_aspect_len = len(temp_text)

            temp_answer_start = [-1] * len(forward_query_list[0]) + forward_answer_list[0][0]
            temp_answer_end = [-1] * len(forward_query_list[0]) + forward_answer_list[0][1]
            temp_query_seg = [0] * len(forward_query_list[0]) + [1] * (len(sentence_token) + 1)
            assert len(temp_query_seg) == len(temp_answer_start) == len(temp_answer_end) == len(temp_text)

            # padding
            # query
            _forward_asp_query = self.tokenizer.convert_tokens_to_ids(temp_text)
            _forward_asp_query.extend([0] * f_asp_pad_len)  # id + padding(0)
            # query_mask
            _forward_asp_query_mask = [1 for _ in range(len(temp_text))]
            _forward_asp_query_mask.extend([0] * f_asp_pad_len)  # valid query + null + review: 1 + padding:0
            # seg
            _forward_asp_query_seg = temp_query_seg
            _forward_asp_query_seg.extend([1] * f_asp_pad_len)  # 分句标志 query:0   null + review + padding:1
            # answer   query和padding部分都用-1占位  review区用0占位 答案用1占位
            _forward_asp_answer_start = temp_answer_start
            _forward_asp_answer_start.extend([-1] * f_asp_pad_len)
            _forward_asp_answer_end = temp_answer_end
            _forward_asp_answer_end.extend([-1] * f_asp_pad_len)

            # 2.opinion query
            forward_opinion_lens = []
            # 原有问题下，除了第一句是询问方面的，后面的所有句子都是询问意见的
            for i in range(1, len(sample.aspects) + 1):
                temp_text = forward_query_list[i] + ["null"] + sentence_token
                f_opi_pad_len = self.max_forward_opi_len - len(temp_text)
                forward_opinion_lens.append(len(temp_text))

                temp_answer_start = [-1] * len(forward_query_list[i]) + forward_answer_list[i][0]
                temp_answer_end = [-1] * len(forward_query_list[i]) + forward_answer_list[i][1]
                temp_query_seg = [0] * len(forward_query_list[i]) + [1] * (len(sentence_token) + 1)
                assert len(temp_query_seg) == len(temp_answer_start) == len(temp_answer_end) == len(temp_text)

                # query
                single_opinion_query = self.tokenizer.convert_tokens_to_ids(temp_text)
                single_opinion_query.extend([0] * f_opi_pad_len)
                # query_mask
                single_opinion_query_mask = [1 for _ in range(len(temp_text))]
                single_opinion_query_mask.extend([0] * f_opi_pad_len)
                # query_seg
                single_opinion_query_seg = temp_query_seg
                single_opinion_query_seg.extend([1] * f_opi_pad_len)
                # answer
                single_opinion_answer_start = temp_answer_start
                single_opinion_answer_start.extend([-1] * f_opi_pad_len)
                single_opinion_answer_end = temp_answer_end
                single_opinion_answer_end.extend([-1] * f_opi_pad_len)

                _forward_opi_query.append(single_opinion_query)
                _forward_opi_query_mask.append(single_opinion_query_mask)
                _forward_opi_query_seg.append(single_opinion_query_seg)
                _forward_opi_answer_start.append(single_opinion_answer_start)
                _forward_opi_answer_end.append(single_opinion_answer_end)

            # 3.adverb query
            forward_adverb_lens = []
            for i in range(len(sample.aspects) + 1, len(forward_query_list)):
                temp_text = forward_query_list[i] + ["null"] + sentence_token
                f_adv_pad_len = self.max_forward_adv_len - len(temp_text)
                forward_adverb_lens.append(len(temp_text))

                temp_answer_start = [-1] * len(forward_query_list[i]) + forward_answer_list[i][0]
                temp_answer_end = [-1] * len(forward_query_list[i]) + forward_answer_list[i][1]
                temp_query_seg = [0] * len(forward_query_list[i]) + [1] * (len(sentence_token) + 1)
                assert len(temp_query_seg) == len(temp_answer_start) == len(temp_answer_end) == len(temp_text)

                # padding
                # query
                single_adverb_query = self.tokenizer.convert_tokens_to_ids(temp_text)
                single_adverb_query.extend([0] * f_adv_pad_len)
                # query mask
                single_adverb_query_mask = [1 for _ in range(len(temp_text))]
                single_adverb_query_mask.extend([0] * f_adv_pad_len)
                # query seg
                single_adverb_query_seg = temp_query_seg
                single_adverb_query_seg.extend([1] * f_adv_pad_len)
                # query answer start && end
                single_adverb_answer_start = temp_answer_start
                single_adverb_answer_start.extend([-1] * f_adv_pad_len)
                single_adverb_answer_end = temp_answer_end
                single_adverb_answer_end.extend([-1] * f_adv_pad_len)

                _forward_adv_query.append(single_adverb_query)
                _forward_adv_query_mask.append(single_adverb_query_mask)
                _forward_adv_query_seg.append(single_adverb_query_seg)
                _forward_adv_answer_start.append(single_adverb_answer_start)
                _forward_adv_answer_end.append(single_adverb_answer_end)

            # PAD
            # **********************是否要对opi和adv的问句都进行填充？*************************************
            _forward_opi_query.extend(
                [[0 for _ in range(self.max_forward_opi_len)]] * (self.max_forward_opi_nums - forward_opi_nums))
            _forward_opi_query_mask.extend(
                [[0 for _ in range(self.max_forward_opi_len)]] * (self.max_forward_opi_nums - forward_opi_nums))
            _forward_opi_query_seg.extend(
                [[0 for _ in range(self.max_forward_opi_len)]] * (self.max_forward_opi_nums - forward_opi_nums))
            _forward_opi_answer_start.extend(
                [[-1 for _ in range(self.max_forward_opi_len)]] * (self.max_forward_opi_nums - forward_opi_nums))
            _forward_opi_answer_end.extend(
                [[-1 for _ in range(self.max_forward_opi_len)]] * (self.max_forward_opi_nums - forward_opi_nums))
            assert len(_forward_opi_query) == len(_forward_opi_query_mask) == len(_forward_opi_query_seg) == len(
                _forward_opi_answer_start) == len(_forward_opi_answer_end) == self.max_forward_opi_nums

            _forward_adv_query.extend(
                [[0 for _ in range(self.max_forward_adv_len)]] * (self.max_forward_adv_nums - forward_adv_nums))
            _forward_adv_query_mask.extend(
                [[0 for _ in range(self.max_forward_adv_len)]] * (self.max_forward_adv_nums - forward_adv_nums))
            _forward_adv_query_seg.extend(
                [[0 for _ in range(self.max_forward_adv_len)]] * (self.max_forward_adv_nums - forward_adv_nums))
            _forward_adv_answer_start.extend(
                [[-1 for _ in range(self.max_forward_adv_len)]] * (self.max_forward_adv_nums - forward_adv_nums))
            _forward_adv_answer_end.extend(
                [[-1 for _ in range(self.max_forward_adv_len)]] * (self.max_forward_adv_nums - forward_adv_nums))
            assert len(_forward_adv_query) == len(_forward_adv_query_mask) == len(_forward_adv_query_seg) == len(
                _forward_adv_answer_start) == len(_forward_adv_answer_end) == self.max_forward_adv_nums
            # =========================Backward=========================
            # 4.adverb query
            temp_text = backward_query_list[0] + ["null"] + sentence_token
            b_adv_pad_len = self.max_backward_adv_len - len(temp_text)
            backward_adverb_len = len(temp_text)

            temp_answer_start = [-1] * len(backward_query_list[0]) + backward_answer_list[0][0]
            temp_answer_end = [-1] * len(backward_query_list[0]) + backward_answer_list[0][1]
            temp_query_seg = [0] * len(backward_query_list[0]) + [1] * (len(sentence_token) + 1)
            assert len(temp_query_seg) == len(temp_answer_start) == len(temp_answer_end) == len(temp_text)

            # padding
            # query
            _backward_adv_query = self.tokenizer.convert_tokens_to_ids(temp_text)
            _backward_adv_query.extend([0] * b_adv_pad_len)
            # mask
            _backward_adv_query_mask = [1 for _ in range(len(temp_text))]
            _backward_adv_query_mask.extend([0] * b_adv_pad_len)
            # seg
            _backward_adv_query_seg = temp_query_seg
            _backward_adv_query_seg.extend([1] * b_adv_pad_len)
            # answer
            _backward_adv_answer_start = temp_answer_start
            _backward_adv_answer_start.extend([-1] * b_adv_pad_len)
            _backward_adv_answer_end = temp_answer_end
            _backward_adv_answer_end.extend([-1] * b_adv_pad_len)

            # 5. opinion query
            backward_opinion_lens = []
            for i in range(1, len(sample.adverbs) + 1):
                temp_text = backward_query_list[i] + ["null"] + sentence_token
                b_opi_pad_len = self.max_backward_opi_len - len(temp_text)
                backward_opinion_lens.append(len(temp_text))

                temp_answer_start = [-1] * len(backward_query_list[i]) + backward_answer_list[i][0]
                temp_answer_end = [-1] * len(backward_query_list[i]) + backward_answer_list[i][1]
                temp_query_seg = [0] * len(backward_query_list[i]) + [1] * (len(sentence_token) + 1)
                assert len(temp_query_seg) == len(temp_answer_start) == len(temp_answer_end) == len(temp_text)

                # query
                single_opinion_query = self.tokenizer.convert_tokens_to_ids(temp_text)
                single_opinion_query.extend([0] * b_opi_pad_len)
                # query_mask
                single_opinion_query_mask = [1 for _ in range(len(temp_text))]
                single_opinion_query_mask.extend([0] * b_opi_pad_len)
                # query_seg
                single_opinion_query_seg = temp_query_seg
                single_opinion_query_seg.extend([1] * b_opi_pad_len)
                # answer
                single_opinion_answer_start = temp_answer_start
                single_opinion_answer_start.extend([-1] * b_opi_pad_len)
                single_opinion_answer_end = temp_answer_end
                single_opinion_answer_end.extend([-1] * b_opi_pad_len)

                _backward_opi_query.append(single_opinion_query)
                _backward_opi_query_mask.append(single_opinion_query_mask)
                _backward_opi_query_seg.append(single_opinion_query_seg)
                _backward_opi_answer_start.append(single_opinion_answer_start)
                _backward_opi_answer_end.append(single_opinion_answer_end)

            # 6. Aspect query
            backward_aspect_lens = []
            for i in range(len(sample.adverbs) + 1, len(backward_query_list)):
                temp_text = backward_query_list[i] + ["null"] + sentence_token
                b_asp_pad_len = self.max_backward_asp_len - len(temp_text)
                backward_aspect_lens.append(len(temp_text))

                temp_answer_start = [-1] * len(backward_query_list[i]) + backward_answer_list[i][0]
                temp_answer_end = [-1] * len(backward_query_list[i]) + backward_answer_list[i][1]
                temp_query_seg = [0] * len(backward_query_list[i]) + [1] * (len(sentence_token) + 1)
                assert len(temp_query_seg) == len(temp_answer_start) == len(temp_answer_end) == len(temp_text)

                # padding
                # query
                single_aspect_query = self.tokenizer.convert_tokens_to_ids(temp_text)
                single_aspect_query.extend([0] * b_asp_pad_len)
                # query_mask
                single_aspect_query_mask = [1 for _ in range(len(temp_text))]
                single_aspect_query_mask.extend([0] * b_asp_pad_len)
                # query_seg
                single_aspect_query_seg = temp_query_seg
                single_aspect_query_seg.extend([1] * b_asp_pad_len)
                # answer
                single_aspect_answer_start = temp_answer_start
                single_aspect_answer_start.extend([-1] * b_asp_pad_len)
                single_aspect_answer_end = temp_answer_end
                single_aspect_answer_end.extend([-1] * b_asp_pad_len)

                _backward_asp_query.append(single_aspect_query)
                _backward_asp_query_mask.append(single_aspect_query_mask)
                _backward_asp_query_seg.append(single_aspect_query_seg)
                _backward_asp_answer_start.append(single_aspect_answer_start)
                _backward_asp_answer_end.append(single_aspect_answer_end)

            # PAD
            _backward_opi_query.extend(
                [[0 for _ in range(self.max_backward_opi_len)]] * (self.max_backward_opi_nums - backward_opi_nums))
            _backward_opi_query_mask.extend(
                [[0 for _ in range(self.max_backward_opi_len)]] * (self.max_backward_opi_nums - backward_opi_nums))
            _backward_opi_query_seg.extend(
                [[0 for _ in range(self.max_backward_opi_len)]] * (self.max_backward_opi_nums - backward_opi_nums))
            _backward_opi_answer_start.extend(
                [[-1 for _ in range(self.max_backward_opi_len)]] * (self.max_backward_opi_nums - backward_opi_nums))
            _backward_opi_answer_end.extend(
                [[-1 for _ in range(self.max_backward_opi_len)]] * (self.max_backward_opi_nums - backward_opi_nums))
            assert len(_backward_opi_query) == len(_backward_opi_query_mask) == len(_backward_opi_query_seg) == len(
                _backward_opi_answer_start) == len(_backward_opi_answer_end) == self.max_backward_opi_nums

            _backward_asp_query.extend(
                [[0 for _ in range(self.max_backward_asp_len)]] * (self.max_backward_asp_nums - backward_asp_nums))
            _backward_asp_query_mask.extend(
                [[0 for _ in range(self.max_backward_asp_len)]] * (self.max_backward_asp_nums - backward_asp_nums))
            _backward_asp_query_seg.extend(
                [[0 for _ in range(self.max_backward_asp_len)]] * (self.max_backward_asp_nums - backward_asp_nums))
            _backward_asp_answer_start.extend(
                [[-1 for _ in range(self.max_backward_asp_len)]] * (self.max_backward_asp_nums - backward_asp_nums))
            _backward_asp_answer_end.extend(
                [[-1 for _ in range(self.max_backward_asp_len)]] * (self.max_backward_asp_nums - backward_asp_nums))
            assert len(_backward_asp_query) == len(_backward_asp_query_mask) == len(_backward_asp_query_seg) == len(
                _backward_asp_answer_start) == len(_backward_asp_answer_end) == self.max_backward_asp_nums

            # category
            sentiment_category_lens = []
            assert len(category_query_list) == len(sentiment_query_list)
            for i in range(len(category_query_list)):
                question_tokenized = category_query_list[i] + ["null"] + sentence_token
                question_tokenized2 = sentiment_query_list[i] + ["null"] + sentence_token
                pad_len = self.max_pair_len - len(question_tokenized)
                pad_len2 = self.max_pair_len - len(question_tokenized2)
                assert len(question_tokenized) == len(question_tokenized2)
                sentiment_category_lens.append(len(question_tokenized))

                # mask
                question_mask = [1] * len(question_tokenized)
                question_mask2 = [1] * len(question_tokenized2)
                # segment
                question_seg = [0] * len(category_query_list[i]) + [1] * (len(sentence_token) + 1)
                question_seg2 = [0] * len(sentiment_query_list[i]) + [1] * (len(sentence_token) + 1)
                # answer
                answer = category_answer_list[i]
                answer2 = sentiment_answer_list[i]

                # padding
                # query
                question_tokenized = self.tokenizer.convert_tokens_to_ids(question_tokenized)
                question_tokenized.extend([0] * pad_len)
                question_tokenized2 = self.tokenizer.convert_tokens_to_ids(question_tokenized2)
                question_tokenized2.extend([0] * pad_len2)
                # query mask
                question_mask.extend([0 for _ in range(pad_len)])
                question_mask2.extend([0 for _ in range(pad_len2)])
                # query seg
                question_seg.extend([1] * pad_len)
                question_seg2.extend([1] * pad_len2)

                assert len(question_tokenized) == len(question_mask) == len(question_seg)
                assert len(question_tokenized2) == len(question_mask2) == len(question_seg2)

                _category_query_mask.append(question_mask)
                _category_query_seg.append(question_seg)
                _category_answer.append(answer)
                _category_query.append(question_tokenized)
                _sentiment_query_mask.append(question_mask2)
                _sentiment_query_seg.append(question_seg2)
                _sentiment_answer.append(answer2)
                _sentiment_query.append(question_tokenized2)

            # PAD: max_pair_nums 将批次处理的数据都填充到同样的长度
            _category_query.extend(
                [[0 for _ in range(self.max_pair_len)]] * (self.max_pair_nums - len(category_query_list)))
            _category_query_mask.extend(
                [[0 for _ in range(self.max_pair_len)]] * (self.max_pair_nums - len(category_query_list)))
            _category_query_seg.extend(
                [[0 for _ in range(self.max_pair_len)]] * (self.max_pair_nums - len(category_query_list)))
            _category_answer.extend([-1] * (self.max_pair_nums - len(category_query_list)))
            assert len(_category_query) == len(_category_query_mask) == len(_category_query_seg) == len(
                _category_answer) == self.max_pair_nums
            _sentiment_query.extend(
                [[0 for _ in range(self.max_pair_len)]] * (self.max_pair_nums - len(sentiment_query_list)))
            _sentiment_query_mask.extend(
                [[0 for _ in range(self.max_pair_len)]] * (self.max_pair_nums - len(sentiment_query_list)))
            _sentiment_query_seg.extend(
                [[0 for _ in range(self.max_pair_len)]] * (self.max_pair_nums - len(sentiment_query_list)))
            _sentiment_answer.extend([-1] * (self.max_pair_nums - len(sentiment_query_list)))
            assert len(_sentiment_query) == len(_sentiment_query_mask) == len(_sentiment_query_seg) == len(
                _sentiment_answer) == self.max_pair_nums

            assert len(category_query_list) == len(sentiment_query_list)

            sample = TokenizedSample(sentence_token, len(sentence_token),
                                     sample.aspects, sample.opinions, sample.adverbs, sample.pairs, sample.aste_triplets,
                                     sample.aoc_triplets, sample.aoa_triplets, sample.quintuples,
                                     _forward_asp_query, _forward_asp_answer_start, _forward_asp_answer_end,
                                     _forward_asp_query_mask, _forward_asp_query_seg,
                                     _forward_opi_query, _forward_opi_answer_start, _forward_opi_answer_end,
                                     _forward_opi_query_mask, _forward_opi_query_seg,
                                     _forward_adv_query, _forward_adv_answer_start, _forward_adv_answer_end,
                                     _forward_adv_query_mask, _forward_adv_query_seg,
                                     _backward_asp_query, _backward_asp_answer_start, _backward_asp_answer_end,
                                     _backward_asp_query_mask, _backward_asp_query_seg,
                                     _backward_opi_query, _backward_opi_answer_start, _backward_opi_answer_end,
                                     _backward_opi_query_mask, _backward_opi_query_seg,
                                     _backward_adv_query, _backward_adv_answer_start, _backward_adv_answer_end,
                                     _backward_adv_query_mask, _backward_adv_query_seg,
                                     _category_query, _category_answer, _category_query_mask, _category_query_seg,
                                     _sentiment_query, _sentiment_answer, _sentiment_query_mask, _sentiment_query_seg,
                                     forward_opi_nums, forward_adv_nums, backward_opi_nums, backward_asp_nums, len(category_query_list),
                                     forward_aspect_len, forward_opinion_lens, forward_adverb_lens,
                                     backward_adverb_len, backward_opinion_lens, backward_aspect_lens, sentiment_category_lens)
            tokenized_samples.append(sample)

        return tokenized_samples
