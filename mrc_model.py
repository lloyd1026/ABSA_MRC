import torch.nn as nn
from transformers import BertModel, RobertaModel, RobertaConfig


class MRCModel(nn.Module):
    def __init__(self, args, category_dim):
        """
        初始化函数
        :param args: 控制台传入的模型所需参数
        :param category_dim: 类别维度，表示分类任务中类别的数量
        """
        hidden_size = RobertaConfig.from_pretrained(args.model_path).hidden_size
        super(MRCModel, self).__init__()  # 调用父类的初始化函数，确保子类继承了所有父类的初始化功能

        # BERT或者Robert模型
        if hidden_size == 768:
            self._bert = BertModel.from_pretrained(args.model_path)
        else:
            if 'SentiWSP' in args.model_path or hidden_size == 1024:
                self._bert = BertModel.from_pretrained(args.model_path)  # 并不是继承，而是直接引用了BERT模型
            else:
                self._bert = RobertaModel.from_pretrained(args.model_path)

        # nn.Linear()线性层 in_features输入向量维度[hidden_size]  out_features输出向量维度[2]
        # 实体分类器(entity_classifier) 预测aspect/opinion的开始结束位置概率  输出2个预测值[no, yes]的概率
        self.classifier_start = nn.Linear(hidden_size, 2)
        self.classifier_end = nn.Linear(hidden_size, 2)
        # category classifier
        self._classifier_category = nn.Linear(hidden_size, category_dim)
        # sentiment_classifier
        self._classifier_sentiment = nn.Linear(hidden_size, 3)
        # 程度量化（分类）
        self.opinion_intensity_classifier = nn.Linear(hidden_size, 5)

    def forward(self, query_tensor, query_mask, query_seg, step, inputs_embeds=None):
        """
        @Override forward是nn.Module的一个方法，负责定义数据如何通过模型进行前向传播（即输入到输出的过程）
        :param query_tensor: 输入的序列张量：通常是输入文本的词汇索引
        :param query_mask: 注意力掩码，用于指示哪些位置是真实的输入而不是填充(padding)
        :param query_seg: 分段索引，用于区分不同的句子
        :param step: task [0:实体识别，1:类别识别, 2:情感识别 3:感官程度量化, 4:情感极性程度量化]
        :param inputs_embeds:(optional)如果提供了这个参数，模型将使用嵌入（embeddings）作为输入，而不是 query_tensor
        :return:不同任务的预测分数
        """
        # _bert方法   Bert模型的输出
        hidden_states = self._bert(query_tensor, attention_mask=query_mask,
                                   token_type_ids=query_seg, inputs_embeds=inputs_embeds)[0]

        if step == 0:  # predict entity
            out_scores_start = self.classifier_start(hidden_states)
            out_scores_end = self.classifier_end(hidden_states)
            return out_scores_start, out_scores_end
        elif step == 1:  # predict category
            cls_hidden_states = hidden_states[:, 0, :]  # 切片，3维变成2维
            cls_hidden_scores = self._classifier_category(cls_hidden_states)
            return cls_hidden_scores
        elif step == 2:  # predict sentiment
            cls_hidden_states = hidden_states[:, 0, :]
            cls_hidden_scores = self._classifier_sentiment(cls_hidden_states)
            return cls_hidden_scores
        elif step == 3:  # opinion degree quantify
            # 张量三个维度[batch_size处理样本数量, sequence_length每个样本包含的元素（单词）数量, hidden_size每个元素隐含状态的维度]
            cls_hidden_states = hidden_states[:, 0, :]  # 全要， [cls]， 全要
            cls_hidden_scores = self.opinion_intensity_classifier(cls_hidden_states)
            return cls_hidden_scores
