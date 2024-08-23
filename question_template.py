def get_English_Template():
    English_Forward_Q1 = ["[CLS]", "What", "aspects", "?", "[SEP]"]
    English_Backward_Q1 = ["[CLS]", "What", "opinions", "?", "[SEP]"]
    English_Forward_Q2 = ["[CLS]", "What", "opinions", "for", "the", "aspect", "?", "[SEP]"]
    English_Backward_Q2 = ["[CLS]", "What", "aspects", "for", "the", "opinion", "?", "[SEP]"]
    English_Q3 = ["[CLS]", "What", "category", "for", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]
    English_Q4 = ["[CLS]", "What", "sentiment", "for", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]

    return English_Forward_Q1, English_Backward_Q1, English_Forward_Q2, English_Backward_Q2, English_Q3, English_Q4


def get_Chinese_Template():
    # *******Extraction*******
    # 1.aspect Extraction
    forward_q1 = ["[CLS]", "方", "面", "有", "哪", "些", "？", "[SEP]"]
    # 2.aspect -> opinion(adj)
    forward_q2 = ["[CLS]", "这", "个", "方", "面", "的", "意", "见", "有", "哪", "些", "？", "[SEP]"]
    # 3. opinion -> adverb
    forward_q3 = ["[CLS]", "这", "个", "方", "面", "意", "见", "的", "副", "词", "有", "哪", "些", "？", "[SEP]"]
    # 4. Adverb Extraction
    backward_q1 = ["[CLS]", "副", "词", "有", "哪", "些", "？", "[SEP]"]
    # 5. adverb -> opinion
    backward_q2 = ["[CLS]", "这", "个", "副", "词", "修", "饰", "的", "意", "见", "有", "哪", "些", "？", "[SEP]"]
    # 6. opinion(adj) -> aspect
    backward_q3 = ["[CLS]", "这", "个", "副", "词", "和", "意", "见", "修", "饰", "的", "方", "面", "有", "哪", "些", "？", "[SEP]"]

    # *******Inference*******
    # 7.(a, o) --(model inference) ==> category
    q_classify_category = ["[CLS]", "这", "个", "方", "面", "和", "意", "见", "的", "类", "别", "是", "什", "么", "？", "[SEP]"]
    # 8.text --(model inference) ==> sentiment
    q_classify_sentiment= ["[CLS]", "这", "个", "方", "面", "和", "意", "见", "的", "情", "感", "是", "什", "么", "？", "[SEP]"]
    # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    # 9. quantify intensity of opinion
    # s4_q = ["CLS", "这", "个", "副", "词", "的", "量", "化", "是", "多", "少", "？", "[SEP]"]

    return forward_q1, forward_q2, forward_q3, backward_q1, backward_q2, backward_q3, q_classify_category, q_classify_sentiment
