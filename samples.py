class DataSample(object):
    def __init__(self,
                 sentence_token,
                 aspects, opinions, adverbs, pairs, aste_triplets, aoc_triplets, aoa_triplets, quintuples,
                 forward_query_list, forward_answer_list,
                 backward_query_list, backward_answer_list,
                 category_query_list, category_answer_list,
                 sentiment_query_list, sentiment_answer_list):
        self.sentence_token = sentence_token
        self.aspects = aspects
        self.opinions = opinions
        self.adverbs = adverbs
        self.pairs = pairs
        self.aste_triplets = aste_triplets
        self.aoc_triplets = aoc_triplets
        self.aoa_triplets = aoa_triplets
        self.quintuples = quintuples

        self.forward_query_list = forward_query_list
        self.forward_answer_list = forward_answer_list
        self.backward_query_list = backward_query_list
        self.backward_answer_list = backward_answer_list

        self.category_query_list = category_query_list
        self.category_answer_list = category_answer_list
        self.sentiment_query_list = sentiment_query_list
        self.sentiment_answer_list = sentiment_answer_list


class TokenizedSample(object):
    def __init__(self,
                 sentence_token, sentence_len,
                 aspects, opinions, adverbs, pairs, aste_triplets, aoc_triplets, aoa_triplets, quintuples,
                 _forward_asp_query, _forward_asp_answer_start, _forward_asp_answer_end, _forward_asp_query_mask, _forward_asp_query_seg,
                 _forward_opi_query, _forward_opi_answer_start, _forward_opi_answer_end, _forward_opi_query_mask, _forward_opi_query_seg,
                 _forward_adv_query, _forward_adv_answer_start, _forward_adv_answer_end, _forward_adv_query_mask, _forward_adv_query_seg,
                 _backward_asp_query, _backward_asp_answer_start, _backward_asp_answer_end, _backward_asp_query_mask, _backward_asp_query_seg,
                 _backward_opi_query, _backward_opi_answer_start, _backward_opi_answer_end, _backward_opi_query_mask, _backward_opi_query_seg,
                 _backward_adv_query, _backward_adv_answer_start, _backward_adv_answer_end, _backward_adv_query_mask, _backward_adv_query_seg,
                 _category_query, _category_answer, _category_query_mask, _category_query_seg,
                 _sentiment_query, _sentiment_answer, _sentiment_query_mask, _sentiment_query_seg,
                 _forward_opi_nums, _forward_adv_nums, _backward_opi_nums, _backward_asp_nums, _pairs_nums,
                 forward_aspect_len, forward_opinion_lens, forward_adverb_lens,
                 backward_adverb_len, backward_opinion_lens, backward_aspect_lens, sentiment_category_lens):
        self.sentence_token = sentence_token
        self.sentence_len = sentence_len

        self.aspects = aspects
        self.opinions = opinions
        self.adverbs = adverbs
        self.pairs = pairs
        self.aste_triplets = aste_triplets
        self.aoc_triplets = aoc_triplets
        self.aoa_triplets = aoa_triplets
        self.quintuples = quintuples

        self.forward_asp_query = _forward_asp_query
        self.forward_asp_answer_start = _forward_asp_answer_start
        self.forward_asp_answer_end = _forward_asp_answer_end
        self.forward_opi_query = _forward_opi_query
        self.forward_opi_answer_start = _forward_opi_answer_start
        self.forward_opi_answer_end = _forward_opi_answer_end
        self.forward_adv_query = _forward_adv_query
        self.forward_adv_answer_start = _forward_adv_answer_start
        self.forward_adv_answer_end = _forward_adv_answer_end

        self.forward_asp_query_mask = _forward_asp_query_mask
        self.forward_asp_query_seg = _forward_asp_query_seg
        self.forward_opi_query_mask = _forward_opi_query_mask
        self.forward_opi_query_seg = _forward_opi_query_seg
        self.forward_adv_query_mask = _forward_adv_query_mask
        self.forward_adv_query_seg = _forward_adv_query_seg

        self.backward_asp_query = _backward_asp_query
        self.backward_asp_answer_start = _backward_asp_answer_start
        self.backward_asp_answer_end = _backward_asp_answer_end
        self.backward_opi_query = _backward_opi_query
        self.backward_opi_answer_start = _backward_opi_answer_start
        self.backward_opi_answer_end = _backward_opi_answer_end
        self.backward_adv_query = _backward_adv_query
        self.backward_adv_answer_start = _backward_adv_answer_start
        self.backward_adv_answer_end = _backward_adv_answer_end

        self.backward_asp_query_mask = _backward_asp_query_mask
        self.backward_asp_query_seg = _backward_asp_query_seg
        self.backward_opi_query_mask = _backward_opi_query_mask
        self.backward_opi_query_seg = _backward_opi_query_seg
        self.backward_adv_query_mask = _backward_adv_query_mask
        self.backward_adv_query_seg = _backward_adv_query_seg

        self.category_query = _category_query
        self.category_answer = _category_answer
        self.category_query_mask = _category_query_mask
        self.category_query_seg = _category_query_seg

        self.sentiment_query = _sentiment_query
        self.sentiment_answer = _sentiment_answer
        self.sentiment_query_mask = _sentiment_query_mask
        self.sentiment_query_seg = _sentiment_query_seg

        self.forward_opi_nums = _forward_opi_nums
        self.forward_adv_nums = _forward_adv_nums
        self.backward_opi_nums = _backward_opi_nums
        self.backward_asp_nums = _backward_asp_nums
        self.pairs_nums = _pairs_nums

        self.forward_aspect_len = forward_aspect_len
        self.forward_opinion_lens = forward_opinion_lens
        self.forward_adverb_lens = forward_adverb_lens
        self.backward_adverb_len = backward_adverb_len
        self.backward_opinion_lens = backward_opinion_lens
        self.backward_aspect_lens = backward_aspect_lens
        self.sentiment_category_lens = sentiment_category_lens
