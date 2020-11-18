import os
import time
import argparse
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import RobertaTokenizer, RobertaModel
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nltk
from collections import namedtuple

from .utils_sent_min_kw_min import bert_cos_score_tfidf_batch as bertscore_smkm
from .utils_sent_min_kw_min import collate_tf_idf, get_bert_embedding_with_kw, greedy_cos_tf_idf
from seq_learn.data_utils import fit_list_to_block_size, get_tgt_mask

Ex_score = namedtuple('Ex_score', ['sim', 'lm', 'form'])
sim_tuple = namedtuple('sim_tuple', ['bert_score', 'bert_kw'])

class SaSrc():
    def __init__(self, tokens, bert_stats, tknzd_tokens=None):
        self.tokens = tokens
        self.bert_stats = bert_stats
        self.tknzd_tokens = tknzd_tokens

    def expand_by_idx(self, org_idxs):
        expanded_stats = []
        for stat in self.bert_stats:
            expand = torch.zeros(len(org_idxs), *stat.size()[1:]).cuda()
            idx = torch.longTensor(org_idxs)
            for _ in range(len(stat.size()) -1):
                idx = idx.unsqueeze(-1)
            idx = idx.expand(-1, *stat.size()[1:])
            expand = torch.gather(stat.cuda(), 0, idx.cuda())
            expanded_stats.append(expand.cuda())
        return expanded_stats

    def expand_orgs_by_idx(self, org_idxs):
        expanded_orgs = [self.tknzd_tokens[id] for id in org_idxs]
        return expanded_orgs

    @classmethod
    def from_org_tokens(cls, org_tokens, model, sent_layer, kw_layer, tknzr, punc_ids, stopwords, cls_id):
        src_bert_inp = collate_tf_idf(org_tokens, tknzr, punc_ids=punc_ids,
                                      device='cuda', keywords=stopwords, cls_id=cls_id, sep_id=tknzr.sep_token_id)
        src_bert_stats = get_bert_embedding_with_kw(*src_bert_inp, model, \
                                                    sent_layer, kw_layer, device='cuda', mask_head_tail=True)
        src_bert_stats = [x.cpu() for x in src_bert_stats]
        tknzd_orgs = [nltk.tokenize.word_tokenize(sent) for sent in org_tokens]
        return cls(org_tokens, src_bert_stats, tknzd_orgs)


class SAScorer():
    def __init__(self, bsz, model, tokenizer, punc_ids=None,
                 lm_model=None,
                 gpt_model=None, gpt_tknzr=None, stopwords=None,
                 verbose=True, tinit=3e-2, C=3e-4, sent_layer=8, kw_layer=1, alpha=3, beta=8,
                 device='cuda', fm_wght=0.125, lm_wght=0.625, score_mode='min',
                 use_lm_diff=False, use_cls_diff=False, if_fm_voc=None):

        self.tokenizer = tokenizer
        self.punc_ids = punc_ids

        self.model = model
        if hasattr(self.model, 'module'):
            self.model.module.roberta.encoder.output_hidden_states = True
        else:
            self.model.encoder.roberta.output_hidden_states = True


        self.lm_model = lm_model
        if lm_model:
            self.lm_dict = lm_model.dict
        else:
            self.lm_dict = None

        self.gpt_model = gpt_model
        self.gpt_tknzr = gpt_tknzr


        self.tinit = tinit
        self.C = C
        self.alpha = alpha
        self.beta = beta
        self.sent_layer = sent_layer
        self.kw_layer = kw_layer


        self.fm_wght = fm_wght
        self.lm_wght = lm_wght

        if score_mode == 'sakm':
            self.bert_score_func = bertscore_sakm
        elif score_mode == 'sasm':
            self.bert_score_func = bertscore_sasm
        elif score_mode == 'smkm':
            self.bert_score_func = bertscore_smkm
        else:
            raise ValueError('undefined score_mode')


        self.if_fm_voc = if_fm_voc
        self.last_score = 0.
        self.init_lm_score = 0.
        self.init_style_score = 0.

        self.stopwords = stopwords

        self.sampler = torch.distribution.uniform.Uniform(torch.zeros([bsz]), torch.ones([bsz]))
        self.tf_dict = defaultdict(lambda: 1.)


    def init(self, orgs):
        self.init_lm_score = self.lm_score(orgs).tolist()
        self.init_style_score = self.style_score(orgs).tolist()

    def sim_score(self, cands_bert_inp, cands_idx, verbose=False, use_idf=False):
        assert all([len(x) == len(cands_idx)] for x in cands_bert_inp)


        cands_stats = get_bert_embedding_with_kw(*cands_bert_inp, self.model,\
                                                 self.sent_layer, self.kw_layer, device='cuda', mask_head_tail=True)
        # src_stats = get_bert_embedding_with_kw(s)
        src_stats = self.sa_srcs.expand_by_idx(cands_idx)
        P, R, F1, F_sent, F_kw = greedy_cos_tf_idf(*src_stats, *cands_stats, self.alpha, self.beta)

        return P, R, F1.type(torch.double), F_sent, F_kw

    def lm_score_gpt(self, cands_org, gpt, tokenizer):
        cands = []
        for cand in cands_org:
            cands_tknzed_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cand, add_prefix_space=True))
            cand_tknzed_id = tokenizer.build_inputs_with_special_tokens(cands_tknzed_text)
            cand_tknzed_id = fit_list_to_block_size(cand_tknzed_id, 50, tokenizer.pad_token_id)
            cands.append(torch.tensor(cand_tknzed_id))
        cands = torch.stack(cands, dim=0).cuda()

        tgt_mask = [get_tgt_mask(x, tokenizer) for x in cands]
        tgt_mask = torch.stack(tgt_mask).cuda()

        total_bsz = cands.size(0)
        n_gpu = torch.cuda.device_count()
        max_hold = total_bsz - total_bsz % n_gpu

        if total_bsz < n_gpu:
            with torch.no_grad():
                logits = gpt.module(cands, labels=cands, retrive_logits=True, tgt_mask=tgt_mask)[0]
                mask = cands.ne(tokenizer.pad_token_id).float()[..., 1:]
                score = (logits.squeeze(-1) * mask).sum(-1) / mask.sum(-1)
                total_score = torch.exp(score)
        elif max_hold != 0 and max_hold < total_bsz:
            cands1, cands2 = torch.split(cands, max_hold)
            # print(f"cannot exactly divide. cands2 = {cands2.shape}")
            with torch.no_grad():
                total_score = []
                logits1 = gpt.module(cands, labels=cands1, retrive_logits=True, tgt_mask=tgt_mask)[0]
                mask1 = cands1.ne(tokenizer.pad_token_id).float()[..., 1:]
                score1 = (logits1.squeeze(-1) * mask1).sum(-1) / mask1.sum(-1)
                total_score.append(torch.exp(score1))

                logits2 = gpt.module(cands, labels=cands2, retrive_logits=True, tgt_mask=tgt_mask)[0]
                mask2 = cands2.ne(tokenizer.pad_token_id).float()[..., 1:]
                score2 = (logits2.squeeze(-1) * mask2).sum(-1) / mask2.sum(-1)
                total_score.append(torch.exp(score2))

                total_score = torch.cat(total_score, dim=0)
        else:
            with torch.no_grad():
                logits = gpt.module(cands, labels=cands, retrive_logits=True, tgt_mask=tgt_mask)[0]
                mask = cands.ne(tokenizer.pad_token_id).float()[..., 1:]
                score = (logits.squeeze(-1) * mask).sum(-1) / mask.sum(-1)
                total_score = torch.exp(score)
        assert total_bsz == total_score.size(0)

        return total_score.type(torch.double) * 100

    def lm_score(self, cands):
        tensor, penal = sents2tensor(self.lm_model.dict, cands)
        score = lm.eval_batch(self.lm_model, tensor,
                              self.lm_model.dict['<blank>'], self.lm_model.dict['<unk>'])
        # penal = torch.where(has_repeat_tok.eq(1), torch.full_like(score, 1e-10), torch.ones_like(score))
        score = score / penal.type(torch.float32).cuda()
        return score.type(torch.double)
    
    def diverse(self, cands, cands_idx):
        div = []
        refs = self.sa_srcs.expand_by_idx(cands_idx)
        cands = [nltk.tokenize.word_tokenize(cand) for cand in cands]
        for ref, hyp in zip(refs, cands):
            score = nltk.translate.bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1
            )
            div.append(1. - score)
        return torch.tensor(div).type(torch.double).cuda()

    def style_score(self, cands_id, cands_mask):
        inp_type = torch.full_like(cands_id, 1)
        score = self.model('cls', cands_id, attention_mask=cands_mask, token_type_ids=inp_type)[0]
        score = F.softmax(score, dim=-1)[:, 1]
        return score.type(torch.double)


    def eval(self, cands, cands_idx, use_idf=False):
        """

        :param cands:
        :param refs:
        :return: cand_3d_score: list len = bsz*k or nkw
        """
        # lm score
        lm_prob = self.lm_score_gpt(cands, self.gpt_model, self.gpt_tknzr)

        # sim score
        cand_bert_inp = collate_tf_idf(cands, self.tokenizer, self.tf_dict, self.punc_ids,
                                       device='cuda', mask_head_tail=False, keywords=self.stopwords,
                                       cls_id=self.tokenizer.cls_token_id, sep_id=self.tokenizer.sep_token_id)

        _, _, sim, sim_bert, sim_kw = self.sim_score(cand_bert_inp, cands_idx, verbose=False, use_idf=use_idf)

        # style score
        formal = self.style_score(cand_bert_inp[0], cand_bert_inp[-1])

        sim_l = sim.tolist()
        # div_l = div.tolist()
        formal_l = formal.tolist()
        lm_prob_l = lm_prob.tolist()
        cand_3d_score = [Ex_score(sim_l[i], lm_prob_l[i], formal_l[i]) for i in range(sim.shape[0])]

        sim_bert = sim_bert.tolist()
        sim_kw = sim_kw.tolist()
        similar_tuple = [sim_tuple(sim_bert[i], sim_kw[i]) for i in range(sim.shape[0])]

        return sim * (formal ** self.fm_wght) * (lm_prob ** self.lm_wght)

    def anneal(self, step, cands, cands_score, verbose=False):

        cands_score_tensor = torch.tensor(cands_score).cuda()
        if step == 0:
            self.last_score = torch.zeros_like(cands_score_tensor)

        acc_probs = torch.exp(
            (cands_score_tensor - self.last_score) / (self.tinit - self.C * step)
        )
        acc_probs = acc_probs.clamp(0., 1.)

        thld = torch.rand(acc_probs.size(0)).cuda().type(torch.double)
        self.last_score = torch.where(acc_probs.ge(thld), cands_score_tensor, self.last_score)

        return acc_probs.ge(thld).type(torch.float32), acc_probs
