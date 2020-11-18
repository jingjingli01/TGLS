import os
import time
import argparse
import torch
from collections import defaultdict, namedtuple
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
import torch.nn.functional as F
import numpy as np
from rake_nltk import Rake
from .utils_sent_min_kw_min import padding, collate_tf_no_tokenize, merge_sub_tokens, ids2sents


def expand_times(arr, times):
    ''' expand arr[i] by times[i]'''
    arr_expand = []
    for a, t in zip(arr, times):
        arr_expand += [a] * t
    return arr_expand

class Surfer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        model.eval()

    def forward(self, K, cand_inps, cand_mask, ins_pos, mask_idx, pad_idx, do_ins=True):
        bsz, seqlen = cand_inps.size()

        t = 0

        with torch.no_grad():
            editable = cand_mask.float()
            while editable.eq(1).any():
                if do_ins:
                    dropout_mask = None
                else:
                    dropout_mask = editable.cuda()
                attn_mask = cand_inps.ne(pad_idx).type(torch.float)
                inp_type = torch.full_like(cand_inps, 0)
                attn_mask = cand_inps.ne(pad_idx).float()
                inp_type = torch.full_like(cand_inps, 0)

                if t == 0:
                    outs = self.model('maskedlm', cand_inps, 
                                token_type_ids=inp_type, 
                                attention_mask=attn_mask, 
                                dropout_mask=dropout_mask)[0]
                    outs = outs.transpose(1, 2).contiguous().view(bsz, -1)
                    outs = torch.where(editable.repeat(1, 50266).eq(1).cuda(),
                                        outs, torch.full_like(outs, outs.min()))
                    ins_probs, cand_words = torch.topk(outs, K, dim=-1)
                    del outs, ins_pos

                    edit_pos_t = cand_words % seqlen
                    cand_words = cand_words // seqlen
                    cand_inps = cand_inps.repeat(K, 1)
                    editable = editable.repeat(K, 1).cuda()
                    edit_pos_t = editable.view(-1, 1)
                    cand_words = cand_words.view(-1, 1)
                else:
                    cand_words_all = []
                    outs = self.model('maskedlm', cand_inps, 
                                token_type_ids=inp_type, 
                                attention_mask=attn_mask, 
                                dropout_mask=dropout_mask)[0]
                    outs = outs.transpose(1, 2).contiguous().view(bsz, -1)
                    outs = torch.where(editable.repeat(1, 50266).eq(1).cuda(),
                                        outs, torch.full_like(outs, outs.min()))
                    ins_probs, cand_words = torch.topk(outs, 1, dim=-1)
                    edit_pos_t = cand_words % seqlen
                    cand_words = cand_words // seqlen

                    del outs, ins_probs, editable
                t += 1
                assert cand_words.ne(mask_idx).all()
                new_cand_inps = cand_inps.scatter(1, edit_pos_t)

                cand_inps = torch.where(editable.eq(1), new_cand_inps, cand_inps)
                del new_cand_inps
                edit_pos_t = edit_pos_t.view(-1, 1)
                editable = editable.scatter(1, edit_pos_t.cuda(), 
                                                torch.zeros_like(edit_pos_t).float().cuda())
        return cand_inps

class SASearcher():
    def __init__(self, simulator, rbt_model, rbt_tknzr, k=10, do_copy=False, formal_mask_file=None,
                device='cuda', temp=8., rand_pos=True, if_fm_voc=None, nplh=2):
        self.simulator = simulator
        self.tokenizer = rbt_tknzr
        self.surfer = Surfer(rbt_model)

        self.K = k
        self.temp = temp
        self.do_copy = do_copy
        self.rand_pos = rand_pos
        self.if_fm_voc = if_fm_voc

        self.set_nplh(nplh)
        self.kw_extractor = Rake()
        self.mask_idx = self.tokenizer.convert_tokens_to_ids(['<lm-mask>'])[0]
        self.pad_idx = self.tokenizer.convert_tokens_to_ids(['<pad>'])[0]
        self.temp = temp

    def set_nplh(self, n):
        self.nplh = n

    def plhs(self):
        return self.nplh * ['<lm-mask>']

    def accept_subtok(self):
        return len(self.plhs()) > 1

    def make_edit_pos(self, orgs, rand=True, npls=1):
        if rand:
            org_lens = torch.tensor([len(org.split()) for org in orgs]).type(torch.float32)
            pos = (torch.rand(len(org_lens)) * (org_lens)).clamp(0, 100)
            pos = pos.type(torch.long).tolist()
        else:
            pos = []
            for org in orgs:
                logits = torch.tensor(
                    [int(self.if_fm_voc[0][x]) / (int(self.if_fm_voc[1][x]) + 1.) for x in org.split()])
                sampler = torch.distribution.categorical.Categorical(logits=logits)
                pos.append(sampler.sample().tolist())
        return pos

    def create_ins_inps_with_keywords(self, orgs, ins_pos, keywords):
        inps = []
        ins_pos_subtok = []
        masks = []
        targets = []

        for b, (org, pos) in enumerate(zip(orgs, ins_pos)):
            for which, kw in enumerate(keywords[b]):
                inp = org.split().copy()
                inp = inp[:pos] + kw.split(' ') + inp[pos:]
                inps.append(' '.join(inp))
        return inps


    def create_ins_inps(self, orgs, ins_pos, newtokens=None):
        inps = []
        ins_pos_subtok = []
        masks = []

        for b, (org, pos) in enumerate(zip(orgs, ins_pos)):
            # for plh in self.plhs:
            inp = org.split().copy()
            left_seg = self.tokenizer.tokenize(' '.join(inp[:pos]), add_prefix_space=True)
            rght_seg = self.tokenizer.tokenize(' '.join(inp[pos:]), add_prefix_space=True)
            inp = left_seg + self.plhs() + rght_seg
            inps.append(inp)

            pos_sub = len(left_seg)
            ins_pos_subtok.append(list(range(pos_sub, pos_sub + len(self.plhs()))))

            mask = [0] * len(left_seg) + [1] * len(self.plhs()) + [0] * len(rght_seg)
            assert len(mask) == len(inp)
            masks.append(mask)

        ins_pos_subtok = torch.tensor(ins_pos_subtok)
        assert ins_pos_subtok.size(0) == len(orgs)

        Input = namedtuple('Input', 'inps pos mask')
        return Input(inps, ins_pos_subtok, masks)

    def create_sub_inps(self, orgs, ins_pos, newtokens=None):
        inps = []
        ins_pos_subtok = []
        masks = []
        mtokens = [self.tokenizer.tokenize(inp.split()[pos]), add_prefix_space=True\
                    for inp, pos in zip(orgs, ins_pos)]
        for b, (org, pos, masked_token) in enumerate(zip(orgs, ins_pos, mtokens)):
            # for plh in self.plhs:
            inp = org.split().copy()
            left_seg = self.tokenizer.tokenize(' '.join(inp[:pos]), add_prefix_space=True)
            rght_seg = self.tokenizer.tokenize(' '.join(inp[pos + 1:]), add_prefix_space=True)
            inp = left_seg + masked_token + rght_seg
            inps.append(inp)

            pos_sub = len(left_seg)
            ins_pos_subtok.append(list(range(1+pos_sub, 1+pos_sub + len(masked_token))))

            mask = [0] * len(left_seg) + [1] * len(masked_token) + [0] * len(rght_seg)
            assert len(mask) == len(inp)
            masks.append(mask)

        ins_pos_subtok = torch.tensor(ins_pos_subtok)
        assert ins_pos_subtok.size(0) == len(orgs)

        Input = namedtuple('Input', 'inps pos mask')
        return Input(inps, ins_pos_subtok, masks)

    def cal_ins_score(self, bert_inp, kw_inps=None, kw_cands_idx=None,
                      verbose=False, do_ins=True, keywords=None):
        cand_inps_sents, cand_score, cand_2d_score, cand_sim_tuple = self.retrieve_cal_cand_score(
            bert_inp.inps, bert_inp.pos, bert_inp.mask, verbose=False, do_ins=do_ins)

        bsz = len(bert_inp.inps)

        cand_score = cand_score.view(self.K, bsz).transpose(1, 0)

        if kw_inps:
            cand_kw_score, cand_kw_2d_score, cand_kw_sim_tuple = self.simulator.eval(kw_inps, kw_cands_idx)

            cand_inps_sents += kw_inps
            cand_2d_score += cand_kw_2d_score
            cand_sim_tuple += cand_kw_sim_tuple

            cand_kw_score = cand_kw_score.unsqueeze(0).repeat(bsz, 1)

            last_start = 0
            kw_out_mask = torch.zeros_like(cand_kw_score).cuda()
            for b, kws in enumerate(keywords):
                stt = last_start
                end = last_start + len(kws)
                kw_out_mask[b, stt:end] = 1.
                last_start += len(kws)

            cand_kw_score = cand_kw_score * kw_out_mask

            assert not (torch.isinf(cand_score).any()) \
                   or (torch.isinf(cand_kw_score).any())

            cand_score = torch.cat([cand_score, cand_kw_score], dim=-1)  # bsz x (K+nkw)

        return cand_inps_sents, cand_score, cand_2d_score, cand_sim_tuple

    def retrieve_cal_cand_score(self, inps, ins_pos, ins_mask, verbose=False, do_ins=True, orgs=None):

        cand_inps, _, cand_lens, _ = collate_tf_no_tokenize(
            inps, self.tokenizer.convert_tokens_to_ids, tf_dict=None)
        bsz, seqlen = cand_inps.size()


        cand_mask, _, _ = padding(ins_mask, 0, dtype=torch.long)
        cand_mask = torch.cat([
            torch.zeros(bsz, 1).long(),
            cand_mask,
            torch.zeros(bsz, 1).long()], 1)
        cand_mask = cand_mask.byte()

        K = self.K
        cand_inps = self.surfer(K, cand_inps, cand_mask, ins_pos, self.mask_idx, self.pad_idx, do_ins=do_ins)
        cand_inps_toks = ids2sents(cand_inps.view(-1, seqlen), self.tokenizer.convert_ids_to_tokens)
        del cand_inps

        with torch.no_grad():

            cand_inps_sents = [self.tokenizer.convert_tokens_to_string(x).lstrip() for x in cand_inps_toks]

            cands_idx = []
            for _ in range(K):
                cands_idx += list(range(bsz))
            assert len(cand_inps_sents) == len(cands_idx)
            score, ex_score, sim_tuple = self.simulator.eval(cand_inps_sents, cands_idx)


        return cand_inps_sents, score, ex_score, sim_tuple

    def create_del_inps(self, orgs):
        inps = []
        for i, org in enumerate(zip(orgs)):
            tknzed_org = self.tokenizer.tokenize(org, add_prefix_space=True)
            lens = len(tknzed_org)
            pos = int((torch.rand(lens) * lens).clamp(0, 100).tolist()[0])
            if len(tknzed_org) <= 3:
                print('db: no={}, inp={}, current (org, pos)=({}, {})'.format(i, org, org, pos))
                print('db: ALL orgs:{}, del_pos:{}'.format(orgs, del_pos))
            if len(tknzed_org) > 1:
                inp = tknzed_org[:pos] + tknzed_org[pos + 1:]
                inps.append(self.tokenizer.convert_tokens_to_string(inp))
            else:
                inps.append(org)
        return inps

    def cal_del_probs(self, inps, verbose=False):
        bsz = len(inps)
        cand_idx = list(range(bsz))
        score, inps_2d_score, sim_tuple = self.simulator.eval(inps, cand_idx)
        # bsz x 1
        return score.unsqueeze(dim=1), inps_2d_score, sim_tuple


    def create_rule_cands_refs(self, orgs, sub_dicts):
        inps = []
        cands_idx = []
        score_map = []
        for i, org in enumerate(orgs):
            org_ = ' ' + org + ' '
            for k, v in sub_dicts[i].items():
                k = ' ' + k + ' '
                if k in org_:
                    v = ' ' + v + ' '
                    inp = org_.replace(k, v)
                    inps.append(inp.lstrip().rstrip())
                    cands_idx.append(i)
                    score_map[i]
        return inps, cands_idx, torch.tensor(score_map).unsqueeze(0)

    def create_rule_score(self, cands, cands_idx, orgs=None):
        score, _ ,_ = self.simulator.eval(cands, cands_idx, orgs=orgs)
        return score.unsqueeze(dim=0)

    def rollout(self, case_pos=None, orgs=None,keywords=None, sub_dicts=None):

        if all([len(x) == 0 for x in keywords]): self.do_copy = False


        if not isinstance(orgs, list):
            orgs = list(orgs.values)
        bsz = len(orgs)
        ins_pos = self.make_edit_pos(orgs, rand=True)
        sub_pos = self.make_edit_pos(orgs, rand=self.rand_pos)

        # for insert
        bert_inp = self.create_ins_inps(orgs, ins_pos)  # masked input for insert
        if self.do_copy:
            kw_cands_idx = []  # ref sents for each kw
            for b, kws in enumerate(keywords):
                kw_cands_idx += [b] * len(kws)
            kw_inps = self.create_ins_inps_with_keywords(orgs, ins_pos, keywords)
            ins_cands, ins_score, ins_2d_score, ins_sim_tuple = self.cal_ins_score(bert_inp,
                            kw_inps=kw_inps, kw_cands_idx=kw_cands_idx, verbose=False, keywords=keywords)


        else:
            ins_cands, ins_score, ins_2d_score, ins_sim_tuple = self.cal_ins_score(bert_inp,
                                 kw_inps=None,kw_cands_idx=None, verbose=False, keywords=keywords)

        # re_ins_cands = [None] * bsz
        # re_2d_score = [None] * bsz
        # for i in range(bsz):
        #     re_ins_cands[i] = [ins_cands[k] for k in range(i, bsz * self.K, bsz)]
        #     re_2d_score[i] = [ins_2d_score[k] for k in range(i, bsz * self.K, bsz)]
        # for item in re_ins_cands:
        #     item.extend(ins_cands[bsz * self.K:])
        # ins_score_l = ins_score.tolist()
        # combined = [list(zip(re_ins_cands[id], ins_score_l[id], re_2d_score[id])) for id in range(bsz)]

        # for substitute
        bert_inp = self.create_sub_inps(orgs, sub_pos)
        sub_cands, sub_score, sub_2d_score, sub_sim_tuple = self.cal_ins_score(bert_inp,
                                                                               kw_inps=None, kw_cands_idx=None,
                                                                               verbose=False, do_ins=False, keywords=keywords)

        nkws = ins_score.size(1) - sub_score.size(1)
        if nkws > 0:
            sub_score = torch.cat([sub_score,
                                   torch.zeros(ins_score.size(0), nkws).cuda().type(torch.double)], dim=1)


        del_inps = self.create_del_inps(orgs)
        del_score, del_2d_score, del_sim_tuple = self.cal_del_probs(del_inps)


        cand_score = torch.cat([ins_score, sub_score, del_score], dim=1)  # bsz x (K+nkw + k+nkw +1)

        if sub_dicts:
            rule_cands, rule_cands_idx, rule_score_map = self.create_rule_cands_refs(orgs, sub_dicts)
        else:
            rule_cands = {}

        nrule = len(rule_cands)

        if nrule > 0:
            rule_score_flat = self.cal_rule_score(rule_cands, rule_cands_idx, cands_idx=rule_cands_idx)
            rule_score = torch.zeros(bsz, nrule).type(torch.double).cuda()

            for b, (score, idx) in enumerate(zip(rule_score_flat.squeeze().tolist(), rule_cands_idx)):
                rule_score[idx, b] = score
            rule_score = rule_score.type(torch.double)
            cand_score = torch.cat([cand_score, rule_score], dim=1)


        cand_probs = (cand_score ** self.temp).clamp(1e-300, 1e5)
        cand_probs = cand_probs / cand_probs.sum(-1, keepdim=True)

        sampler = torch.distributions.categorical.Categorical(probs=cand_probs)
        sampled_cands_idx = sampler.sample()  # bsz


        ins_cand_idx_end = self.K + len(kw_cands_idx) if self.do_copy else self.K
        sub_cand_idx_end = (self.K + len(kw_cands_idx)) * 2 if self.do_copy else self.K * 2

        cands_sampled = []
        cands_sampled_probs = []
        cands_sampled_score = []
        cands_sampled_3d = []
        cands_sampled_sim_tuple = []
        cand_probs = cand_probs.cpu()

        acc_rule = 0

        for b, kidx in enumerate(sampled_cands_idx.cpu()):
            cands_sampled_probs.append(cand_probs[b, kidx])
            cands_sampled_score.append(cand_score[b, kidx])
            if kidx < ins_cand_idx_end:  # ins
                if kidx < self.K:  # vocab
                    cand = ins_cands[kidx * bsz + b]
                    cand_3d = ins_2d_score[kidx * bsz + b]
                    cand_sim = ins_sim_tuple[kidx * bsz + b]
                else:  # kw
                    cand = ins_cands[self.K * bsz + (kidx - self.K)]
                    cand_3d = ins_2d_score[self.K * bsz + (kidx - self.K)]
                    cand_sim = ins_sim_tuple[self.K * bsz + (kidx - self.K)]
            elif kidx >= ins_cand_idx_end and kidx < sub_cand_idx_end:  # sub
                if kidx < ins_cand_idx_end + self.K:  # vocab
                    cand = sub_cands[(kidx - ins_cand_idx_end) * bsz + b]
                    cand_3d = sub_2d_score[(kidx - ins_cand_idx_end) * bsz + b]
                    cand_sim = sub_sim_tuple[(kidx - ins_cand_idx_end) * bsz + b]
                else:  # kw
                    cand = sub_cands[self.K * bsz + (kidx - ins_cand_idx_end - self.K)]
                    cand_3d = sub_2d_score[self.K * bsz + (kidx - ins_cand_idx_end - self.K)]
                    cand_sim = sub_sim_tuple[self.K * bsz + (kidx - ins_cand_idx_end - self.K)]
            elif kidx == sub_cand_idx_end:  # del
                cand = del_inps[b]
                cand_3d = del_2d_score[b]
                cand_sim = del_sim_tuple[b]
            elif kidx > sub_cand_idx_end: # rule
                cand = rule_cands[kidx - sub_cand_idx_end - 1]
                acc_rule += 1
                cand_3d = None
                cand_sim=None
            else:
                raise ValueError('')

            cands_sampled.append(cand)
            cands_sampled_3d.append(cand_3d)
            cands_sampled_sim_tuple.append(cand_sim)

        return cands_sampled, cands_sampled_probs, cands_sampled_score, cands_sampled_3d, cands_sampled_sim_tuple
