import torch
from math import log
from itertools import chain
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from rake_nltk import Rake
from collections import defaultdict

def sent_encode(tokenizer, sent):
    "Encoding as sentence based on the tokenizer"
    return tokenizer.encode(sent.strip(), add_special_tokens=True,
                            add_prefix_space=True,
                            max_length=tokenizer.max_len)

def read_tf(vocab_path, tknzer):
    tf_dict = defaultdict(lambda: 0.)
    with open(vocab_path, encoding='utf-8') as f:
        for line in f:
            term, freq = line.split()
            for subtok in tknzer.tokenize(term):
                subid = tknzer.vocab[subtok]
                tf_dict[subid] += int(freq)
    return tf_dict


def padding(arr, pad_token, dtype=torch.long, mask_head_tail=False):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        if mask_head_tail:
            mask[i, 1:lens[i] - 1] = 1
        else:
            mask[i, :lens[i]] = 1
    return padded, lens, mask


def _bert_encode(model, x, attention_mask):
    model.eval()
    x_seg = torch.zeros_like(x, dtype=torch.long)
    with torch.no_grad():
        x_encoded_layers, pooled_output = model(x, x_seg, attention_mask=attention_mask,
                                                output_all_encoded_layers=False)
    return x_encoded_layers


def bert_encode(model, x, attention_mask, sent_layer=-1, kw_layer=1):
    with torch.no_grad():
        x = x.to(torch.device('cuda', 1))
        attention_mask = attention_mask.to(torch.device('cuda', 1))
        inp_type = torch.full_like(x, 0).to(torch.device('cuda', 1))

        out = model('bertscore', x, attention_mask=attention_mask, token_type_ids=inp_type)[-1][1:]
    return out[sent_layer], out[sent_layer], out[kw_layer][:, 0, :]


def process(a, tokenizer=None):
    if tokenizer is not None:
        a = sent_encode(tokenizer, a)
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.
    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    return idf_dict


def collate_tf_no_tokenize(arr, numericalize, tf_dict=None,
                           pad="<pad>", device='cuda'):
    """
    Helper function that pads a list of sentences to have the same length and
    loads idf score for words in the sentences.
    Args:
        - :param: `arr` (list of str): sentences to process.
        - :param: `tokenize` : a function that takes a string and return list
                  of tokens.
        - :param: `numericalize` : a function that takes a list of tokens and
                  return list of token indexes.
        - :param: `idf_dict` (dict): mapping a word piece index to its
                               inverse document frequency
        - :param: `pad` (str): the padding token.
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    arr = [["<s>"] + a + ["</s>"] for a in arr]
    arr = [numericalize(a) for a in arr]

    if tf_dict is None:
        tf_matrix = [[1. for i in a] for a in arr]
    else:
        tf_matrix = [[tf_dict[i] for i in a] for a in arr]

    pad_token = numericalize([pad])[0]

    # masked_pos =

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_tf, _, _ = padding(tf_matrix, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_tf, lens, mask


def collate_tf_idf(arr, tokenizer, tf_dict=None, punc_ids=None, device='cuda',
                   pad="<pad>", mask_head_tail=False, keywords=None,
                   cls_id=-1, sep_id=-1):

    # kws = [sent_encode(tokenizer, w)[1:-1] for w in ' '.join(kw).split(' ') for kw in keywords]
    arr = [sent_encode(tokenizer, a) for a in arr]
    kw_mask = [[0.]*len(a) for a in arr]
    for b, a in enumerate(arr):
        for i, subtok in enumerate(a):
            if subtok in punc_ids:
                kw_mask[b][i] = 0.

    # arr = [numericalize(a) for a in arr]

    idf_weights = [[0. if i in [cls_id, sep_id] else 1. for i in a] for a in arr]

    pad_token = tokenizer.pad_token_id

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long, mask_head_tail=mask_head_tail)
    padded_kw, _, _ = padding(kw_mask, pad_token, dtype=torch.float)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, padded_kw, lens, mask


def get_bert_embedding_with_kw(padded_sens, padded_idf, padded_kw, lens, mask,
                               model, sent_layer, kw_layer,
                               batch_size=-1, device='cuda', mask_head_tail=False):
    if batch_size == -1: batch_size = len(padded_sens)

    embeddings = []
    embeddings_kw = []
    embeddings_bos = []

    with torch.no_grad():
        for i in range(0, len(padded_sens), batch_size):
            batch_embedding, batch_embedding_kw, batch_bos_embedding = bert_encode(model, padded_sens[i:i + batch_size],
                                                           attention_mask=mask[i:i + batch_size], sent_layer=sent_layer, kw_layer=kw_layer)
            # batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            embeddings_kw.append(batch_embedding_kw)
            embeddings_bos.append(batch_bos_embedding)
            del batch_embedding
            del embeddings_kw
            del batch_bos_embedding

    total_embedding = torch.cat(embeddings, dim=0)
    total_embedding_kw = torch.cat(embeddings_kw, dim=0)
    total_embedding_bos = torch.cat(embeddings_bos, dim=0)
    return total_embedding, total_embedding_kw, total_embedding_bos, lens, mask, padded_idf, padded_kw


def greedy_cos_tf_idf(ref_embedding, ref_embedding_kw, ref_bos_emb, ref_lens, ref_masks, ref_idf, ref_kw,
                        hyp_embedding, hyp_embedding_kw, hyp_bos_emb, hyp_lens, hyp_masks, hyp_idf, hyp_kw,
                        alpha, beta):
    '''use ref_embedding_kw'''
    # ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    # hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1)) # [bsz, hyplen, bert dim]

    ref_embedding_kw.div_(torch.norm(ref_embedding_kw, dim=-1).unsqueeze(-1))
    hyp_embedding_kw.div_(torch.norm(hyp_embedding_kw, dim=-1).unsqueeze(-1))

    batch_size = ref_embedding.size(0)

    # sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2)) # [bsz,hyplen, reflen]
    sim_kw = torch.bmm(hyp_embedding_kw, ref_embedding_kw.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, masks.size(1), masks.size(2)) \
        .contiguous().view_as(sim_kw)

    hyp_kw = hyp_kw.to(sim_kw.device)  # [10, 13]
    ref_kw = ref_kw.to(sim_kw.device)  # [10, 12]
    kw_masks = torch.bmm(hyp_kw.unsqueeze(2).float(), ref_kw.unsqueeze(1).float())

    masks = masks.float().to(sim_kw.device)
    sim_kw = sim_kw * masks * kw_masks

    word_precision_kw = sim_kw.max(dim=2)[0]  # [bsz*K, ]
    word_recall_kw = sim_kw.max(dim=1)[0]  # [bsz*K, ]

    # == minimal ==
    large1 = torch.full_like(word_precision_kw, 1e5)
    large2 = torch.full_like(word_recall_kw, 1e5)

    # P = torch.min(torch.where(word_precision > 0, word_precision, large1), 1)[0]
    # R = torch.min(torch.where(word_recall > 0, word_recall, large2), 1)[0]
    P_kw = torch.min(torch.where(word_precision_kw > 0, word_precision_kw, large1), 1)[0]
    R_kw = torch.min(torch.where(word_recall_kw > 0, word_recall_kw, large2), 1)[0]
    # if P_kw.eq(0.).any():

    # ! hyp_kw maybe 0 in one sentence.
    small = torch.full_like(P_kw, 1e-10)
    P_kw = torch.where(torch.ge(torch.ones_like(P_kw), P_kw), P_kw, small)
    R_kw = torch.where(torch.ge(torch.ones_like(R_kw), R_kw), R_kw, small)


    # F = 2 * P * R / (P + R)
    F_kw = 2 * P_kw * R_kw / (P_kw + R_kw)
    F = torch.tensor([0.]) * len(F_kw)

    assert not (torch.isinf(F_kw).any() or torch.isnan(F_kw).any())

    ref_bos_emb.div_(torch.norm(ref_bos_emb, dim=-1).unsqueeze(-1))
    hyp_bos_emb.div_(torch.norm(hyp_bos_emb, dim=-1).unsqueeze(-1))
    sem_sim = torch.sum(ref_bos_emb * hyp_bos_emb, dim=-1)
    sem_sim = (sem_sim + 1)/2

    F1 = (F_kw ** alpha + 1e-10) * (sem_sim ** beta)
    return P_kw, R_kw, F1, F, F_kw


def bert_cos_score_tfidf_batch(model, refs, hyps, tokenizer, kw_layer, tf_dict, alpha, beta,
                               verbose=False, device='cuda', mask_head_tail=True):
    """
    Compute BERTScore.
    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `idf_dict` : a dictionary mapping a word piece index to its
                               inverse document frequency
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """

    preds = []
    kw_extractor = Rake()

    ref_stats_kw = get_bert_embedding_with_kw(refs, model, tokenizer, kw_layer, tf_dict, kw_extractor,
                                              device=device, mask_head_tail=True)
    hyp_stats_kw = get_bert_embedding_with_kw(hyps, model, tokenizer, kw_layer, tf_dict, kw_extractor,
                                              device=device, mask_head_tail=True)

    P, R, F1, F_sent, F_kw = greedy_cos_tf_idf(*ref_stats_kw, *hyp_stats_kw, alpha, beta)
    preds.append(torch.stack((P, R, F1), dim=1))
    preds = torch.cat(preds, dim=0)
    return preds, (F_sent, F_kw)


def merge_sub_tokens(subtokens):
    while ' ##' in subtokens:
        subtokens = subtokens.replace(' ##', '')
    return subtokens


def convert_tensor_to_tokens(ids_tensor, mask_tensor, i2w, rm_cls=True, rm_sep=True):
    '''convert a tensor to list of (list of tokens)'''
    tokens = []
    for ids, mask in zip(ids_tensor.cpu().numpy(), mask_tensor.cpu().numpy()):
        ids = ids[:mask]
        str_ = [i2w(x) for x in ids]
        if rm_cls: str_ = str_[1:]
        if rm_sep: str_ = str_[:-1]
        tokens.append(str_)
    return tokens
