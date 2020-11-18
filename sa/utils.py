import torch
from torch import nn 
import numpy as np
import nltk
from torch.autograd import Variable

def positional_encoding(len_, sz):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    pe = np.array([
        [pos / np.power(10000, 2 * (j // 2) / sz) for j in range(sz)]
         for pos in range(len_)])

    pe[1:, 0::2] = np.sin(pe[1:, 0::2]) # dim 2i
    pe[1:, 1::2] = np.cos(pe[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(pe).type(torch.FloatTensor)

def seq_msk_loss(loss, pred, tgt, mask):
    if not issubclass(type(loss), nn.modules.loss._Loss):
        raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
    dim = pred.size(-1)
    msk_loss = torch.matmul(
            loss(pred.contiguous().view(-1, dim), tgt.contiguous().view(-1)),
            mask.contiguous().view(-1))
    avg_loss = torch.sum(msk_loss) / torch.max(torch.sum(mask), Variable(torch.FloatTensor([1.]).cuda()))
    #print('avg loss', torch.mean(avg_loss), torch.sum(mask))
    return avg_loss

def reverse_mask(mask):
    '''0 to 1, 1 to 0'''
    msk = mask.clone()
    msk[msk==0] = 2
    msk[msk==1] = 0
    msk[msk==2] = 1
    return msk

def cal_bleu_score(decoded, target):
    return nltk.translate.bleu_score.sentence_bleu([target], decoded, 
            smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1) 

def cal_bleu_score_multi(decoded, target):
    return nltk.translate.bleu_score.sentence_bleu(target, decoded,
            smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)