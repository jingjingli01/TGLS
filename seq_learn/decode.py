import logging
import os
import torch
from torch import nn
from torch.nn import CrossEntropy
from torch.nn import functional as F

from transformers.configuration_utils import PretrianedConfig

class ATSearcher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        src_memory_bank,
        input_ids=None,
        max_length=None,
        do_sample=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        pad_token_id=None,
        eos_token_ids=None,
        batch_size=None,
        length_penalty=None,
        num_beams=None,
        num_return_sequences=None
    ):
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
            )

        model = self.model
        model_self = model.module if hasttr(model, 'module') else model
        max_length = max_length if max_length is not None else model_self.config.max_length
        do_sample = do_sample if do_sample is not None else model_self.config.do_sample
        num_beams = num_beams if num_beams is not None else model_self.config.num_beams
        temperature = temperature if temperature is not None else model_self.config.temperature
        top_k = top_k if top_k is not None else model_self.config.top_k
        top_p = top_p if top_p is not None else model_self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else model_self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else model_self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else model_self.config.pad_token_id
        eos_token_ids = eos_token_ids if eos_token_ids is not None else model_self.config.eos_token_ids
        length_penalty = length_penalty if length_penalty is not None else model_self.config.length_penalty

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
        assert temperature > 0, "`temperature` should be strictely positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, "`bos_token_id` should be a positive integer."
        assert isinstance(pad_token_id, int) and pad_token_id >= 0, "`pad_token_id` should be a positive integer."
        assert isinstance(eos_token_ids, (list, tuple)) and (
            e >= 0 for e in eos_token_ids
        ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert length_penalty > 0, "`length_penalty` should be strictely positive."

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        cur_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size

        if num_return_sequences != 1:
            input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
            input_ids = input_ids.contiguous().view(
                batch_size * num_return_sequences, cur_len
            )
            batch_size = batch_size * num_return_sequences
        else:
            pass
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)

        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        past = None
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(**model_inputs)
            scores = outputs[0][:, -1, :]

            if self._do_output_past(outputs):
                past = outputs[1]

            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty


            scores = F.log_softmax(scores, dim=-1)  
            assert scores.size() == (batch_size * num_beams, vocab_size)
            _scores = scores + beam_scores[:, None].expand_as(scores) 
            _scores = _scores.view(batch_size, num_beams * vocab_size)
            next_scores, next_words = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, 2 * num_beams)

            next_batch_beam = []

            for batch_ex in range(batch_size):

                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                next_sent_beam = []

                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                        generated_hyps[batch_ex].add(
                            input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    if len(next_sent_beam) == num_beams:
                        break

                assert len(next_sent_beam) == 0 if cur_len + 1 == max_length else num_beams
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            if past:
                reordered_past = []
                for layer_past in past:
                    reordered_layer_past = [layer_past[:, i].unsqueeze(1).clone().detach() for i in beam_idx]
                    reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
                    assert reordered_layer_past.shape == layer_past.shape
                    reordered_past.append(reordered_layer_past)
                past = tuple(reordered_past)

            cur_len = cur_len + 1

            if all(done):
                break

        tgt_len = input_ids.new(batch_size)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  
            best.append(best_hyp)

        decoded = input_ids.new(batch_size, tgt_len.max().item()).fill_(pad_token_id)
        for i, hypo in enumerate(best):
            decoded[i, : tgt_len[i] - 1] = hypo
            decoded[i, tgt_len[i] - 1] = eos_token_ids[0]

        return decoded