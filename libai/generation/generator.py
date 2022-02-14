# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import oneflow as flow
import oneflow.nn as nn

from libai.models.utils import ModelType

from .beam_search import BeamSearchScorer


class Generator(nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        beam_size=1,
        min_length=1,
        max_length=200,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        repetition_penalty=1.0,
        temperature=1.0,
        no_repeat_ngram_size=0,
        num_return_sequences=1,
        num_beam_groups=1,
        device=None,
        placement=None,
        sbp=None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.unk_token_id = tokenizer.unk_token_id
        self.min_length = min_length
        self.max_length = max_length

        self.vocab_size = len(tokenizer)

        self.num_beams = beam_size
        self.num_beam_groups = num_beam_groups
        self.num_sub_beams = beam_size // num_beam_groups
        self.num_return_sequences = num_return_sequences

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.no_repeat_ngram_size = no_repeat_ngram_size

        self.device = device
        self.placement = placement
        self.sbp = sbp
        assert (self.device is None) or (
            self.placement is None
        ), "device and placement cannot be set at the same time."

    def cuda(self):
        self.model.cuda()
        return self

    def is_beam_mode(self):
        return (self.num_beams == 1) and (self.num_beam_groups == 1)

    def is_group_mode(self):
        return (self.num_beams > 1) and (self.num_beam_groups > 1)

    @flow.no_grad()
    def generate(self, input_ids=None, do_sample=False, use_cache=True):
        """
        这个类专注于自回归生成，也就是说，不适用于BERT或NAT那样的非自回归模型。
        可以生成的语言模型分两类，一种是GPT那样的，根据提示生成，一种是BART那样的，根据源语句生成。
        它们的区别在于：
            GPT类的模型，input_ids是上文/提示，按照这个继续生成即可。
            BART类的模型，input_ids是源句，因此，先调用编码器，生成encoder_states，然后构造新的input_ids，继续生成。
        对于GPT类的模型：CasualLM
            分两种情况：输入为空，或输入提示，这里特判即可
            然后就是正常的生成（贪心解码、beam search、采样）。
        对于BART类模型：Seq2SeqLM
            先调用编码器生成encoder_states，然后构造新的input_ids。继续生成。
        """
        model_kwargs = {}
        model_kwargs["use_cache"] = use_cache
        model_kwargs["past_key_values"] = None

        if input_ids is not None:  # 输入不为空，输入维度是 (batch_size, seq_length)
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, seq_length)."
            batch_size = input_ids.shape[0]
        else:  # 输入为空，例如gpt，让模型自由生成
            batch_size = 1
            input_ids = self._prepare_input_ids(batch_size)

        model_kwargs = self._prepare_attention_mask(
            input_ids, **model_kwargs
        )  # 准备attention_mask，对所有模型一视同仁，均计算

        if self.model.model_type == ModelType.encoder_decoder:
            # 如果是encoder_decoder类模型，先调用encoder，得到encoder_states
            # 然后准备decoder的参数，如encoder_states，encoder_attention_mask
            # 重新生成input_ids，attention_mask
            model_kwargs = self._prepare_encoder_decoder_kwargs(input_ids, **model_kwargs)
            input_ids = self._prepare_input_ids(batch_size)
            model_kwargs = self._prepare_attention_mask(input_ids, **model_kwargs)

        if self.is_group_mode:
            if do_sample:
                raise ValueError(
                    "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
                )
            return self.group_beam_search(input_ids, **model_kwargs)
        else:
            if self.is_beam_mode:
                if do_sample:
                    return self.beam_sample(input_ids, **model_kwargs)
                else:
                    return self.beam_search(input_ids, **model_kwargs)
            else:
                if do_sample:
                    return self.sample(input_ids, **model_kwargs)
                else:
                    return self.greedy_search(input_ids, **model_kwargs)

    def _prepare_input_ids(self, batch_size):
        """prepare input_ids for auto-regressive models"""
        input_ids = flow.full(
            (batch_size, 1),
            self.bos_token_id,
            dtype=flow.long,
            device=self.device,
            placement=self.placement,
            sbp=self.sbp,
        )
        return input_ids

    def _prepare_attention_mask(self, input_ids, **model_kwargs):
        """prepare attention_mask for generation"""
        if self.pad_token_id is not None:
            attention_mask = input_ids.ne(self.pad_token_id)
        else:
            attention_mask = input_ids.new_ones(input_ids.shape, dtype=flow.int8)
        model_kwargs["attention_mask"] = attention_mask
        return model_kwargs

    def _prepare_encoder_decoder_kwargs(self, input_ids, **model_kwargs):
        attention_mask = model_kwargs["attention_mask"]
        encoder_states = self.model.encoder(input_ids, attention_mask)
        model_kwargs["encoder_states"] = encoder_states
        model_kwargs["encoder_attention_mask"] = attention_mask
        return model_kwargs

    def _prepare_beam_search_scorer(self, batch_size):
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=self.num_beams,
            device=self.device,
            length_penalty=self.len_penalty,
            do_early_stopping=self.early_stopping,
            num_beam_hyps_to_keep=self.num_return_sequences,
            num_beam_groups=self.num_beam_groups,
        )
        return beam_scorer

    def _prepare_model_inputs(self, input_ids, **model_kwargs):
        if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
            input_ids = input_ids[:, -1:]
        model_inputs = {}
        model_inputs["input_ids"] = input_ids
        for k, v in model_kwargs.items():
            model_inputs[k] = v
        return model_inputs

    def _expand_inputs(self, input_ids, expand_size=1, **model_kwargs):
        if expand_size == 1:
            return input_ids, model_kwargs

        expanded_return_idx = (
            flow.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, self.num_beams)
            .view(-1)
            .to(self.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx.to(input_ids.device))

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if self.model.model_type == ModelType.encoder_decoder:
            encoder_states = model_kwargs["encoder_states"]
            encoder_attention_mask = model_kwargs["encoder_attention_mask"]
            model_kwargs["encoder_states"] = encoder_states.index_select(
                0, expanded_return_idx.to(encoder_states.device)
            )
            model_kwargs["encoder_attention_mask"] = encoder_attention_mask.index_select(
                0, expanded_return_idx.to(encoder_attention_mask.device)
            )

        return input_ids, model_kwargs

    def _update_model_kwargs(self, outputs, **model_kwargs):
        if len(outputs) > 1:
            model_kwargs["past_key_values"] = outputs[1]

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = flow.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        return model_kwargs

    def greedy_search(self, input_ids, **model_kwargs):
        is_encoder_decoder = self.model.model_type == ModelType.encoder_decoder

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        while True:
            model_inputs = self._prepare_model_inputs(input_ids, **model_kwargs)
            if is_encoder_decoder:
                outputs = self.model.decoder(**model_inputs)
            else:
                outputs = self.model(**model_inputs)

            next_token_logits = outputs[0][:, -1, :]

            # pre-process distribution
            next_token_scores = self.process_logits(input_ids, next_token_logits)

            # argmax
            next_tokens = flow.argmax(next_token_scores, dim=-1)  # (batch_size, 1)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (
                1 - unfinished_sequences
            )

            # update generated ids, model inputs, and length for next step
            input_ids = flow.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs(outputs, **model_kwargs)
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                (next_tokens != self.eos_token_id).long()
            )

            if unfinished_sequences.max() == 0:
                break

        return input_ids

    def sample(self, input_ids, **model_kwargs):
        is_encoder_decoder = self.model.model_type == ModelType.encoder_decoder

        input_ids, model_kwargs = self._expand_inputs(
            input_ids,
            expand_size=self.num_return_sequences,
            **model_kwargs,
        )

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        while True:
            model_inputs = self._prepare_model_inputs(input_ids, **model_kwargs)
            if is_encoder_decoder:
                outputs = self.model.decoder(**model_inputs)
            else:
                outputs = self.model(**model_inputs)

            next_token_logits = outputs[0][:, -1, :]

            # pre-process distribution
            next_token_scores = self.process_logits(input_ids, next_token_logits)

            # sample
            probs = flow.softmax(next_token_scores, dim=-1)
            next_tokens = flow.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (
                1 - unfinished_sequences
            )

            # update generated ids, model inputs, and length for next step
            input_ids = flow.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs(outputs, **model_kwargs)
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                (next_tokens != self.eos_token_id).long()
            )

            if unfinished_sequences.max() == 0:
                break

        return input_ids

    def beam_search(self, input_ids, **model_kwargs):
        is_encoder_decoder = self.model.model_type == ModelType.encoder_decoder
        batch_size, cur_len = input_ids.shape

        beam_scorer = self._prepare_beam_search_scorer(batch_size)
        input_ids, model_kwargs = self._expand_inputs(
            input_ids,
            expand_size=self.num_beams,
            **model_kwargs,
        )

        beam_scores = flow.zeros(
            (batch_size, self.num_beams), dtype=flow.float32, device=self.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * self.num_beams,))

        while True:
            model_inputs = self._prepare_model_inputs(input_ids, **model_kwargs)
            if is_encoder_decoder:
                outputs = self.model.decoder(**model_inputs)
            else:
                outputs = self.model(**model_inputs)

            next_token_logits = outputs[0][:, -1, :]
            next_token_scores = flow.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = self.process_logits(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores
            )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, self.num_beams * vocab_size)

            next_token_scores, next_tokens = flow.topk(
                next_token_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens - next_indices * vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = flow.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self._update_model_kwargs(outputs, **model_kwargs)

            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self.model.reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            max_length=self.max_length,
        )
        return sequence_outputs["sequences"]

    def beam_sample(self, input_ids, **model_kwargs):
        is_encoder_decoder = self.model.model_type == ModelType.encoder_decoder
        batch_size, cur_len = input_ids.shape

        beam_scorer = self._prepare_beam_search_scorer(batch_size)
        input_ids, model_kwargs = self._expand_inputs(
            input_ids,
            expand_size=self.num_beams * self.num_return_sequences,
            **model_kwargs,
        )

        beam_scores = flow.zeros(
            (batch_size, self.num_beams), dtype=flow.float32, device=self.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * self.num_beams,))

        while True:
            model_inputs = self._prepare_model_inputs(input_ids, **model_kwargs)
            if is_encoder_decoder:
                outputs = self.model.decoder(**model_inputs)
            else:
                outputs = self.model(**model_inputs)

            next_token_logits = outputs[0][:, -1, :]
            next_token_scores = flow.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores = self.process_logits(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, self.num_beams * vocab_size)

            probs = flow.softmax(next_token_logits, dim=-1)

            next_tokens = flow.multinomial(probs, num_samples=2 * self.num_beams)
            next_token_scores = flow.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = flow.sort(next_token_scores, descending=True, dim=1)
            next_tokens = flow.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens - next_indices * vocab_size

            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = flow.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self._update_model_kwargs(outputs, **model_kwargs)

            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self.model.reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            max_length=self.max_length,
        )
        return sequence_outputs["sequences"]

    def group_beam_search(self, input_ids, **model_kwargs):
        is_encoder_decoder = self.model.model_type == ModelType.encoder_decoder
        batch_size, cur_len = input_ids.shape

        beam_scorer = self._prepare_beam_search_scorer(batch_size)
        input_ids, model_kwargs = self._expand_inputs(
            input_ids,
            expand_size=self.num_beams * self.num_return_sequences,
            **model_kwargs,
        )

        beam_scores = flow.full(
            (batch_size, self.num_beams), -1e9, dtype=flow.float32, device=self.device
        )
        beam_scores[:, :: self.num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * self.num_beams,))

        while True:
            # predicted tokens in cur_len step
            current_tokens = flow.zeros(
                batch_size * self.num_beams,
                dtype=input_ids.dtype,
                device=self.device,
            )

            # indices which will form the beams in the next time step
            reordering_indices = flow.zeros(
                batch_size * self.num_beams, dtype=flow.long, device=self.device
            )

            model_inputs = self._prepare_model_inputs(input_ids, **model_kwargs)
            if is_encoder_decoder:
                outputs = self.model.decoder(**model_inputs)
            else:
                outputs = self.model(**model_inputs)

            for beam_group_idx in range(self.num_beam_groups):
                group_start_idx = beam_group_idx * self.num_sub_beams
                group_end_idx = min(group_start_idx + self.num_sub_beams, self.num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [
                            batch_idx * self.num_beams + idx
                            for idx in range(group_start_idx, group_end_idx)
                        ]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of current group only
                next_token_logits = outputs.logits[batch_group_indices, -1, :]
                next_token_scores = flow.log_softmax(
                    next_token_logits, dim=-1
                )  # (batch_size * group_size, vocab_size)
                vocab_size = next_token_scores.shape[-1]

                next_token_scores_processed = self.process_logits(
                    group_input_ids,
                    next_token_scores,
                    current_tokens=current_tokens,
                    beam_group_idx=beam_group_idx,
                )
                next_token_scores = next_token_scores_processed + beam_scores[
                    batch_group_indices
                ].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = flow.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens - next_indices * vocab_size

                # stateless
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = flow.cat(
                    [group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)],
                    dim=-1,
                )
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    self.num_beams * (beam_idx // group_size)
                    + group_start_idx
                    + (beam_idx % group_size)
                )

            input_ids = flow.cat([input_ids[beam_idx, :], current_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self._update_model_kwargs(outputs, **model_kwargs)

            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self.model.reorder_cache(
                    model_kwargs["past_key_values"], reordering_indices
                )

            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            max_length=self.max_length,
        )
        return sequence_outputs["sequences"]

    def process_logits(
        self,
        input_ids,
        logits,
        current_tokens=None,
        beam_group_idx=None,
        filter_value=-1e9,
        min_tokens_to_keep=1,
    ):
        # if self.diversity_penalty > 0.0:
        #     logits = self.hamming_diversity_logits_process(input_ids, logits, current_tokens=current_tokens, beam_group_idx=beam_group_idx)
        if self.repetition_penalty != 1.0:
            logits = self.repetition_penalty_logits_process(input_ids, logits)
        if self.no_repeat_ngram_size > 0:
            logits = self.no_repeat_ngram_logits_process(input_ids, logits)
        if self.min_length > -1:
            logits = self.min_length_logits_process(input_ids, logits)
        if self.temperature != 1.0:
            logits = self.temperature_logits(logits)
        if self.top_k > 0:
            logits = self.top_k_filtering(
                logits, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep
            )
        if self.top_p > 0.0:
            logits = self.top_p_filtering(
                logits, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep
            )
        return logits

    def hamming_diversity_logits_process(self, input_ids, logits, current_tokens, beam_group_idx):
        batch_size = current_tokens.shape[0] // self.num_beams
        group_start_idx = beam_group_idx * self.num_sub_beams
        group_end_idx = min(group_start_idx + self.num_sub_beams, self.num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = logits.shape[-1]

        if group_start_idx == 0:
            return logits

        for batch_idx in range(batch_size):
            previous_group_tokens = current_tokens[
                batch_idx * self.num_beams : batch_idx * self.num_beams + group_start_idx
            ]
            token_frequency = flow.bincount(previous_group_tokens, minlength=vocab_size).to(
                logits.device
            )  # 目前不支持bincount算子
            logits[batch_idx * group_size : (batch_idx + 1) * group_size] -= (
                self.diversity_penalty * token_frequency
            )

        return logits

    def repetition_penalty_logits_process(self, input_ids, logits):
        """If repetition_penalty equals 1.0, this means no penalty.
        Refers to `<https://arxiv.org/pdf/1909.05858.pdf>` for more details.
        """
        logit = flow.gather(logits, 1, input_ids)
        logit = flow.where(
            logit < 0, logit * self.repetition_penalty, logit / self.repetition_penalty
        )
        logits.scatter_(1, input_ids, logit)
        return logits

    def min_length_logits_process(self, input_ids, logits):
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            logits[:, self.eos_token_id] = -1e9
        return logits

    def no_repeat_ngram_logits_process(self, input_ids, logits):
        num_hypos = logits.shape[0]
        cur_len = input_ids.shape[-1]
        if cur_len + 1 < self.no_repeat_ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            banned_batch_tokens = [[] for _ in range(num_hypos)]
        else:
            generated_ngrams = [{} for _ in range(num_hypos)]
            for idx in range(num_hypos):
                gen_tokens = input_ids[idx].tolist()
                generated_ngram = generated_ngrams[idx]
                for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                    prev_ngram_tuple = tuple(ngram[:-1])
                    generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                        prev_ngram_tuple, []
                    ) + [ngram[-1]]

            banned_batch_tokens = []
            for hypo_idx in range(num_hypos):
                start_idx = cur_len + 1 - self.no_repeat_ngram_size
                ngram_idx = tuple(input_ids[hypo_idx][start_idx:cur_len].tolist())
                banned_tokens = generated_ngrams[hypo_idx].get(ngram_idx, [])
                banned_batch_tokens.append(banned_tokens)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            logits[i, banned_tokens] = -1e9

        return logits

    def force_eos_token_logits_process(self, input_ids, logits):
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            logits[:, :] = -1e9
            logits[:, self.eos_token_id] = 0
        return logits

    def temperature_logits(self, logits):
        return logits / self.temperature

    def top_k_filtering(self, logits, filter_value=-1e9, min_tokens_to_keep=1):
        top_k = min(max(self.top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < flow.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

        return logits

    def top_p_filtering(self, logits, filter_value=-1e9, min_tokens_to_keep=1):
        sorted_logits, sorted_indices = flow.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep - 1 because we add the first one below)
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)

        return logits
