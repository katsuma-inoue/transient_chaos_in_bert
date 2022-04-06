import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Tuple
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer
)


logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForAlbertLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.10
    n_gram: int = 3

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training
        # (with probability args.mlm_probability defaults to 0.10 in ALBERT)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        # n-gram masking.
        ngrams = np.arange(1, self.n_gram+1)
        pvals = 1. / np.arange(1, self.n_gram + 1)
        pvals /= pvals.sum(keepdims=True)

        batch_size = labels.shape[0]
        seq_len = labels.shape[1] - labels.eq(self.tokenizer.pad_token_id).sum(axis=1)
        seq_len = seq_len.numpy()
        max_mask_cnt = seq_len.astype(np.float) * self.mlm_probability
        max_mask_cnt = max_mask_cnt.astype(np.int32)
        masked_place = torch.zeros_like(labels)
        for i in range(batch_size):
            mask_cnt = 0
            iterate_cnt = 0
            while mask_cnt < max_mask_cnt[i]:
                n = np.random.choice(ngrams, p=pvals)
                n = min(n, max_mask_cnt[i] - mask_cnt)
                random_start = int(np.random.choice(np.arange(seq_len[i]-n)))
                random_end = int(random_start + n)
                phrase_probs = probability_matrix[i, random_start:random_end].numpy()
                if phrase_probs.shape[0] == 0 or np.cumprod(phrase_probs)[-1] == 0.0:
                    continue
                else:
                    masked_place[i, random_start:random_end] = 1
                    mask_cnt += n
                iterate_cnt += 1
                if iterate_cnt > seq_len[i]*2:
                    break

        masked_indices = masked_place.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
