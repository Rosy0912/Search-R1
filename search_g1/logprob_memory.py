"""FP32 selected log-probabilities with a bounded-scratch first-order VJP.

Each row still reduces over the entire vocabulary. Chunking reduces temporary
storage; it does not approximate the softmax or alter token/loss weighting.
The dense logits gradient remains unavoidable. Higher-order derivatives are
not supported by this custom VJP.
"""

import itertools
import os

import torch
from torch.autograd.function import once_differentiable


DEFAULT_CHUNK_ROWS = 128
MAX_SCRATCH_BYTES = 80 * 1024 * 1024


def chunk_rows(vocab_size):
    """Use a configurable row chunk, capped at about 80 MiB of FP32 scratch.

    A 151936-token vocabulary uses about 74.2 MiB at the default 128 rows.
    A single vocabulary row is the minimum allocation for this implementation.
    """
    requested = int(os.environ.get('SEARCH_G1_LOGPROB_CHUNK_ROWS', DEFAULT_CHUNK_ROWS))
    if requested < 1:
        raise ValueError('SEARCH_G1_LOGPROB_CHUNK_ROWS must be positive')
    bounded = max(1, MAX_SCRATCH_BYTES // (vocab_size * 4))
    return min(requested, bounded)


def _row_views(logits, labels, normalizer, extra):
    # Prefix iteration preserves strided B,T,V views without a reshape copy.
    if logits.ndim == 1:
        yield tuple(value.unsqueeze(0) for value in (logits, labels, normalizer, extra))
    else:
        prefixes = itertools.product(*(range(n) for n in logits.shape[:-2]))
        for prefix in prefixes:
            yield tuple(value[prefix] for value in (logits, labels, normalizer, extra))


class _SelectedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        if logits.dtype != torch.float32:
            raise TypeError('selected_log_probs requires FP32 logits')
        if logits.ndim < 1 or logits.shape[-1] < 1:
            raise ValueError('logits must have a nonempty vocabulary dimension')
        if labels.shape != logits.shape[:-1]:
            raise ValueError('labels must match the logits prefix dimensions')
        rows_per_chunk = chunk_rows(logits.shape[-1])
        normalizer = torch.empty(labels.shape, dtype=torch.float32, device=logits.device)
        output = torch.empty_like(normalizer)
        for rows, ids, lse, target in _row_views(logits, labels, normalizer, output):
            for start in range(0, len(rows), rows_per_chunk):
                stop = start + rows_per_chunk
                lse[start:stop].copy_(torch.logsumexp(rows[start:stop], dim=-1))
                selected = rows[start:stop].gather(-1, ids[start:stop].unsqueeze(-1)).squeeze(-1)
                target[start:stop].copy_(selected - lse[start:stop])
        ctx.rows_per_chunk = rows_per_chunk
        ctx.save_for_backward(logits, labels, normalizer)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, weights):
        logits, labels, normalizer = ctx.saved_tensors
        gradient = torch.empty_like(logits, memory_format=torch.contiguous_format)
        # One FP32 scratch block: exp_(x-lse) avoids a second vocabulary buffer.
        views = _row_views(logits, labels, normalizer, weights)
        if logits.ndim == 1:
            targets = iter((gradient.unsqueeze(0),))
        else:
            prefixes = itertools.product(*(range(n) for n in logits.shape[:-2]))
            targets = (gradient[prefix] for prefix in prefixes)
        for (rows, ids, lse, upstream), target in zip(views, targets):
            for start in range(0, len(rows), ctx.rows_per_chunk):
                stop = start + ctx.rows_per_chunk
                block = rows[start:stop] - lse[start:stop].unsqueeze(-1)
                block.exp_().mul_(-upstream[start:stop].unsqueeze(-1))
                block.scatter_add_(-1, ids[start:stop].unsqueeze(-1),
                                   upstream[start:stop].unsqueeze(-1))
                target[start:stop].copy_(block)
        return gradient, None


def selected_log_probs(logits, labels):
    """Return z[label] - logsumexp(z), with first-order VJP g*(onehot-softmax)."""
    return _SelectedLogProbs.apply(logits, labels)
