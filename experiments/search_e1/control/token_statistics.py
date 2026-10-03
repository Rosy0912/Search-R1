"""Checkpoint row-local entropy/log-prob computations without changing reductions."""
HELPER='''def _e1_token_statistics(logits, labels, entropy_fn):
    chunk = int(os.environ.get('E1_LOGIT_CHUNK', '0'))
    if chunk < 0:
        raise ValueError('Invalid statistics chunk')
    if not chunk:
        return entropy_fn(logits), logprobs_from_logits(logits, labels)
    shape = labels.shape
    rows = logits.reshape(-1, logits.shape[-1])
    targets = labels.reshape(-1)
    entropies, probabilities = [], []
    def statistics(values, target):
        return entropy_fn(values), logprobs_from_logits(values, target)
    for start in range(0, rows.shape[0], chunk):
        values, target = rows[start:start + chunk], targets[start:start + chunk]
        if torch.is_grad_enabled() and values.requires_grad:
            entropy, probability = checkpoint(statistics, values, target, use_reentrant=True)
        else:
            entropy, probability = statistics(values, target)
        entropies.append(entropy); probabilities.append(probability)
    return torch.cat(entropies).reshape(shape), torch.cat(probabilities).reshape(shape)


'''
def patch_statistics(source):
 if 'def _e1_token_statistics(' in source:raise ValueError('Statistics overlay already applied')
 source='import os\nfrom torch.utils.checkpoint import checkpoint\n'+source
 source=source.replace('class DataParallelPPOActor(',HELPER+'class DataParallelPPOActor(',1)
 first='''                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)'''
 assert source.count(first)==1
 source=source.replace(first,'''                entropy_rmpad, log_probs = _e1_token_statistics(
                    logits_rmpad, input_ids_rmpad_rolled, self.compute_entropy_from_logits)''',1)
 second="                log_probs = logprobs_from_logits(logits, micro_batch['responses'])\n                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)"
 assert source.count(second)==1
 return source.replace(second,"                entropy, log_probs = _e1_token_statistics(logits, micro_batch['responses'], verl_F.entropy_from_logits)",1)
