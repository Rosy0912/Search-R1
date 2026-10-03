"""Physical subbatches preserving the upstream logical token-mean microbatch.

Only the R1 configuration arm enables this adapter. Optimizer boundaries and
logical gradient accumulation remain unchanged; every physical loss receives
its fraction of the logical microbatch's masked tokens.
"""
def patch_actor(source):
    source = 'import os\n' + source
    before, update = source.split("    def update_policy", 1)
    update = update.replace("        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error", "        physical = int(os.environ.get('E1_PHYSICAL_MICRO', '0'))\n        if physical and (self.config.use_dynamic_bsz or os.environ.get('SEARCH_E1_TRAJECTORY_BATCHING') == '1'):\n            raise ValueError('Physical token adapter requires static upstream token reduction')\n        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error")
    source = before + '    def update_policy' + update
    source = source.replace('            micro_batches = batch.split(micro_batch_size)', "            score_cap = int(os.environ.get('E1_PHYSICAL_SCORE', '0'))\n            micro_batches = batch.split(min(micro_batch_size, score_cap) if score_cap else micro_batch_size)", 1)
    start = source.index('            for data in micro_batches:')
    end = source.index('\n            grad_norm = self._optimizer_step()', start)
    block = source[start:end]
    block = block[block.index('\n')+1:]
    block = block.replace('                loss = policy_loss / self.gradient_accumulation', '                loss = policy_loss * token_fraction / self.gradient_accumulation')
    block = block.replace("                    metrics['actor/kl_loss'] = kl_loss.detach().item()", "                    logical_kl += token_fraction * kl_loss.detach().item()")
    block = block.replace('                append_to_dict(metrics, data)', "                for key, value in data.items():\n                    logical_metrics[key] = logical_metrics.get(key, 0.) + token_fraction * value")
    block = '\n'.join('    ' + line if line else '' for line in block.split('\n'))
    replacement = '''            for logical_data in micro_batches:
                logical_metrics = {}
                logical_kl = 0.
                pieces = logical_data.split(physical) if physical else [logical_data]
                response_length = logical_data['responses'].size(1)
                logical_mask = logical_data['loss_mask'] if self.config.state_masking else logical_data['attention_mask'][:, -response_length:]
                logical_tokens = logical_mask.sum().item()
                if physical and logical_tokens <= 0:
                    raise FloatingPointError('Logical microbatch has no masked tokens')
                for data in pieces:
                    piece_mask = data['loss_mask'] if self.config.state_masking else data['attention_mask'][:, -response_length:]
                    piece_tokens = piece_mask.sum().item()
                    token_fraction = piece_tokens / logical_tokens if physical else 1.
                    if physical and piece_tokens == 0:
                        continue
''' + block + '''
                append_to_dict(metrics, logical_metrics)
                if self.config.use_kl_loss:
                    metrics['actor/kl_loss'] = logical_kl
'''
    return source[:start] + replacement + source[end:]
