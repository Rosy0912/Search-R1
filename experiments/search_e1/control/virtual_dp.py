"""Eight physical workers execute the original four masked-token workers."""
import torch
import torch.distributed as dist


def stripe_partitions(partitions, logical_micro=16):
    """Keep each original logical minibatch, assigning its two halves to a pair."""
    if len(partitions) != 4 or logical_micro % 2:
        raise ValueError('Expected four logical workers and an even microbatch')
    half = logical_micro // 2
    order = []
    for partition in partitions:
        if len(partition) % logical_micro:
            raise ValueError('Logical partition must contain complete microbatches')
        for side in range(2):
            for offset in range(0, len(partition), logical_micro):
                order.extend(partition[offset + side * half:offset + (side + 1) * half])
    return order


def make_pair_group():
    if dist.get_world_size() != 8:
        raise ValueError('Virtual DP requires exactly eight physical workers')
    selected = None
    for first in range(0, 8, 2):
        group = dist.new_group(ranks=[first, first + 1])
        if dist.get_rank() in (first, first + 1):
            selected = group
    return selected


def pair_tokens(local_tokens, group):
    device = torch.device('cuda', torch.cuda.current_device()) if dist.get_backend() == 'nccl' else torch.device('cpu')
    total = torch.tensor(local_tokens, dtype=torch.int64, device=device)
    dist.all_reduce(total, group=group)
    invalid = (total <= 0).to(torch.int32)
    # Reject together before FSDP collectives, including otherwise valid pairs.
    dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
    if invalid.item():
        raise FloatingPointError('A virtual logical microbatch has no masked tokens')
    return total.item()
