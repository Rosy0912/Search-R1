"""Matched local-error stages; never call after an unknown original compute error."""
import torch

MARKER = 'FRESH200_CALIBRATION_MATCHED_REFUSAL_V2'


class MatchedCalibrationRefusal(RuntimeError):
    pass


def matched_stage(stage, local_errors, *, distributed=None):
    distributed = torch.distributed if distributed is None else distributed
    # Only reached by all ranks after the same successful original call sequence.
    # Failure of this collective itself is unknown and propagates; no recovery collective.
    flag = torch.tensor([int(bool(local_errors))], device=torch.cuda.current_device(), dtype=torch.int32)
    distributed.all_reduce(flag, op=distributed.ReduceOp.MAX)
    if int(flag.item()):
        try:
            diagnostic = repr(local_errors).encode('utf-8', errors='replace')[:1024]
        except BaseException:
            diagnostic = b'Local observer diagnostic formatting failed'
        # All ranks took the same failed-MAX branch. This bounded second
        # collective is a known matched refusal journal, never recovery from an
        # unknown original or failed collective. Its own errors propagate directly.
        payload = torch.zeros(1024, dtype=torch.uint8, device=torch.cuda.current_device())
        if diagnostic:
            payload[:len(diagnostic)] = torch.tensor(list(diagnostic), dtype=torch.uint8, device=payload.device)
        gathered = [torch.empty_like(payload) for _ in range(distributed.get_world_size())]
        distributed.all_gather(gathered, payload)
        rank_errors = {rank: bytes(row.cpu().tolist()).rstrip(b'\0').decode('utf-8', errors='replace')
                       for rank, row in enumerate(gathered)}
        raise MatchedCalibrationRefusal(MARKER + '/' + stage + '; rank_errors=' + repr(rank_errors))

    return dict(stage=stage, matched=True)


def is_matched_refusal(error):
    """A typed cause chain is required; a matching message alone is insufficient."""
    pending, seen = [error], set()
    while pending and len(seen) < 16:
        item = pending.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, MatchedCalibrationRefusal):
            return True
        for key in ('cause', '__cause__', '__context__'):
            try:
                pending.append(getattr(item, key, None))
            except BaseException:
                pass
    return False
