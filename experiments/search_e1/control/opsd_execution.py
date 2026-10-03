"""Apply execution-only OPSD guards to a copied, verified source file."""
import ast


HELPERS = '''def _opsd_collective_reject(local_failed, world, message):
    failed = torch.tensor(int(local_failed), device='cuda', dtype=torch.int64)
    if world > 1:
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise FloatingPointError(message)


def _opsd_global_validation(agent, validation, rank, world):
    local = validation[rank::world]
    local_em = evaluate(agent, local) if local else 0.
    counts = torch.tensor([round(local_em * len(local)), len(local)],
                          device='cuda', dtype=torch.int64)
    if world > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    if counts[1].item() == 0:
        raise RuntimeError('Validation must be nonempty.')
    return counts[0].item() / counts[1].item()


'''


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('OPSD source changed or already patched: ' + before.strip())
    return source.replace(before, after, 1)


def _function_text(source, name):
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def patch_opsd(source):
    """Preserve pair_loss and original trajectory mean/AdamW update semantics.

    Accept the archive or its existing finite-check overlay. Existing local
    raises are replaced so every rank reaches the same rejection collective.
    No files, GPU state, or process groups are touched here.
    """
    if 'def _opsd_collective_reject(' in source:
        raise ValueError('OPSD execution overlay already applied')
    original_pair_loss = _function_text(source, 'pair_loss')
    old_loss_guard = ("                if not torch.isfinite(loss).all():\n"
                      "                    raise FloatingPointError('Nonfinite OPSD loss')\n")
    if old_loss_guard in source:
        source = _replace_once(source, old_loss_guard, '')
    old_gradient_guard = ("            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)\n"
                          "            if not torch.isfinite(grad_norm).all():\n"
                          "                raise FloatingPointError('Nonfinite OPSD gradient; optimizer update refused')")
    if old_gradient_guard in source:
        source = _replace_once(source, old_gradient_guard,
                               '            torch.nn.utils.clip_grad_norm_(trainable, 1.)')
    source = _replace_once(source, 'def main():', HELPERS + 'def main():')
    source = _replace_once(source, 'best = evaluate(agent, validation) if rank == 0 else 0.',
                           'best = _opsd_global_validation(agent, validation, rank, world)')
    source = _replace_once(source, 'em = evaluate(agent, validation) if rank == 0 else 0.',
                           'em = _opsd_global_validation(agent, validation, rank, world)')
    source = _replace_once(source, '            for example in local:',
                           '            local_failed = False\n            for example in local:')
    source = _replace_once(source, '                (loss / len(batch)).backward()',
                           "                if not torch.isfinite(loss).all().item():\n"
                           "                    local_failed = True\n"
                           "                    continue\n"
                           "                (loss / len(batch)).backward()")
    source = _replace_once(source, '            if world > 1:\n                for parameter in trainable:',
                           "            _opsd_collective_reject(local_failed, world, 'Nonfinite OPSD loss; all ranks refuse update')\n"
                           '            if world > 1:\n                for parameter in trainable:')
    source = _replace_once(source, '            torch.nn.utils.clip_grad_norm_(trainable, 1.)',
                           '            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)\n'
                           '            invalid_grad = not torch.isfinite(grad_norm).all().item()\n'
                           '            invalid_grad = invalid_grad or any(\n'
                           '                p.grad is not None and not torch.isfinite(p.grad).all().item() for p in trainable)\n'
                           "            _opsd_collective_reject(invalid_grad, world, 'Nonfinite OPSD gradient; all ranks refuse update')")
    ast.parse(source)
    if _function_text(source, 'pair_loss') != original_pair_loss:
        raise AssertionError('Execution overlay changed pair_loss')
    return source
