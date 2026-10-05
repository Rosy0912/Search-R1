"""A genuinely constant first-write LR; do not inherit the old zero-step warmup bug."""
def constant_scheduler(optimizer):
    from torch.optim.lr_scheduler import LambdaLR
    assert optimizer.param_groups and all(group['lr']==1e-6 for group in optimizer.param_groups)
    scheduler=LambdaLR(optimizer,lr_lambda=lambda _:1.0)
    assert all(group['lr']==1e-6 for group in optimizer.param_groups)
    return scheduler
