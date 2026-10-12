"""Expose the actual shuffled permutation/cursor for remote full-state saving."""

import torch
from torch.utils.data import RandomSampler, Sampler


class RecordedRandomSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.permutation = None
        self.cursor = 0
        self.epoch = -1
        self._resume_existing_epoch = False

    def __iter__(self):
        resume = self._resume_existing_epoch
        self._resume_existing_epoch = False
        if self.permutation is None or (self.cursor == len(self.permutation) and not resume):
            # Delegate to the existing torch sampler rather than inventing a
            # different question order. DataLoader has already consumed its seed.
            self.permutation = list(RandomSampler(self.dataset))
            self.cursor = 0
            self.epoch += 1
        while self.cursor < len(self.permutation):
            index = self.permutation[self.cursor]
            self.cursor += 1
            yield index

    def __len__(self): return len(self.dataset)

    def state_dict(self):
        if self.permutation is None:
            raise ValueError('Sampler has not actually produced a permutation')
        return dict(permutation=self.permutation[:], cursor=self.cursor, epoch=self.epoch,
                    dataset_size=len(self.dataset), semantics='actual yielded positions including abandoned attempts')

    def load_state_dict(self, state, *, resume_existing_epoch=False):
        permutation = state['permutation']
        if (state['dataset_size'] != len(self.dataset) or
                any(type(value) is not int for value in permutation) or
                sorted(permutation) != list(range(len(self.dataset)))):
            raise ValueError('Restored sampler question domain differs')
        if type(state['cursor']) is not int or not 0 <= state['cursor'] <= len(permutation):
            raise ValueError('Invalid restored sampler cursor')
        if type(state['epoch']) is not int or state['epoch'] < 0:
            raise ValueError('Invalid restored sampler epoch')
        self.permutation = permutation[:]; self.cursor = state['cursor']; self.epoch = state['epoch']
        self._resume_existing_epoch = resume_existing_epoch


def restored_iterator(loader, sampler_state, driver_rng):
    """Finish the saved epoch without consuming another DataLoader base seed."""
    from search_g1.actor_probe_state import restore_rng
    if loader.num_workers != 0 or not isinstance(loader.sampler, RecordedRandomSampler):
        raise ValueError('Exact iterator restore requires the admitted synchronous sampler')
    loader.sampler.load_state_dict(sampler_state, resume_existing_epoch=True)
    iterator = iter(loader)
    # A replacement iterator consumes a base seed, even with no loader workers.
    # Undo just that extra consumption before the saved permutation is resumed.
    restore_rng(driver_rng)
    return iterator


def epoch_batches(loader, resume=None):
    """Use the restored iterator once, then create each new epoch normally."""
    epoch = 0 if resume is None else resume['epoch']
    while True:
        if resume is not None:
            iterator, start = resume['iterator'], resume['next_batch_index']
            resume = None
        else:
            iterator, start = iter(loader), 0
        yield epoch, enumerate(iterator, start=start)
        epoch += 1
