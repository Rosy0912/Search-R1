"""Expose the actual shuffled permutation/cursor for remote full-state saving."""

import torch
from torch.utils.data import RandomSampler, Sampler


class RecordedRandomSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.permutation = None
        self.cursor = 0
        self.epoch = -1

    def __iter__(self):
        if self.permutation is None or self.cursor == len(self.permutation):
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

    def load_state_dict(self, state):
        permutation = state['permutation']
        if state['dataset_size'] != len(self.dataset) or sorted(permutation) != list(range(len(self.dataset))):
            raise ValueError('Restored sampler question domain differs')
        if type(state['cursor']) is not int or not 0 <= state['cursor'] <= len(permutation):
            raise ValueError('Invalid restored sampler cursor')
        self.permutation = permutation[:]; self.cursor = state['cursor']; self.epoch = state['epoch']
