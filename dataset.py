#!/usr/bin/env python3

import warnings
from collections import defaultdict
from random import sample

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


class PostDataset(Dataset):
    def __init__(self, posts, embedding, labeled=True, padded_seq='max'):
        self.posts = posts
        self.post_ids = list(posts.keys())
        self.labeled = labeled

        self.embedding = embedding

        self.max_seq_length = max(len(self.posts[ID]['tokens']) for ID in self.posts)

        self.padded_seq = None
        self.set_padding(padded_seq)

        if self.labeled:
            self.n_classes = max(self.posts[ID]['label'] for ID in self.posts) + 1

            self.label_idx = defaultdict(list)  # Dict of post ids per class
            for i, post in enumerate(self.post_ids):
                self.label_idx[self.posts[post]['label']].append(i)
        else:
            self.n_classes = len(self.posts[self.post_ids[0]]['bounds'])

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        ID = self.post_ids[idx]

        # Get embeddings vectors from word tokens
        X = self.embedding.get_vecs_by_tokens(self.posts[ID]['tokens'])

        # Add zero left padding to standardize word embedding matrix size
        X = F.pad(X, [0, 0, (self.padded_seq - X.shape[0]), 0])

        if self.labeled:
            y = self.posts[ID]['label']
        else:
            y = torch.tensor(self.posts[ID]['bounds'], dtype=torch.float)

        return X, y

    def set_padding(self, pad):
        if pad == 'max':
            pad = self.max_seq_length
        else:
            try:
                pad = int(pad)
            except TypeError as e:
                raise Exception('Invalid padding. Padding must be an integer or '
                                '"max" which will set padding to the length of max input sequence.') from e

        self.padded_seq = pad

    def sample_classes(self, n_samples=1000):
        assert self.labeled

        min_class = min(len(p_ids) for p_ids in self.label_idx.values())
        if n_samples == 'all':
            n_samples = min_class
        elif n_samples > min_class:
            warnings.warn(('Number of sample greater than examples of smallest class. '
                           f'Setting n_samples = {min_class:,}'))
            n_samples = min_class
        elif type(n_samples) is not int:
            raise TypeError('Invalid n_samples. n_samples must be an integer or '
                            '"all" which will set n_samples to the number of examples in the smallest class.')

        sample_idx = [
            p_i
            for p_indices in self.label_idx.values()
            for p_i in sample(p_indices, k=n_samples)
        ]

        return Subset(self, sample_idx)
