#!/usr/bin/env python3

import os
import requests
import shutil

import numpy as np
from tqdm import tqdm

import torch
from gensim.models import KeyedVectors

"""
    Hack to convert word2vec binary file to torchtext format.
    
    Torchtext vocab.Vectors cannot handle the word2vec binary format.
    Load vectors into gensim and manually convert to format for torchtext.
"""

name = 'GoogleNews-vectors-negative300'
fname = f'{name}.bin.gz'
w2v_path = f'data/word2vec'
fpath = f'{w2v_path}/{fname}'
out_path = f'{w2v_path}/{name}.txt.pt'

os.makedirs(w2v_path, exist_ok=True)

if not os.path.isfile(fpath):
    # Permanent download link?
    # 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'

    url = f'https://s3.amazonaws.com/dl4j-distribution/{fname}'
    print(f'No word2vec vectors file found. Downloading from\n{url}')

    # TODO: Progress bar with shutil?
    with requests.get(url, stream=True) as r, open(fpath, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

print('Loading pre-trained vectors into Gensim...')
embed = KeyedVectors.load_word2vec_format(fpath, binary=True, encoding='ISO-8859-1', unicode_errors='ignore')

itos = []
stoi = {}
dim = 300
vectors = torch.empty((len(embed.vocab), dim))
print('Converting to torchtext format...\n')
for i, token in enumerate(tqdm(embed.vocab)):
    itos.append(token)
    stoi[token] = i
    vectors[i] = torch.from_numpy(np.array(embed[token]))

del embed  # Remove gensim model from memory

print(f'Saving to {out_path}')
torch.save((itos, stoi, vectors, dim), out_path)
