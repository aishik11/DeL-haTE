#!/usr/bin/env python3

import os
import re
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import torch
from torchtext.vocab import GloVe, FastText, Vectors
from dataset import PostDataset


def clean_text(text, stemmer, stops=None):
    text = text.lower()

    # Remove urls
    www_exp = r'www.[^ ]+'
    http_exp = r'https?[^\s]+'
    text = re.sub('|'.join((www_exp, http_exp)), '', text)

    # Remove encoded Unicode symbols (removes emoticons etc.)
    text = re.sub(r'&.*?;', ' ', text)

    # Replace user mentions and hashtags with placeholder
    text = re.sub(r'#([\w\-]+)', r' HASHTAGHERE \1', text)
    text = re.sub(r'@[\w\-]+', r'MENTIONHERE ', text)

    # Remove non-letter chars
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove stop words and apply stemming
    nltk_stops = set(stopwords.words('english'))
    stops = set(stops).union(nltk_stops) if stops else nltk_stops
    text = ' '.join(
        stemmer.stem(word)
        for word in word_tokenize(text)
        if word not in stops
    )

    # Replace stemmed mention and hashtag placeholders
    text = re.sub(r'mentionher', 'MENTIONHERE', text)
    text = re.sub(r'hashtagher', 'HASHTAGHERE', text)

    return text.strip()


def load_word_lists(stemmer):
    with open('data/word_lists/hate_words.txt') as f:
        hate_words = set(stemmer.stem(word.strip()) for word in f)
    with open('data/word_lists/offensive_words.txt') as f:
        offensive_words = set(stemmer.stem(word.strip()) for word in f)
    with open('data/word_lists/positive_words.txt') as f:
        positive_words = set(stemmer.stem(word.strip()) for word in f)

    return hate_words, offensive_words, positive_words


def load_dataset(dataset, split, embedding, labeled, pad):
    with open(f'data/{dataset}/{dataset}_{split}.json') as f:
        data = json.load(f)

    return PostDataset(data, embedding, labeled=labeled, padded_seq=pad)


def load_embedding(embed_corpus):
    corpora = ['glove_twitter', 'glove_commoncrawl', 'fasttext_wiki', 'fasttext_commoncrawl', 'word2vec']
    dim = 300

    os.makedirs('data/glove', exist_ok=True)
    os.makedirs('data/fast_text', exist_ok=True)
    os.makedirs('data/word2vec', exist_ok=True)

    if embed_corpus == 'glove_twitter':
        # GloVe trained on Twitter corpus
        embedding = GloVe(name='twitter.27B', dim=200, cache='data/glove/')
        dim = 200
    elif embed_corpus == 'glove_commoncrawl':
        # GloVe trained on Common Crawl corpus
        embedding = GloVe(name='42B', dim=300, cache='data/glove/')
    elif embed_corpus == 'fasttext_wiki':
        # FastText trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset
        embedding = FastText(language='en', cache='data/fast_text/')
    elif embed_corpus == 'fasttext_commoncrawl':
        # FastText trained on Common Crawl corpus
        embedding = Vectors(name='crawl-300d-2M.vec',
                            url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip',
                            cache='data/fast_text/')
    elif embed_corpus == 'word2vec':
        # Word2Vec trained on Google New corpus
        name = 'GoogleNews-vectors-negative300.txt'
        if os.path.isfile(f'data/word2vec/{name}.pt'):
            embedding = Vectors(name=name,
                                cache='data/word2vec/')
        else:
            raise FileNotFoundError(('No torchtext formatted word2vec vectors file found. '
                                     'See load_word2vec.py to create the necessary pt file. Requires gensim.'))
    else:
        raise ValueError(f'Invalid pre-trained word embedding vectors. Options are {"/".join(corpora)}.')

    return embedding, dim


def weak_loss(output, bounds, weight=(1., 1., 1.)):
    weight = torch.tensor(weight, device=output.device)

    # Convert class scores to probabilities
    probs = torch.softmax(output, dim=1)

    LBs = bounds[:, :, 0]
    UBs = bounds[:, :, 1]

    ones = torch.ones_like(probs, device=output.device)

    # Loss is applied if output probability falls outside the heuristic bounds
    min_lb = torch.min(ones, (ones + probs - LBs))
    min_ub = torch.min(ones, (ones + UBs - probs))

    nll = -torch.log(min_lb) - torch.log(min_ub)

    loss = torch.sum(nll, dim=0)
    loss *= weight

    # Sum per-class loss
    return torch.sum(loss)


def calculate_bounds(tokens, hate_words, offensive_words, positive_words):
    # TODO: Simplify bounds logic
    placeholders = {'HASHTAGHERE', 'MENTIONHERE'}
    unique_words = set(tokens) - placeholders
    num_unique = len(unique_words)

    post_hate = unique_words.intersection(hate_words)
    post_offensive = unique_words.intersection(offensive_words) - post_hate
    post_positive = unique_words.intersection(positive_words)

    num_hate = len(post_hate)
    num_offensive = len(post_offensive)
    num_positive = len(post_positive)
    try:
        if post_hate:
            hate_LB = min(1., (num_hate + num_offensive) / num_unique)
            hate_UB = 1.

            offensive_LB = 0.5 * num_offensive / num_unique
            offensive_UB = 1 - (num_hate + num_positive) / num_unique

            neither_LB = 0.
            neither_UB = max(0., 1 - (num_hate + num_offensive) / num_unique)
        elif post_offensive:
            hate_LB = 0.
            hate_UB = 1 - (num_offensive + num_positive) / num_unique

            offensive_LB = (num_offensive + 0.5 * num_positive) / num_unique
            offensive_UB = 1 - 0.5 * num_positive / num_unique

            neither_LB = 0.5 * num_positive / num_unique
            neither_UB = 1 - (num_hate + num_offensive) / num_unique
        else:
            hate_LB = 0.
            hate_UB = 1 - num_positive / num_unique

            offensive_LB = 0.
            offensive_UB = 1 - num_positive / num_unique

            neither_LB = num_positive / num_unique
            neither_UB = 1.
    except ZeroDivisionError:
        return (0., 1.), (0., 1.), (0., 1.)

    return (hate_LB, hate_UB), (offensive_LB, offensive_UB), (neither_LB, neither_UB)


def parse_train_args():
    parser = ArgumentParser()

    parser.add_argument('dataset', type=str,
                        help='Dataset on which to train the model. Options are hon/olid/combined/gab')

    parser.add_argument('model_name', type=str,
                        help='Model name. The model attributes and state dict will be saved as <model-name>.pt.')

    corpora = ['glove_twitter', 'glove_commoncrawl', 'fasttext_wiki', 'fasttext_commoncrawl', 'word2vec']
    parser.add_argument('--embed_corpus', type=str, default='glove_commoncrawl',
                        help=f'Pre-trained word embedding model/corpus to use. '
                             f'Options are {"/".join(corpora)}. Default is glove_commoncrawl')

    parser.add_argument('--pad', default=50,
                        help='Length of padded input to pass to the model. Default is 50.')

    parser.add_argument('--n_models', type=int, default=5,
                        help='Number of models in the ensemble. Default is 5.')

    parser.add_argument('--n_filters', type=int, default=32,
                        help='Number of filters in the CNN layer. Default is 32.')

    parser.add_argument('--filter_width', type=int, default=5,
                        help='Width of each filter in the CNN layer. Default is 5.')

    parser.add_argument('--pool_size', type=int, default=4,
                        help='Kernel size for max pooling. Default is 4.')

    parser.add_argument('--n_hidden', type=int, default=100,
                        help='Size of the hidden layers. Default is 100.')

    parser.add_argument('--rnn_type', type=str, default='gru',
                        help='Type of RNN to use in the model. Options are gru/lstm/None. Default is gru')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability during model training. Default is 0.2.')

    parser.add_argument('--use_val', default=True, action='store_true',
                        help='Default True. Use validation data to evaluate model after each training epoch.')

    parser.add_argument('--early_stop', default=False, action='store_true',
                        help='Default False. Use early stopping during training to save model state at the epoch '
                             'with minimum validation loss.')

    parser.add_argument('--use_weak_loss', dest='weak_loss', default=False, action='store_true',
                        help='Default False. Use heuristic class bounds and weak loss for training. '
                             'When False, use true class labels and cross entropy loss.')

    parser.add_argument('--class_weight', type=float, default=[1., 1., 1.], nargs='+',
                        help='Per-class weight for weak loss calculation. Default (1., 1., 1.).')

    parser.add_argument('--learn_rate', type=float, default=1e-3,
                        help='Learning rate for Adam optimizer during training. Default is 1e-3.')

    parser.add_argument('--n_samples', default=1000,
                        help='Number of observations to sample from each class at each training epoch. '
                             'May be an integer or "all" to set sample size equal to the number of observations '
                             'in the smallest class. Default is 1000.')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use during training. Default is 32.')

    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs. Default is 20.')

    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Use GPU for model training. Default is True.')

    return parser.parse_args()


def plot_loss(dataset, train_losses, val_losses, early_stop=False):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Training loss')

    if val_losses:
        plt.plot(val_losses, label='Validation loss')
        if early_stop:
            plt.vlines(val_losses.index(min(val_losses)),
                       min([*train_losses, *val_losses]), max([*train_losses, *val_losses]),
                       linestyles='dashed')

    plt.legend(frameon=False)
    plt.title(f'Model training loss on {dataset.upper()} data')
    plt.xlabel('Epoch')
    plt.ylabel('Avg. Loss per Batch')
    plt.show()
    plt.close()
