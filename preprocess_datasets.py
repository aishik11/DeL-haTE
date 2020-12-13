#!/usr/bin/env python3

import json
import pandas as pd

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from argparse import ArgumentParser
import utils


def main():
    hon_data = pd.read_csv('data/hon/HON_labeled_data.csv',
                           header=0, index_col=0)
    hon_data.rename(columns={'class': 'label'}, inplace=True)
    hon_data.index = 'h_' + hon_data.index.astype(str)

    olid_data = pd.read_csv('data/olid/olid-training-v1.0.tsv',
                            sep='\t', header=0, index_col=0)
    olid_data['label'] = olid_data.apply(flatten_labels, axis=1)
    olid_data.index = 'o_' + olid_data.index.astype(str)

    with open('data/gab/gab_posts_random_sample_100k.json') as f:
        gab_data = json.load(f)

    stemmer = PorterStemmer()
    addl_stops = ['mt', 'rt']

    hate_words, offensive_words, positive_words = utils.load_word_lists(stemmer)

    print('Processing HON data...')
    hon_train, hon_test = process_csv_data(hon_data,
                                           stemmer,
                                           hate_words,
                                           offensive_words,
                                           positive_words,
                                           stops=addl_stops)

    print('Processing OLID data...')
    olid_train, olid_test = process_csv_data(olid_data,
                                             stemmer,
                                             hate_words,
                                             offensive_words,
                                             positive_words,
                                             stops=addl_stops)

    combined_train = pd.concat([hon_train, olid_train])
    combined_test = pd.concat([hon_test, olid_test])

    print('Processing Gab data...')
    gab_train = process_gab_data(gab_data,
                                 stemmer,
                                 hate_words,
                                 offensive_words,
                                 positive_words,
                                 stops=addl_stops)

    cols = ['text', 'tokens', 'bounds', 'label']
    datasets = {
        'hon': (hon_train[cols], hon_test[cols]),
        'olid': (olid_train[cols], olid_test[cols]),
        'combined': (combined_train[cols], combined_test[cols])
    }

    # Save processed text, bounds, and label to json
    print('Saving datasets to JSON...')
    for name, (train, test) in datasets.items():
        with open(f'data/{name}/{name}_train.json', 'w') as f:
            json.dump(train.to_dict('index'), f, indent=2)
        with open(f'data/{name}/{name}_test.json', 'w') as f:
            json.dump(test.to_dict('index'), f, indent=2)

    with open(f'data/gab/gab_train.json', 'w') as f:
        json.dump(gab_train, f, indent=2)


def flatten_labels(row):
    if row['subtask_a'] == 'NOT':
        return 2
    elif row['subtask_b'] == 'UNT':
        return 1
    else:
        return 0 if row['subtask_c'] == 'GRP' else 1


def process_csv_data(data, stemmer, hate_words, offensive_words, positive_words, stops=None):
    # Clean text
    data['text'] = data['tweet'].apply(utils.clean_text, stemmer=stemmer, stops=stops)
    data['tokens'] = data['text'].apply(word_tokenize)

    # Calculate class bounds for weak supervision
    data['bounds'] = data['tokens'].apply(utils.calculate_bounds,
                                          hate_words=hate_words,
                                          offensive_words=offensive_words,
                                          positive_words=positive_words)

    # Split dataset into train, val, and test sets
    test_data = data.sample(frac=args.test_frac)
    data.drop(test_data.index, inplace=True)

    return data, test_data


def process_gab_data(data, stemmer, hate_words, offensive_words, positive_words, stops=None):
    placeholders = {'HASHTAGHERE', 'MENTIONHERE'}
    clean_data = {}
    for post in data:
        text = utils.clean_text(post['body'], stemmer, stops=stops)
        words = set(word_tokenize(text))

        if text and not words.issubset(placeholders):
            clean_data[f'g_{post["post_id"]}'] = {
                'text': text,
                'tokens': word_tokenize(text),
                'bounds': utils.calculate_bounds(text, hate_words, offensive_words, positive_words)
            }

    return clean_data


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--val-frac', type=float, default=0.1,
                        help='Fraction of dataset to use as validation set. Default is 0.1.')

    parser.add_argument('--test-frac', type=float, default=0.1,
                        help='Fraction of dataset to use as test set. Default is 0.1.')

    args = parser.parse_args()
    main()
