#!/usr/bin/env python3

import os
from argparse import ArgumentParser

import torch
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

import utils
from model import DelhateEnsemble


def main():
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    try:
        _, dataset, embed, model_type, model_name = args.model_path.split('/')
        model = DelhateEnsemble.load_model(args.model_path)
    except FileNotFoundError:
        raise

    embedding, dim = utils.load_embedding(model.embed_corpus)

    test_data = utils.load_dataset(args.dataset, 'test', embedding, labeled=True, pad=model.seq_length)

    model.to(device)
    y_pred, y_true = model.evaluate(test_data, device=device)

    print('pred:', Counter(y_pred))
    print('true:', Counter(y_true))

    report = classification_report(y_true, y_pred, target_names=['H', 'O', 'N'], digits=3)
    conf_mat = confusion_matrix(y_true, y_pred)

    model_name = model_name.replace('.pt', '')
    out_path = f'metrics/{dataset.upper()}/{embed}/{model_type}'
    os.makedirs(out_path, exist_ok=True)

    with open(f'{out_path}/{model_name}_{args.dataset}.txt', 'w') as f:
        f.write(report)
        f.write('\n')
        f.write('\n'.join('  '.join(str(x) for x in y) for y in conf_mat))
        f.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('dataset', type=str,
                        help='Dataset on which to evaluate the model. Options are hon/olid/combined/gab')

    parser.add_argument('model_path', type=str,
                        help='File path for the trained model. See train_model.py for training and saving a model.')

    parser.add_argument('--use-gpu', type=bool, default=True,
                        help='Use GPU for model training. Default is True.')

    args = parser.parse_args()
    main()
