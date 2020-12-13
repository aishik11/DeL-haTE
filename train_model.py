#!/usr/bin/env python3

import os

import torch
import torch.nn.functional as F

import utils
from model import DelhateEnsemble


def main():
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    rnn_str = args.rnn_type if args.rnn_type else 'cnn'
    weak_str = '_weak' if args.weak_loss else ''

    out_path = f'models/{args.dataset}/{args.embed_corpus}/delhate_{rnn_str}{weak_str}'
    os.makedirs(out_path, exist_ok=True)

    embedding, dim = utils.load_embedding(args.embed_corpus)

    labeled = not args.weak_loss

    train_data = utils.load_dataset(args.dataset, 'train', embedding, labeled, args.pad)

    model = DelhateEnsemble(
        n_models=args.n_models,
        seq_length=train_data.padded_seq,
        embed_corpus=args.embed_corpus,
        embed_dim=dim,
        n_classes=train_data.n_classes,
        n_filters=args.n_filters,
        filter_width=args.filter_width,
        pool_size=args.pool_size,
        n_hidden=args.n_hidden,
        rnn_type=args.rnn_type,
        dropout=args.dropout
    )

    if args.weak_loss:
        loss_fn = lambda x, y: utils.weak_loss(x, y, weight=args.class_weight)
    else:
        loss_fn = F.cross_entropy

    model.train_models(
        train_data,
        loss_fn=loss_fn,
        lr=args.learn_rate,
        n_samples=args.n_samples,
        use_val=args.use_val,
        early_stop=args.early_stop,
        batch_size=args.batch_size,
        EPOCHS=args.epochs,
        device=device
    )

    model.save_model(f'{out_path}/{args.model_name}.pt')


if __name__ == '__main__':
    args = utils.parse_train_args()
    main()
