#!/usr/bin/env python3

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import PostDataset

import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict


class DelhateEnsemble(nn.Module):
    def __init__(
            self,
            n_models,
            seq_length,
            embed_corpus,
            embed_dim,
            n_classes,
            n_filters,
            filter_width,
            pool_size=4,
            n_hidden=100,
            rnn_type='gru',
            dropout=0.2
    ):
        super(DelhateEnsemble, self).__init__()

        self.n_models = n_models
        self.seq_length = seq_length
        self.embed_corpus = embed_corpus
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.pool_size = pool_size
        self.n_hidden = n_hidden
        self.dropout = dropout

        if rnn_type is None or rnn_type.casefold() == 'none':
            self.rnn_type = None
        elif rnn_type.casefold() == 'lstm':
            self.rnn_type = 'lstm'
        elif rnn_type.casefold() == 'gru':
            self.rnn_type = 'gru'
        else:
            raise ValueError('Invalid RNN type selected. Options are gru/lstm/None')

        self.sub_models = nn.ModuleList()
        for _ in range(self.n_models):
            self.sub_models.append(
                ComponentModel(
                    self.seq_length,
                    self.embed_corpus,
                    self.embed_dim,
                    self.n_classes,
                    self.n_filters,
                    self.filter_width,
                    self.pool_size,
                    self.n_hidden,
                    self.rnn_type,
                    self.dropout,
                )
            )

    # TODO: Add reset_parameters() function

    def forward(self, x):
        # Shape: batch size X num models X num classes
        logits = torch.empty((x.shape[0], self.n_models, self.n_classes))
        for i, model in enumerate(self.sub_models):
            logits[:, i, :] = model(x.clone())

        return logits

    def evaluate(self, data, batch_size=64, device='cpu'):
        data_loader = DataLoader(data, batch_size=batch_size)
        y_pred, y_true = [], []

        self.eval()
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)

                logits = self(X)
                preds, _ = torch.softmax(logits, dim=2).argmax(dim=2).mode(dim=1)

                y_pred.extend(preds.tolist())
                y_true.extend(y.tolist())

        return y_pred, y_true

    def train_models(
            self,
            train_data,
            loss_fn,
            lr=1e-3,
            n_samples=1000,
            use_val=False,
            early_stop=False,
            batch_size=24,
            EPOCHS=20,
            device='cpu'
    ):
        # TODO: Parallelize?
        losses = defaultdict(list)
        self.to(device)
        for i, model in enumerate(self.sub_models):
            desc = f'Training model {i}'

            optimizer = optim.Adam(model.parameters(), lr=lr)

            val_data = None
            if use_val or early_stop:
                val_size = int(0.1 * len(train_data))
                train_sub, val_data = random_split(train_data, [(len(train_data) - val_size), val_size])

                # Create training Dataset for use class-balanced sampling
                train_posts = {
                    train_data.post_ids[p_id]: train_data.posts[train_data.post_ids[p_id]]
                    for p_id in train_sub.indices
                }

                train_sub = PostDataset(posts=train_posts,
                                        embedding=train_data.embedding,
                                        labeled=train_data.labeled,
                                        padded_seq=train_data.padded_seq)
            else:
                train_sub = train_data

            t_loss, v_loss = model.train_model(train_sub,
                                               loss_fn,
                                               optimizer,
                                               val_data,
                                               n_samples,
                                               early_stop,
                                               batch_size,
                                               EPOCHS,
                                               device,
                                               desc=desc)

            losses['train'].append(t_loss)
            losses['val'].append(v_loss)

        return losses

    def tune_models(
            self,
            data,
            lr=5e-4,
            batch_size=16,
            EPOCHS=10,
            device='cpu'
    ):
        self.to(device)
        losses = []
        for model in self.sub_models:
            tune_loss = model.tune_model(data, lr, batch_size, EPOCHS, device)
            losses.append(tune_loss)

        return losses

    def get_model_attributes(self):
        return {
            'n_models': self.n_models,
            'seq_length': self.seq_length,
            'embed_corpus': self.embed_corpus,
            'embed_dim': self.embed_dim,
            'n_classes': self.n_classes,
            'n_filters': self.n_filters,
            'filter_width': self.filter_width,
            'pool_size': self.pool_size,
            'n_hidden': self.n_hidden,
            'dropout': self.dropout,
            'rnn_type': self.rnn_type
        }

    def save_model(self, path):
        torch.save((self.get_model_attributes(), self.state_dict()), path)

    @classmethod
    def load_model(cls, model_path):
        model_attr, state_dict = torch.load(model_path)
        model = cls(
            n_models=model_attr['n_models'],
            seq_length=model_attr['seq_length'],
            embed_corpus=model_attr['embed_corpus'],
            embed_dim=model_attr['embed_dim'],
            n_classes=model_attr['n_classes'],
            n_filters=model_attr['n_filters'],
            filter_width=model_attr['filter_width'],
            pool_size=model_attr['pool_size'],
            n_hidden=model_attr['n_hidden'],
            rnn_type=model_attr['rnn_type'],
            dropout=model_attr['dropout']
        )
        model.load_state_dict(state_dict)

        return model


class ComponentModel(nn.Module):
    def __init__(
            self,
            seq_length,
            embed_corpus,
            embed_dim,
            n_classes,
            n_filters,
            filter_width,
            pool_size,
            n_hidden,
            rnn_type,
            dropout
    ):
        super(ComponentModel, self).__init__()

        self.seq_length = seq_length
        self.embed_corpus = embed_corpus
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.pool_size = pool_size
        self.n_hidden = n_hidden
        self.rnn_type = rnn_type
        self.dropout = dropout

        if self.rnn_type is None:
            rnn = None
        elif self.rnn_type == 'lstm':
            rnn = nn.LSTM(self.embed_dim // self.pool_size, self.n_hidden, batch_first=True)
        elif self.rnn_type == 'gru':
            rnn = nn.GRU(self.embed_dim // self.pool_size, self.n_hidden, batch_first=True)
        else:
            raise ValueError('Invalid RNN type selected. Options are gru/lstm/None')

        self.features = nn.Sequential(OrderedDict([
            ('conv1d', nn.Conv1d(self.seq_length, self.n_filters, self.filter_width,
                                 padding=(self.filter_width - 1) // 2)),
            ('batchnorm', nn.BatchNorm1d(self.n_filters)),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool1d(self.pool_size))
        ]))

        if rnn:
            self.features.add_module('rnn', rnn)

        self.globalpool = nn.AdaptiveMaxPool1d(self.n_hidden)

        self.classifier = nn.Sequential(OrderedDict([
            ('linear_h', nn.Linear(self.n_hidden, self.n_hidden // 2)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=self.dropout)),
            ('linear_out', nn.Linear(self.n_hidden // 2, self.n_classes))
        ]))

    def forward(self, x):
        if self.rnn_type is None:
            x = self.features(x)
            x = x.view(-1, 1, self.n_filters * self.embed_dim // self.pool_size)
            # h = torch.zeros(x.shape[0], 1, self.n_hidden, device=x.device)
        else:
            if self.rnn_type == 'lstm':
                x, (h, c) = self.features(x)
            elif self.rnn_type == 'gru':
                x, h = self.features(x)
            else:
                raise ValueError

            x = x.contiguous().view(-1, 1, self.n_filters * self.n_hidden)
            h = torch.transpose(h, 0, 1)

        x = self.globalpool(x)

        # Include hidden state for classifier?
        # If so, need to change classifier input shape
        # x = torch.cat([x, h], dim=-1)

        x = self.classifier(x)

        return x.squeeze()

    def evaluate(self, data, criterion, batch_size=64, device='cpu'):
        data_loader = DataLoader(data, batch_size=batch_size)

        y_pred, y_true = [], []
        val_loss = 0

        self.eval()
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)

                logits = self(X)
                val_loss += criterion(logits, y).item()

                preds = torch.softmax(logits, dim=1).argmax(dim=1)

                y_pred.extend(preds.tolist())
                y_true.extend(y.tolist())

        return y_pred, y_true, (val_loss / len(data_loader))

    def train_model(
            self,
            train_data,
            criterion,
            optimizer,
            val_data=None,
            n_samples=1000,
            early_stop=False,
            batch_size=24,
            EPOCHS=20,
            device='cpu',
            desc=''
    ):

        best_model, best_epoch = None, -1
        val_loss_min = np.Inf

        if early_stop:
            assert val_data

        # If using heuristic bounds and weak loss, use entire training data at each epoch (no class balancing)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        self.to(device)
        train_losses, val_losses = [], []
        for epoch in tqdm(range(EPOCHS), desc=desc):
            self.train()
            running_loss = 0

            # If using labeled data and cross entropy loss,
            # sample training data at each epoch to create a class-balanced training subset
            if train_data.labeled:
                train_sample = train_data.sample_classes(n_samples=n_samples)
                train_loader = DataLoader(train_sample, batch_size=batch_size, shuffle=True)

            for X, y in train_loader:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()

                logits = self(X)
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_losses.append(running_loss / len(train_loader))

            if val_data:
                _, _, val_loss = self.evaluate(val_data, criterion, batch_size=batch_size, device=device)
                val_losses.append(val_loss)

                if early_stop and val_loss <= val_loss_min:
                    val_loss_min = val_loss
                    best_model = self.state_dict()
                    best_epoch = epoch

        if early_stop:
            self.load_state_dict(best_model)

        return train_losses, val_losses

    def tune_model(
            self,
            data,
            lr,
            batch_size=32,
            EPOCHS=20,
            device='cpu'
    ):
        train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

        # Freeze feature extraction component
        for param in self.features.parameters():
            param.requires_grad = False

        criterion = F.cross_entropy
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(device)
        losses = []
        for epoch in tqdm(EPOCHS):
            self.train()
            running_loss = 0

            for X, y in train_loader:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()

                logits = self(X)
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            losses.append(running_loss / len(train_loader))

        return losses
