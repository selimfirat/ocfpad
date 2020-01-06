import argparse

import numpy as np
import os

import pandas as pd
import torch
from pyod.models.base import BaseDetector
from pyod.utils import invert_order
from sklearn.metrics import roc_curve, roc_auc_score
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
import h5py

from convlstm_cell import ConvLSTMCell
import torch
np.random.seed(0)
torch.manual_seed(0)

class ConvLSTMAutoencoder():

    def __init__(self, num_epochs=1, lr=0.001, cuda_idx=0, reg=0.5, kernel_size=3):
        self.cuda_idx = cuda_idx
        self.model = ConvLSTMCell(in_channels=3, out_channels=3, kernel_size=kernel_size, cuda_idx=cuda_idx).cuda(self.cuda_idx)

        self.optimizer = Adam(self.model.parameters(), lr=lr) # default lr: 1e-3

        self.num_epochs = num_epochs

        self.reg_coef = reg


    def initial_hidden(self):

        return (Variable(torch.zeros(1, 3, 224, 224)).cuda(self.cuda_idx),
                Variable(torch.zeros(1, 3, 224, 224)).cuda(self.cuda_idx))

    def partial_fit_video(self, X, y):

        h, c = self.initial_hidden()

        self.model.zero_grad()
        self.optimizer.zero_grad()

        y_pred = torch.empty(X.shape).cuda(self.cuda_idx)

        for fidx in range(X.shape[0]):
            frame = X[fidx, :, :, :]

            h, c = self.model.forward(frame, h, c)

            y_pred[fidx, :, :, :] = h

        loss = torch.mean((y_pred - y) ** 2)

        if self.reg_coef > 0:
            reg = 0

            for param in self.model.parameters():
                reg +=  (param ** 2).sum()

            loss += reg * self.reg_coef

        loss.backward(retain_graph=True)

        self.optimizer.step()

        return loss.item()
        # return h, c # to make stateful

    def fit(self, X):
        self._classes = 2

        data = [x.unsqueeze(1) for x in X]

        targets = [data[i].clone() for i in range(len(data))]

        for i in range(self.num_epochs):
            losses = []

            for X_vid, y_vid in tqdm(zip(data, targets)):
                X_vid = X_vid.cuda(self.cuda_idx)
                y_vid = y_vid.cuda(self.cuda_idx)
                loss = self.partial_fit_video(X_vid, y_vid)

                losses.append(loss)

            mean_loss = np.array(losses).mean()

            print(mean_loss)

        # self.decision_scores_ = invert_order(self.decision_function(X))

        # self._process_decision_scores()

    def decision_function(self, X):

        data = [x.unsqueeze(1) for x in X]
        targets = [data[i].clone() for i in range(len(data))]

        reconstruction_errors = np.empty((len(X), 1))

        for idx, (X_vid, y_vid) in enumerate(zip(data, targets)):

            X_vid = X_vid.cuda(self.cuda_idx)
            y_vid = y_vid.cuda(self.cuda_idx)

            h, c = self.initial_hidden()

            y_pred = torch.empty(X_vid.shape).cuda(self.cuda_idx)

            for fidx in range(X_vid.shape[0]):
                frame = X_vid[fidx, :, :, :]

                h, c = self.model.forward(frame, h, c)

                y_pred[fidx, :, :, :] = h

            reconstruction_errors[idx] = torch.mean((y_pred - y_vid) ** 2).item()

        return reconstruction_errors
