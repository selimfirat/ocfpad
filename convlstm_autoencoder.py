import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

from convolution_lstm import ConvLSTMCell
import torch
np.random.seed(0)
torch.manual_seed(0)

class ConvLSTMAutoencoder():

    def __init__(self, num_epochs=1, lr=0.001):
        self.model = ConvLSTMCell(input_channels=3, hidden_channels=3, kernel_size=3).cuda()

        self.optimizer = Adam(self.model.parameters(), lr=lr) # default lr: 1e-3

        self.num_epochs = num_epochs

    def partial_fit_video(self, X, y):

        h, c = self.model.init_hidden(batch_size=1, hidden=3, shape=(224, 224))

        self.model.zero_grad()
        self.optimizer.zero_grad()

        y_pred = torch.empty(X.shape).cuda()

        for fidx in range(X.shape[0]):
            frame = X[fidx, :, :, :]

            h, c = self.model.forward(frame, h, c)

            y_pred[fidx, :, :, :] = h

        loss = torch.mean((y_pred - y) ** 2)

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
                X_vid = X_vid.cuda()
                y_vid = y_vid.cuda()
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

        for idx, (X_vid, y_vid) in tqdm(enumerate(zip(data, targets))):

            X_vid = X_vid.cuda()
            y_vid = y_vid.cuda()

            h, c = self.model.init_hidden(batch_size=1, hidden=3,
                                          shape=(224, 224))

            y_pred = torch.empty(X_vid.shape).cuda()

            for fidx in range(X_vid.shape[0]):
                frame = X_vid[fidx, :, :, :]

                h, c = self.model.forward(frame, h, c)

                y_pred[fidx, :, :, :] = h

            reconstruction_errors[idx] = torch.mean((y_pred - y_vid) ** 2).item()


        return reconstruction_errors


def calculate_metrics(y, y_pred, threshold=None):

    if threshold == None:
        fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
        fnr = 1 - tpr
        threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    else:

        eer = sum(np.ones_like(y)[np.argwhere(np.logical_and(y == 0, y_pred >= threshold))])/sum(1 - y) # far
        eer += sum(np.ones_like(y)[np.argwhere(np.logical_and(y == 1, y_pred < threshold))])/sum(y) # frr
        eer /= 2
        eer = eer[0]

    roc_auc = roc_auc_score(y, y_pred)

    return eer, roc_auc, threshold


def get_eval_videos(f, split, data_name):

    data = {}

    if data_name == "replay_attack":
        for m in ["fixed", "hand"]:
            for i, vid_idx in enumerate(f[split]["attack"][m]):

                vid = f[split]["attack"][m][vid_idx]

                vid_arr = np.array(vid, dtype=np.float32) / 255

                vid_arr = torch.tensor(vid_arr)

                data[vid_idx] = {
                    "label": 0, # means imposter
                    "features": vid_arr
                }
    else:
        for vid_idx in f[split]["attack"]:
            vid = f[split]["attack"][vid_idx]

            vid_arr = np.array(vid, dtype=np.float32) / 255

            vid_arr = torch.tensor(vid_arr)

            data[vid_idx] = {
                "label": 0,  # means imposter
                "features": vid_arr
            }

    for i, vid_idx in enumerate(f[split]["real"]):

        vid = f[split]["real"][vid_idx]

        vid_arr = np.array(vid, dtype=np.float32) / 255

        vid_arr = torch.tensor(vid_arr)

        data[vid_idx] = {
            "label": 1, # genuine
            "features": vid_arr
        }

    return data


if __name__ == "__main__":
    data_name = "replay_mobile"
    path = os.path.join("/mnt/storage2/pad/", data_name, "raw_normalized_faces.h5")

    f = h5py.File(path, "r")

    X = []
    for vid_idx, vid in tqdm(f["train"]["real"].items()):

        vid_arr = np.array(vid, dtype=np.float32) / 255

        vid_arr = torch.tensor(vid_arr)

        X.append(vid_arr)

    stae = ConvLSTMAutoencoder()

    stae.fit(X)

    dev = get_eval_videos(f, "devel", data_name)

    y_dev = np.zeros((len(dev.keys()), 1))
    y_dev_pred = np.zeros((len(dev.keys()), 1))

    dev_videos = []

    data = []
    for i, (name, vid) in tqdm(enumerate(dev.items())):
        frame_scores = -stae.decision_function([vid["features"]])

        y_dev_pred[i] = frame_scores
        y_dev[i] = vid["label"]

        dev_videos.append(name)

    dev_eer, dev_roc_auc, threshold = calculate_metrics(y_dev, y_dev_pred)

    print("Per-Video Results")
    print(f"Development EER: {np.round(dev_eer, 4)} ROC (AUC): {np.round(dev_roc_auc,4)}")
