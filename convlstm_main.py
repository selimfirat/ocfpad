import argparse
import pickle

import numpy as np
import os
from sklearn.metrics import roc_curve, roc_auc_score
from torch.autograd import Variable
from tqdm import tqdm
import h5py

import torch

from convlstm_autoencoder import ConvLSTMAutoencoder


def calculate_metrics(y, y_pred, threshold=None, test_videos=None):

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

    if test_videos:
        vid_bonafide = np.zeros(len(test_videos), dtype=bool)
        vid_mobile = np.zeros(len(test_videos), dtype=bool)
        vid_highdef = np.zeros(len(test_videos), dtype=bool)
        vid_print = np.zeros(len(test_videos), dtype=bool)

        for i, vid in enumerate(test_videos):
            if not "attack" in vid:
                vid_bonafide[i] = True
            elif "photo" in vid:
                vid_mobile[i] = True
            elif "video" in vid:
                vid_highdef[i] = True
            elif "print" in vid:
                vid_print[i] = True

        test_far_mobile = sum(y_test_pred[vid_mobile] >= threshold) / sum(vid_mobile)
        test_far_highdef = sum(y_test_pred[vid_highdef] >= threshold) / sum(vid_highdef)
        test_far_print = sum(y_test_pred[vid_print] >= threshold) / sum(vid_print)
        bpcer = sum(y_test_pred[vid_bonafide] < threshold) / sum(vid_bonafide)

        apcer = max(test_far_mobile, test_far_highdef, test_far_print)

        acer = (apcer + bpcer)/2

        print("ACER", acer)


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

    parser = argparse.ArgumentParser("Running ConvLSTM Autoencoder")

    parser.add_argument("--data", default="replay_attack")
    parser.add_argument("--cuda", default=0, type=int)
    parser.add_argument("--feature", default="raw_normalized_faces", type=str,
                        choices=["raw_faces", "raw_normalized_faces", "raw_frames"])
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--reg", default=0.5, type=float)
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--interdb", action="store_true", default=False)
    parser.add_argument("--log", default="convlstm_results.csv", type=str)

    args = vars(parser.parse_args())

    experiment_name = f"{args['data']}_convlstm_{str(args['epochs'])}epochs_lr{str(args['lr'])}_reg{str(args['reg'])}_{args['feature']}"

    path = os.path.join("/mnt/storage2/pad/", args["data"], args["feature"]+".h5")

    f = h5py.File(path, "r")

    pkl_path = os.path.join("/mnt/storage2/pad/pkl/", experiment_name)

    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)

    model_path = os.path.join(pkl_path, "model.h5")

    if args["interdb"]:
        scores_path = os.path.join(pkl_path, "interdb_scores.pkl")
    else:
        scores_path = os.path.join(pkl_path, "scores.pkl")

    if not os.path.exists(model_path):
        X = []
        for vid_idx, vid in tqdm(f["train"]["real"].items()):

            vid_arr = np.array(vid, dtype=np.float32) / 255

            vid_arr = torch.tensor(vid_arr)

            X.append(vid_arr)

        stae = ConvLSTMAutoencoder(num_epochs=args["epochs"], lr=args["lr"], cuda_idx=args["cuda"], reg=args["reg"], kernel_size=args["kernel_size"])

        stae.fit(X)

        torch.save(stae, model_path)

    stae = torch.load(model_path)

    if not os.path.exists(scores_path):
        dev = get_eval_videos(f, "devel", args["data"])

        y_dev = np.zeros(len(dev.keys()))
        y_dev_pred = np.zeros(len(dev.keys()))

        dev_videos = []

        data = []
        for i, (name, vid) in tqdm(enumerate(dev.items())):
            vid_score = -stae.decision_function([vid["features"]])

            y_dev_pred[i] = vid_score
            y_dev[i] = vid["label"]

            dev_videos.append(name)

        dev_eer, dev_roc_auc, threshold = calculate_metrics(y_dev, y_dev_pred)

        print("Per-Video Results")
        print(f"Development EER: {np.round(dev_eer, 4)} ROC (AUC): {np.round(dev_roc_auc,4)}")


        if args["interdb"]:
            other_data = "replay_attack" if args['data'] == "replay_mobile" else "replay_mobile"


            other_path = os.path.join("/mnt/storage2/pad/", other_data, args["feature"] + ".h5")

            other_f = h5py.File(other_path, "r")
            test = get_eval_videos(other_f, "test", other_data)

        else:
            test = get_eval_videos(f, "test", args["data"])

        y_test = np.zeros(len(test.keys()))
        y_test_pred = np.zeros(len(test.keys()))

        test_videos = []

        data = []
        for i, (name, vid) in tqdm(enumerate(test.items())):
            vid_score = -stae.decision_function([vid["features"]])

            y_test_pred[i] = vid_score
            y_test[i] = vid["label"]

            test_videos.append(name)

        test_eer, test_roc_auc, _ = calculate_metrics(y_test, y_test_pred, test_videos=test_videos)

        print(f"Test HTER: {np.round(test_eer, 4)} ROC (AUC): {np.round(test_roc_auc,4)}")

        with open(scores_path, "wb+") as m:
            pickle.dump((y_dev, y_dev_pred, dev_videos, y_test, y_test_pred, test_videos), m)

    with open(scores_path, "rb") as m:
        y_dev, y_dev_pred, dev_videos, y_test, y_test_pred, test_videos = pickle.load(m)

    dev_eer, dev_roc_auc, threshold = calculate_metrics(y_dev, y_dev_pred)
    test_hter, test_roc_auc, _ = calculate_metrics(y_test, y_test_pred)

    if not os.path.exists(args["log"]):
        with open(args["log"], 'w+') as fd:
            fd.write(",".join(
                ["data", "interdb", "model", "feature", "epochs", "lr", "reg", "kernel_size", "dev_eer", "dev_roc_auc",
                 "test_hter", "test_roc_auc"]) + "\n")

    res = [args["data"], str(args["interdb"]), "convlstm", args["feature"], str(args["epochs"]), str(args["lr"]), str(args["reg"]), str(args["kernel_size"]), str(dev_eer), str(dev_roc_auc), str(test_hter), str(test_roc_auc)]

    print("Per-Video Results")
    print(f"Development EER: {np.round(dev_eer, 4)} ROC (AUC): {np.round(dev_roc_auc,4)}")
    print(f"Test HTER: {np.round(test_hter, 4)} ROC (AUC): {np.round(test_roc_auc,4)}")

    with open(args["log"], 'a+') as fd:
        fd.write(",".join(res) + "\n")


