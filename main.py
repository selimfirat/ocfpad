import argparse
import pickle

import numpy as np
import os

from normalized_model import NormalizedModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

np.random.seed(1)


def get_training_frames(f):
    split = "train"
    num_reals = sum(f[split]["real"][vid].shape[0] for vid in f[split]["real"])

    X = np.zeros((num_reals, 4096))

    cur = 0
    for vid_idx in f[split]["real"]:
        vid = f[split]["real"][vid_idx]

        X[cur:cur + vid.shape[0], :] = vid[:, :]
        cur += vid.shape[0]

    return X


def aggregate(x):
    if args["aggregate"] == "mean":
        return x.mean()
    elif args["aggregate"] == "max":
        return x.max()


def get_eval_videos(f, split, data_name):

    data = {}

    if data_name == "replay_attack":
        for m in ["fixed", "hand"]:
            for vid_idx in f[split]["attack"][m]:
                vid = f[split]["attack"][m][vid_idx]

                data[vid_idx] = {
                    "label": 0, # means imposter
                    "features": vid[:, :]
                }
    else:
        for vid_idx in f[split]["attack"]:
            vid = f[split]["attack"][vid_idx]

            data[vid_idx] = {
                "label": 0,  # means imposter
                "features": vid[:, :]
            }

    for vid_idx in f[split]["real"]:
        vid = f[split]["real"][vid_idx]

        data[vid_idx] = {
            "label": 1, # genuine
            "features": vid[:, :]
        }

    return data


def eval(f, split, data):
    dev = get_eval_videos(f, split, data)

    y_dev = np.zeros(len(dev.keys()))
    y_dev_pred = np.zeros(len(dev.keys()))
    dev_videos = []

    total_frames = sum(vid["features"].shape[0] for vid in dev.values())
    y_dev_frames = np.zeros(total_frames)
    y_dev_frames_pred = np.zeros(total_frames)
    dev_videos_frames = []

    cur_idx = 0
    for i, (name, vid) in tqdm(enumerate(dev.items())):
        frame_scores = model.predict_proba(vid["features"])[:, 0]
        num_frames = frame_scores.shape[0]

        y_dev_frames_pred[cur_idx:cur_idx + num_frames] = frame_scores
        y_dev_frames[cur_idx:cur_idx + num_frames] = vid["label"]
        cur_idx += num_frames

        y_dev_pred[i] = aggregate(frame_scores)
        y_dev[i] = vid["label"]

        dev_videos.append(name)
        dev_videos_frames.append(name)

    return y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames


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


def calculate_all_metrics(y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames):

    dev_eer, dev_roc_auc, threshold = calculate_metrics(y_dev, y_dev_pred)
    test_hter, test_roc_auc, _ = calculate_metrics(y_test, y_test_pred, threshold)

    print("Per-Video Results")
    print(f"Development EER: {np.round(dev_eer, 4)} ROC (AUC): {np.round(dev_roc_auc,4)}")
    print(f"Test HTER: {np.round(test_hter, 4)} ROC (AUC): {np.round(test_roc_auc,4)}")

    dev_frames_eer, dev_frames_roc_auc, frames_threshold = calculate_metrics(y_dev_frames, y_dev_frames_pred)
    test_frames_hter, test_frames_roc_auc, _ = calculate_metrics(y_test_frames, y_test_frames_pred, frames_threshold)


    print("Per-Frame Results")
    print(f"Development EER: {np.round(dev_frames_eer, 4)} ROC (AUC): {np.round(dev_frames_roc_auc,4)}")
    print(f"Test HTER: {np.round(test_frames_hter, 4)} ROC (AUC): {np.round(test_frames_roc_auc,4)}")

    return dev_eer, dev_roc_auc, threshold, test_hter, test_roc_auc, dev_frames_eer, dev_frames_roc_auc, frames_threshold, test_frames_hter, test_frames_roc_auc

parser = argparse.ArgumentParser("One Class Face Presentation Attack Detection Pipeline")

parser.add_argument("--model", default="iforest", choices=["ocsvm", "iforest", "ae", "stae"], type=str, help="Name of the method")
parser.add_argument("--aggregate", default="mean", choices=["mean", "max"], type=str, help="Aggregate block scores via mean/max or None")
parser.add_argument("--data", default="replay_attack", choices=["replay_attack", "replay_mobile"], type=str)
parser.add_argument("--data_path", default="/mnt/storage2/pad/", type=str)
parser.add_argument("--features", default=["vgg16_frames"], nargs="+", choices=["vggface_faces", "vgg16_faces", "vgg16_frames", "vggface_frames", "raw_faces"])
parser.add_argument("--log", default=None, type=str)

parser.add_argument("--interdb", default=False, action="store_true")
parser.add_argument("--normalize", default=False, action="store_true")

args = vars(parser.parse_args())

print(args)

import numpy as np
import h5py
import os
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

path = os.path.join(args["data_path"], args["data"], f"{'_'.join(args['features'])}.h5")

f = h5py.File(path, "r")

experiment_name = f"{args['data']}_{args['model']}_{args['aggregate']}{'_normalized' if args['normalize'] else ''}_{'_'.join(args['features'])}"

save_path = os.path.join("/mnt/storage2/pad/pkl", experiment_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

model_path = os.path.join(save_path, "model.pkl")

if not os.path.exists(model_path):

    models = {
        "ocsvm": OCSVM(),
        "iforest": IForest(behaviour="new"),
    }

    if args["model"] == "ae":
        from pyod.models.auto_encoder import AutoEncoder
        models["ae"] = AutoEncoder(epochs=50, preprocessing=False)

    if args["normalize"]:
        model = NormalizedModel(models[args["model"]])
    else:
        model = models[args["model"]]

    X_train = get_training_frames(f)

    model.fit(X_train)

    with open(model_path, "wb+") as m:
        pickle.dump(model, m)

with open(model_path, 'rb') as m:
    model = pickle.load(m)


if args["interdb"]:
    scores_path = os.path.join(save_path, "scores.pkl")

    if not os.path.exists(scores_path):
        raise Exception("Please do intra-database experiment of the model first.")

    with open(scores_path, "rb") as m:
        y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, _, _, _, _, _, _  = pickle.load(m)

    scores_path = os.path.join(save_path, "interdb_scores.pkl")

    if not os.path.exists(scores_path):

        other_data = "replay_attack" if args['data'] == "replay_mobile" else "replay_mobile"

        other_path = os.path.join(args["data_path"], other_data, f"{'_'.join(args['features'])}.h5")

        other_f = h5py.File(other_path, "r")

        y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames = eval(other_f, "test", other_data)

        with open(scores_path, "wb+") as m:
            pickle.dump((y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames), m)


    with open(scores_path, "rb") as m:
        y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames = pickle.load(m)


    # https://arxiv.org/pdf/1807.00848.pdf

    dev_eer, dev_roc_auc, threshold, test_hter, test_roc_auc, dev_frames_eer, dev_frames_roc_auc, frames_threshold, test_frames_hter, test_frames_roc_auc = calculate_all_metrics(y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames)

else: # Intra database evaluation

    scores_path = os.path.join(save_path, "scores.pkl")

    if not os.path.exists(scores_path):

        y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames = eval(f, "devel", args['data'])

        y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames = eval(f, "test", args['data'])

        with open(scores_path, "wb+") as m:
            pickle.dump((y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames), m)


    with open(scores_path, "rb") as m:
        y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames = pickle.load(m)


    # https://arxiv.org/pdf/1807.00848.pdf
    dev_eer, dev_roc_auc, threshold, test_hter, test_roc_auc, dev_frames_eer, dev_frames_roc_auc, frames_threshold, test_frames_hter, test_frames_roc_auc = calculate_all_metrics(y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames)



if args["log"] is not None:
    res = [args['data'], str(args["interdb"]), args['model'], args['aggregate'], '_'.join(args['features']), str(args["normalize"]), str(np.round(dev_eer, 4)), str(np.round(dev_roc_auc, 4)), str(np.round(test_hter, 4)), str(np.round(test_roc_auc, 4)), str(np.round(dev_frames_eer, 4)), str(np.round(dev_frames_roc_auc, 4)), str(np.round(test_frames_hter, 4)), str(np.round(test_frames_roc_auc, 4))]
    if not os.path.exists(args["log"]):
        with open(args["log"], 'w+') as fd:
            fd.write(",".join(["data", "interdb", "model", "aggregate", "features", "normalize", "dev_eer", "dev_roc_auc", "test_hter", "test_roc_auc", "dev_frames_eer", "dev_frames_roc_auc", "test_frames_hter", "test_frames_roc_auc"]) + "\n")

    with open(args["log"], 'a+') as fd:
        fd.write(",".join(res) + "\n")
