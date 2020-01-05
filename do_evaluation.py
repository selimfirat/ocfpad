import numpy as np
import os
import pickle

from bob.measure import eer, eer_threshold, plot
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from plot_helpers import plot_scores_distributions

pkl_path = "/mnt/storage2/pad/pkl/"



def plot_far_frr(negatives, positives, path, title):
    plot.roc(negatives, positives)

    plt.xlabel("False Acceptance Rate (FAR)")

    plt.ylabel("False Rejection Rate (FRR)")

    plt.title(title)

    plt.savefig(path)

    plt.clf()

def plot_det(negatives, positives, path, title):
    plot.det(negatives, positives)

    plt.xlim((-5, 5))
    plt.ylim((-5, 5))

    plt.xlabel("False Acceptance")

    plt.ylabel("False Rejection")

    plt.title(title)

    plt.savefig(path)

    plt.clf()

def plot_epc(dev_negatives, dev_positives, test_negatives, test_positives, path, title):
    plot.epc(dev_negatives, dev_positives, test_negatives, test_positives)

    plt.xlabel("Cost")

    plt.ylabel("Minimum HTER (%)")

    plt.title(title)

    plt.savefig(path)

    plt.clf()


for model_folder in os.listdir(pkl_path):
    scores_pkl = os.path.join(pkl_path, model_folder, "scores.pkl")
    print(model_folder)

    if not os.path.exists(scores_pkl):
        continue

    with open(scores_pkl, "rb") as m:
        y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames = pickle.load(m)

    dev_negatives, dev_positives = y_dev_pred[y_dev==0], y_dev_pred[y_dev==1]
    test_negatives, test_positives = y_test_pred[y_test==0], y_test_pred[y_test==1]

    plot_far_frr(dev_negatives, dev_positives, "figures/test_roc.pdf", "FAR vs. FRR Curve")
    plot_epc(dev_negatives, dev_positives, test_negatives, test_positives, "figures/test_epc.pdf", "Expected Performance Curve")
    plot_det(dev_negatives, dev_positives, "figures/test_det.pdf", "Detection Error Trade-off Curve")

    y_dev_print = np.zeros((len(y_dev), 1))
    y_dev_mobile = np.zeros((len(y_dev), 1))
    y_dev_highdef = np.zeros((len(y_dev), 1))

    plot_scores_distributions([dev_negatives, dev_positives], [test_negatives, test_positives], path="figures/test_hists.pdf")

    threshold = eer_threshold(dev_negatives, dev_positives)
    eer, far, frr = eer(dev_negatives, dev_positives, also_farfrr=True)

    print(eer, far, frr, threshold)

    print(len(dev_videos))

    for i in range(len(dev_videos)):
        print(dev_videos[i])

    # Replay-Attack
    # attack_mobile
    # attack_highdef
    # attack_print

    # Replay-Mobile
    # attack_mobile
    # attack_highdef
    # attack_print

    break
