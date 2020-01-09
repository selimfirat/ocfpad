import json

import numpy as np
import os
import pickle

from bob.measure import eer
from bob.measure._library import eer_threshold
from sklearn.metrics import roc_auc_score

from do_evaluation import plot_det_comparison


def get_best_models():
    pkl_path = "/mnt/storage2/pad/pkl"

    models = {
        "replay_attack": {
            "ocsvm": {
                "dev_eer": 999
            },
            "iforest": {
                "dev_eer": 999
            },
            "convlstm": {
                "dev_eer": 999
            },
            "ae": {
                "dev_eer": 999
            }
        },
        "replay_mobile": {
            "ocsvm": {
                "dev_eer": 999
            },
            "iforest": {
                "dev_eer": 999
            },
            "convlstm": {
                "dev_eer": 999
            },
            "ae": {
                "dev_eer": 999
            }
        }
    }

    for data in ["replay_mobile", "replay_attack"]:
        for normalized in ["_normalized"]:
            for mi, model in enumerate(["iforest", "ocsvm", "ae"]):
                for ri, region in enumerate(["frames", "faces", "normalized_faces"]):
                    if data == "replay_mobile" and region == "normalized_faces":
                        continue
                    for ai, aggregate in enumerate(["mean", "max"]):
                        scores_pkl_path = os.path.join(pkl_path,
                                                       f"{data}_{model}_{aggregate}{normalized}_vgg16_{region}/scores.pkl")
                        y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames = pickle.load(
                            open(scores_pkl_path, "rb"))

                        dev_negatives, dev_positives = y_dev_pred[y_dev == 0], y_dev_pred[y_dev == 1]
                        threshold = eer_threshold(dev_negatives, dev_positives)
                        dev_eer, far, frr = eer(dev_negatives, dev_positives, also_farfrr=True)
                        dev_auc = roc_auc_score(y_dev, y_dev_pred)


                        if dev_eer < models[data][model]["dev_eer"] or (dev_eer == models[data][model]["dev_eer"] and dev_auc > models[data][model]["dev_eer"]):
                            models[data][model]["dev_eer"] = dev_eer
                            models[data][model]["dev_auc"] = dev_auc

                            test_far = sum(np.ones_like(y_test)[np.argwhere(
                                np.logical_and(y_test == 0, y_test_pred >= threshold))]) / sum(
                                1 - y_test)  # far
                            test_frr = sum(np.ones_like(y_test)[np.argwhere(
                                np.logical_and(y_test == 1, y_test_pred < threshold))]) / sum(
                                y_test)  # frr

                            hter = (test_far + test_frr) / 2
                            hter = hter[0]
                            test_auc = roc_auc_score(y_test, y_test_pred)

                            models[data][model]["dev_eer"] = dev_eer
                            models[data][model]["dev_auc"] = dev_auc

                            models[data][model]["test_hter"] = hter
                            models[data][model]["test_auc"] = test_auc
                            models[data][model]["scores_path"] = scores_pkl_path

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
                            models[data][model]["acer"] = (bpcer + apcer) / 2

                            models[data][model]["dev_eer"] = dev_eer
                            models[data][model]["dev_auc"] = dev_auc

                            models[data][model]["test_hter"] = hter
                            models[data][model]["test_auc"] = test_auc
                            models[data][model]["scores_path"] = scores_pkl_path

        for folder in os.listdir(pkl_path):
            if not os.path.isdir(os.path.join(pkl_path, folder)) or not ("convlstm" in folder) or not (data in folder):
                continue

            model = "convlstm"

            scores_pkl_path = os.path.join(pkl_path, folder, "scores.pkl")
            if not os.path.exists(scores_pkl_path):
                continue

            y_dev, y_dev_pred, dev_videos, y_test, y_test_pred, test_videos = pickle.load(
                open(scores_pkl_path, "rb"))

            dev_negatives, dev_positives = y_dev_pred[y_dev == 0], y_dev_pred[y_dev == 1]
            threshold = eer_threshold(dev_negatives, dev_positives)
            dev_eer, far, frr = eer(dev_negatives, dev_positives, also_farfrr=True)
            dev_auc = roc_auc_score(y_dev, y_dev_pred)

            if dev_eer < models[data][model]["dev_eer"] or (
                    dev_eer == models[data][model]["dev_eer"] and dev_auc > models[data][model]["dev_eer"]):
                models[data][model]["dev_eer"] = dev_eer
                models[data][model]["dev_auc"] = dev_auc

                test_far = sum(np.ones_like(y_test)[np.argwhere(np.logical_and(y_test == 0, y_test_pred >= threshold))]) / sum(
                    1 - y_test)  # far
                test_frr = sum(np.ones_like(y_test)[np.argwhere(np.logical_and(y_test == 1, y_test_pred < threshold))]) / sum(
                    y_test)  # frr
                hter = (test_far+test_frr)/2
                hter = hter[0]
                test_auc = roc_auc_score(y_test, y_test_pred)

                models[data][model]["dev_eer"] = dev_eer
                models[data][model]["dev_auc"] = dev_auc

                models[data][model]["test_hter"] = hter
                models[data][model]["test_auc"] = test_auc
                models[data][model]["scores_path"] = scores_pkl_path

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
                models[data][model]["acer"] = (bpcer + apcer) / 2

    return models


best_models = get_best_models()

# Export Detection-Error Trade-off curve
for data, models in best_models.items():
    scores_pkl_path = models["convlstm"]["scores_path"]
    with open(scores_pkl_path, "rb") as m:
        r = pickle.load(m)
    y_dev, y_dev_pred, dev_videos, y_test, y_test_pred, test_videos = r
    dev_negatives, dev_positives = y_dev_pred[y_dev==0], y_dev_pred[y_dev==1]
    test_negatives, test_positives = y_test_pred[y_test==0], y_test_pred[y_test==1]

    plot_det_comparison(y_test, y_test_pred, test_videos, f"figures/{data}_det.pdf")

print(json.dumps(best_models, indent=4, sort_keys=True))