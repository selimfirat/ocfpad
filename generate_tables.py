import numpy as np
import os
import pickle

import pandas as pd
from bob.measure._library import eer_threshold
from sklearn.metrics import roc_auc_score
from bob.measure import eer, eer_threshold

df = pd.read_csv("results.csv")

df = df[df["normalize"] == "True"]

model_names = {
    "iforest": "iForest",
    "ocsvm": "OC-SVM",
    "ae": "Autoencoder"
}

agg_names = {
    "mean": "Mean",
    "max": "Max"
}

region_names = {
    "normalized_faces": "Normalized Face", #\\begin{tabular}[c]{@{}c@{}}Normalized\\\\ Face\\end{tabular}",
    "faces": "Face",
    "frames": "Frame"
}

path = "figures"
pkl_path = "/mnt/storage2/pad/pkl/"

for normalized in ["", "_normalized"]:
    for data in ["replay_mobile", "replay_attack"]:
        fname = data + "_baselines" + normalized + ".tex"
        table = """
            \\begin{tabular}{@{}ccccc@{}}
            \\toprule
            Model & Region & Aggregation & Video AUC (\\%) & Video EER (\\%) \\\\ \\midrule
        """
        for mi, model in enumerate(["iforest", "ocsvm", "ae"]):
            table += "\\multirow{4}{*}{" + model_names[model] + "} & "

            for ri, region in enumerate(["frames", "faces", "normalized_faces"]):
                if data == "replay_mobile" and region == "normalized_faces":
                    continue

                if ri > 0:
                    table += " & "
                table += "\\multirow{2}{*}{" + region_names[region] + "} & "

                for ai, aggregate in enumerate(["mean", "max"]):
                    if ai > 0:
                        table += " & & "
                    table +=  agg_names[aggregate] + " & "
                    scores_pkl_path = os.path.join(pkl_path, f"{data}_{model}_{aggregate}{normalized}_vgg16_{region}/scores.pkl")
                    y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames = pickle.load(open(scores_pkl_path, "rb"))

                    dev_negatives, dev_positives = y_dev_pred[y_dev == 0], y_dev_pred[y_dev == 1]
                    threshold = eer_threshold(dev_negatives, dev_positives)
                    eer_score, far, frr = eer(dev_negatives, dev_positives, also_farfrr=True)
                    roc_auc = roc_auc_score(y_dev, y_dev_pred)

                    table += f" {str(np.round(roc_auc*100,2))} & {str(np.round(eer_score*100,2))} "

                    table += "\\\\ "

            if mi != 2:
                table += "\\midrule "


        table += """
        \\bottomrule
        \\end{tabular}
        """


        tpath = os.path.join(path, fname)

        with open(tpath, "w+") as t:
            t.write(table)

        print(data, normalized, "Done")


for normalized in ["", "_normalized"]:
    for data in ["replay_mobile", "replay_attack"]:
        fname = data + "_baselines" + normalized + "_frames.tex"
        table = """
            \\begin{tabular}{@{}cccc@{}}
            \\toprule
            Model & Region & Frame AUC (\\%) & Frame EER (\\%) \\\\ \\midrule
        """
        for mi, model in enumerate(["iforest", "ocsvm", "ae"]):
            table += "\\multirow{2}{*}{" + model_names[model] + "} & "

            for ri, region in enumerate(["frames", "faces", "normalized_faces"]):
                if data == "replay_mobile" and region == "normalized_faces":
                    continue

                if ri > 0:
                    table += " & "
                table += "" + region_names[region] + " & "

                for ai, aggregate in enumerate(["mean"]):
                    if ai > 0:
                        table += " & & "


                    scores_pkl_path = os.path.join(pkl_path, f"{data}_{model}_{aggregate}{normalized}_vgg16_{region}/scores.pkl")
                    y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames = pickle.load(open(scores_pkl_path, "rb"))

                    dev_negatives, dev_positives = y_dev_frames_pred[y_dev_frames == 0], y_dev_frames_pred[y_dev_frames == 1]
                    threshold = eer_threshold(dev_negatives, dev_positives)
                    eer_score, far, frr = eer(dev_negatives, dev_positives, also_farfrr=True)
                    roc_auc = roc_auc_score(y_dev_frames, y_dev_frames_pred)

                    table += f" {str(np.round(roc_auc*100,2))} & {str(np.round(eer_score*100,2))} "

                    table += "\\\\ "

            if mi != 2:
                table += "\\midrule "


        table += """
        \\bottomrule
        \\end{tabular}
        """


        tpath = os.path.join(path, fname)

        with open(tpath, "w+") as t:
            t.write(table)

        print(data, normalized, "Done", " Frame Level")


# Image quality
for normalized in ["", "_normalized"]:
    for data in ["replay_attack"]:
        fname = data + "_image_quality" + normalized + ".tex"
        table = """
            \\begin{tabular}{@{}cccc@{}}
            \\toprule
            Model & Aggregation & Video AUC (\\%) & Video EER (\\%) \\\\ \\midrule
        """
        for mi, model in enumerate(["iforest", "ocsvm", "ae"]):
            table += "\\multirow{2}{*}{" + model_names[model] + "} & "

            for ai, aggregate in enumerate(["mean", "max"]):
                if ai > 0:
                    table += " & "
                table +=  agg_names[aggregate] + " & "
                scores_pkl_path = os.path.join(pkl_path, f"{data}_{model}_{aggregate}{normalized}_vgg16_{region}/scores.pkl")
                y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames = pickle.load(open(scores_pkl_path, "rb"))

                dev_negatives, dev_positives = y_dev_pred[y_dev == 0], y_dev_pred[y_dev == 1]
                threshold = eer_threshold(dev_negatives, dev_positives)
                eer_score, far, frr = eer(dev_negatives, dev_positives, also_farfrr=True)
                roc_auc = roc_auc_score(y_dev, y_dev_pred)

                table += f" {str(np.round(roc_auc*100,2))} & {str(np.round(eer_score*100,2))} "

                table += "\\\\ "

            if mi != 2:
                table += "\\midrule "


        table += """
        \\bottomrule
        \\end{tabular}
        """


        tpath = os.path.join(path, fname)

        with open(tpath, "w+") as t:
            t.write(table)

        print(data, normalized, " Image quality", "Done")


for normalized in ["", "_normalized"]:
    for data in ["replay_attack"]:
        fname = data + "_image_quality" + normalized + "_frames.tex"
        table = """
            \\begin{tabular}{@{}ccc@{}}
            \\toprule
            Model & Frame AUC (\\%) & Frame EER (\\%) \\\\ \\midrule
        """
        for mi, model in enumerate(["iforest", "ocsvm", "ae"]):
            table += model_names[model] + " & "

            for ai, aggregate in enumerate(["mean"]):
                if ai > 0:
                    table += " & "


                scores_pkl_path = os.path.join(pkl_path, f"{data}_{model}_{aggregate}{normalized}_vgg16_{region}/scores.pkl")
                y_dev, y_dev_pred, dev_videos, y_dev_frames, y_dev_frames_pred, dev_videos_frames, y_test, y_test_pred, test_videos, y_test_frames, y_test_frames_pred, test_videos_frames = pickle.load(open(scores_pkl_path, "rb"))

                dev_negatives, dev_positives = y_dev_frames_pred[y_dev_frames == 0], y_dev_frames_pred[y_dev_frames == 1]
                threshold = eer_threshold(dev_negatives, dev_positives)
                eer_score, far, frr = eer(dev_negatives, dev_positives, also_farfrr=True)
                roc_auc = roc_auc_score(y_dev_frames, y_dev_frames_pred)

                table += f" {str(np.round(roc_auc*100,2))} & {str(np.round(eer_score*100,2))} "

                table += "\\\\ "

            if mi != 2:
                table += "\\midrule "


        table += """
        \\bottomrule
        \\end{tabular}
        """


        tpath = os.path.join(path, fname)

        with open(tpath, "w+") as t:
            t.write(table)

        print(data, normalized, "Done", " Frame Level")

