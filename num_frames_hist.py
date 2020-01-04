import numpy as np

import h5py
import os
import seaborn as sns
import matplotlib.pyplot as plt
from h5py import Dataset
import pandas as pd

sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

# Set up the matplotlib figure
f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)

data = ["replay_attack", "replay_mobile"]
for datum in data:

    data_path = os.path.join("/mnt/storage2/pad/", datum, "vgg16_frames.h5")

    f = h5py.File(data_path, "r")

    num_frames = []

    for fk, fv in f.items():
        for tidx, typ in fv.items():
            for vidx, vid in typ.items():
                if type(vid) is Dataset:
                    num_frames.append(vid.shape[0])
                else:
                    for viddx, vidd in vid.items():
                        num_frames.append(vidd.shape[0])

    ax = sns.distplot(num_frames, kde=False, color="b", ax=axes)
    ax.set(title = datum.replace("_", " ").title() + " - Number of Frames Histogram", ylabel = "Number of Videos", xlabel="Number of Frames")
    plt.savefig(f"figures/num_frames_{datum}.pdf")

    df = pd.DataFrame(num_frames)
    print(datum)
    print(df.describe())