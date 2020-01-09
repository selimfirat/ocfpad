import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

df = pd.read_csv("results.csv")

ax = df[(df["data"]=="replay_mobile") & (df["features"] != "vgg16_normalized_faces")].boxplot("dev_eer", by=["normalize"], return_type='axes')

plt.title("Replay-Mobile")
plt.suptitle("")
plt.xlabel("Normalization")

plt.ylabel("Dev EER Score")
plt.ylim((0.25, 0.5))


plt.savefig("figures/normalization_replaymobile.pdf")

plt.clf()

ax = df[df["data"]=="replay_attack"].boxplot("dev_eer", by=["normalize"], return_type='axes')

plt.title("Replay-Attack")
plt.suptitle("")
plt.xlabel("Normalization")
plt.ylim((0.25, 0.5))
plt.ylabel("Dev EER Score")

plt.savefig("figures/normalization_replayattack.pdf")
plt.clf()


ax = df[(df["data"]=="replay_mobile") & (df["features"] != "vgg16_normalized_faces")].boxplot("dev_eer", by=["features"], return_type='axes')

plt.title("Replay-Mobile")
plt.suptitle("")
plt.xlabel("Feature")

plt.ylabel("Dev EER Score")
plt.ylim((0.25, 0.5))


plt.savefig("figures/feature_replaymobile.pdf")
plt.clf()


ax = df[df["data"]=="replay_attack"].boxplot("dev_eer", by=["features"], return_type='axes')

plt.title("Replay-Attack")
plt.suptitle("")
plt.xlabel("Feature")
plt.ylim((0.25, 0.5))
plt.ylabel("Dev EER Score")

plt.savefig("figures/feature_replayattack.pdf")
plt.clf()

ax = df[(df["data"]=="replay_mobile") & (df["features"] != "vgg16_normalized_faces")].boxplot("dev_eer", by=["aggregate"], return_type='axes')

plt.title("Replay-Mobile")
plt.suptitle("")
plt.xlabel("Aggregation")

plt.ylabel("Dev EER Score")
plt.ylim((0.25, 0.5))


plt.savefig("figures/aggregate_replaymobile.pdf")
plt.clf()


ax = df[df["data"]=="replay_attack"].boxplot("dev_eer", by=["aggregate"], return_type='axes')

plt.title("Replay-Attack")
plt.suptitle("")
plt.xlabel("Aggregation")
plt.ylim((0.25, 0.5))
plt.ylabel("Dev EER Score")

plt.savefig("figures/aggregate_replayattack.pdf")
plt.clf()

ax = df[(df["data"]=="replay_mobile") & (df["features"] != "vgg16_normalized_faces")].boxplot("dev_eer", by=["model"], return_type='axes')

plt.title("Replay-Mobile")
plt.suptitle("")
plt.xlabel("Model")

plt.ylabel("Dev EER Score")
plt.ylim((0.25, 0.5))


plt.savefig("figures/model_replaymobile.pdf")
plt.clf()


ax = df[df["data"]=="replay_attack"].boxplot("dev_eer", by=["model"], return_type='axes')

plt.title("Replay-Attack")
plt.suptitle("")
plt.xlabel("Model")
plt.ylim((0.25, 0.5))
plt.ylabel("Dev EER Score")

plt.savefig("figures/model_replayattack.pdf")
plt.clf()