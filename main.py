import argparse

parser = argparse.ArgumentParser("One Class Face Presentation Attack Detection Pipeline")

parser.add_argument("--model", default="ocsvm", choices=["ocsvm", "iforest", "ae"], type=str, help="Name of the method")
parser.add_argument("--aggregate", default=None, choices=[None, "mean", "max"], type=str, help="Aggregate block scores via mean/max or None")
parser.add_argument("--data", default="replay_attack", choices=["replay_attack", "replay_mobile"], type=str)
parser.add_argument("--crop", default=None, choices=[None, "face"], type=str)
parser.add_argument("--features", default=["raw_frames"], nargs="+", choices=["image_quality", "lbp", "openpose", "vgg16", "vggface"])

args = vars(parser.parse_args())

print(args)