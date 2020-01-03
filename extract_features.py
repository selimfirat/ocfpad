import h5py

import PIL
import argparse
import skvideo.io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm
from bob.ip.qualitymeasure import galbally_iqm_features as iqm
import numpy as np
import os


class VideoFrames(Dataset):

    def __init__(self, tensors, transform=None):

        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return self.tensors.size(0)


parser = argparse.ArgumentParser("Extract VGG Features From Videos")

parser.add_argument("--input", default="/mnt/storage2/pad/videos/replay_mobile/", type=str, help="Input directory to be extracted.")
parser.add_argument("--output", default="/mnt/storage2/pad/replay_mobile/image_quality.h5", help="Output file for frames to write.")
parser.add_argument("--device", default="cuda:1", type=str)
parser.add_argument("--feature", default="image_quality", type=str, choices=["vgg16", "raw", "vggface", "image_quality"])
parser.add_argument("--type", default="frame", type=str, choices=["frame", "face"])

args = vars(parser.parse_args())

input_path = args["input"]
output_path = args["output"]

device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")


# VGG16
if args["feature"] == "vgg16":
    vgg16 = models.vgg16(pretrained=True).to(device)

    vgg_extractor = torch.nn.Sequential(
                        # stop at conv4
                        *list(vgg16.classifier)[:-2]
                    ).to(device)
elif args["feature"] == "vggface":

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    from keras_vggface.vggface import VGGFace
    from keras.engine import Model
    from keras_vggface import utils

    vggface = VGGFace(model='vgg16')
    out = vggface.get_layer("fc7/relu").output
    vggface_new = Model(vggface.input, out)


def vgg16_features(x):
    x = x.to(device)

    x = vgg16.features(x)
    x = vgg16.avgpool(x)
    x = torch.flatten(x, 1)
    x = vgg_extractor(x)

    x = x.detach().cpu().numpy()

    return x


def vggface_features(x):
    x = x.detach().cpu().numpy()

    ## VGGFACE DISCLAIMER: Note that when using TensorFlow, for best performance you should set `image_data_format="channels_last"` in your Keras config at ~/.keras/keras.json.
    x = utils.preprocess_input(x, data_format="channels_first", version=1)

    res_arr = vggface_new.predict(x)

    return res_arr

def image_quality_features(x):
    x = x.detach().cpu().numpy()

    res = []
    for i in range(x.shape[0]):
        rx = iqm.compute_quality_features(x[i, :, :, :])
        res.append(np.array(rx))

    res = np.array(res)

    return res



def raw_features(x):

    x = x.detach().cpu().numpy()
    x *= 255
    x = x.astype(np.uint8)
    # print(x.max(), x.min(), x.mean(), x.std())

    return x


def crop_faces(r, bboxes):
    res = []
    for i in range(r.shape[0]):

        if "replay_attack" in args["input"]:
            _, x, y, w, h = bboxes[i]
        else:
            x, y, w, h = bboxes[i]

        a = r[i, :, y:y+h+1, x:x+w+1]

        res.append(a)

    return r


def extract_features(inp, feature_extractor, bboxes):

    r = skvideo.io.vread(inp)

    r = r.transpose((0, 3, 1, 2))

    if args["type"] == "face":
        r = crop_faces(r, bboxes)

    r = torch.tensor(r)

    complst = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor()
    ]

    if args["feature"] == "vgg16":
        complst.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    comp = transforms.Compose(complst)

    dl = torch.utils.data.DataLoader(VideoFrames(r, transform=comp), batch_size=64, num_workers=8,
                                     shuffle=False, pin_memory=True)

    batches = []
    for i, inp in enumerate(dl):

        res = feature_extractor(inp)

        batches.append(res)

    res_arr = np.concatenate(batches, axis=0)

    return res_arr


f = h5py.File(output_path, 'a')

for root, dirs, files in tqdm(os.walk(input_path)):
    hpath = root.replace(input_path, "")
    res_dir = os.path.join(output_path, hpath)
    print(hpath)

    for file in tqdm(files):
        if not file.endswith(".mov"):
            continue

        inp = os.path.join(root, file)

        bboxes = None

        if args["type"] == "face":
            if "replay_mobile" in args["input"]:
                faces_path = os.path.join(args["input"], "faceloc", "rect", hpath, file.replace(".mov", ".face"))
            elif "replay_attack" in args["input"]:
                faces_path = os.path.join(args["input"], "face-locations", hpath, file.replace(".mov", ".face"))

            bboxes = open(faces_path, "r").readlines()

            for i in range(len(bboxes)):
                bboxes[i] = list(map(int, bboxes[i].split()))

            bboxes = np.array(bboxes)

        res_arr = extract_features(inp, globals()[args["feature"] + "_features"], bboxes)

        g = f[hpath] if hpath in f else f.create_group(hpath)

        g.create_dataset(file, data=res_arr)
