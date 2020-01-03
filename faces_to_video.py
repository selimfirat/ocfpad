import os

from tqdm import tqdm

input_path = "/mnt/storage2/pad/frames"
videos_path = "/mnt/storage2/pad/videos"
output_path = "/mnt/storage2/pad/faces"


for root, dirs, files in tqdm(os.walk(videos_path)):
    hpath = root.replace(videos_path, "").strip("/")
    print(hpath)

    if "replay_mobile" in hpath:
        fps = 30
    else:
        fps = 25

    for file in tqdm(files):
        if not file.endswith(".mov"):
            continue

        res_path = os.path.join(output_path, hpath)

        if not os.path.exists(res_path):
            os.makedirs(res_path)

        frames_path = os.path.join(input_path, hpath, file.replace('.mov', "_aligned"))

        res_fpath = os.path.join(res_path, file)

        print(frames_path, res_fpath)
        os.system(f"ffmpeg -f image2 -r 30 -i {frames_path}/frame_det_00_%06d.bmp {res_fpath}")
