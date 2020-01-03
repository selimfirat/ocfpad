import os

from tqdm import tqdm

input_path = "/mnt/storage2/pad/videos"
output_path = "/mnt/storage2/pad/frames"

docker_path = "/home/openface-build/build/bin"

videos_path = "/home/openface-build/videos"

for root, dirs, files in tqdm(os.walk(input_path)):
    hpath = root.replace(input_path, "").strip("/")
    print(hpath)

    for file in tqdm(files):
        if not file.endswith(".mov"):
            continue

        fpath = os.path.join(videos_path, hpath, file)

        exec_command = f"docker exec -it openface bash -c 'cd {docker_path}; ./FeatureExtraction -f {fpath} -simsize 224 -simalign'"

        os.system(exec_command)

        aligned_frames_path = os.path.join(docker_path, "processed", file.replace(".mov", "_aligned"))

        res_path = os.path.join(output_path, hpath)
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        cp_command = f"docker cp openface:{aligned_frames_path} {res_path}"

        os.system(cp_command)
