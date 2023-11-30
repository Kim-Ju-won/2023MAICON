import json
import os
from glob import glob
from pathlib import Path
import cv2
banned_folders = ["boxes"]
def get_video_paths(data_path, dataset, excluded_videos=[]):
    videos_folders = os.listdir(data_path)
    videos_paths = []
    for folder in videos_folders:
        if folder not in banned_folders:
            if folder != "test_set" : 
                folder_path = os.path.join(data_path, folder)
                inner_folders = os.listdir(folder_path)
                for inner in inner_folders:
                    final_path = os.path.join(folder_path,inner)
                    all_files = os.listdir(final_path)
                    videos_paths.extend([os.path.join(final_path, file) for file in all_files if os.path.isfile(os.path.join(final_path, file))])
            else : 
                folder_path = os.path.join(data_path, folder)

                all_files = os.listdir(folder_path)
                videos_paths.extend([os.path.join(folder_path, file) for file in all_files if os.path.isfile(os.path.join(folder_path, file))])
    return videos_paths

def resize(image, image_size):
    try:
        return cv2.resize(image, dsize=(image_size, image_size))
    except:
        return []

def get_original_video_paths(root_dir, basename=False):
    originals = set()
    originals_v = set()
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "REAL":
                original = k
                originals_v.add(original)
                originals.add(os.path.join(dir, original))
    originals = list(originals)
    originals_v = list(originals_v)

    return originals_v if basename else originals


        
def get_method_from_name(video):
    methods = ["youtube", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "MAICON"]
    for method in methods:
        if method in video:
            return method

def get_method(video, data_path):
    methods = []
    for dataset_path in ['training_set','validation_set']:
        for attribute in ["fake", "real"]:
            methods.extend(os.listdir(os.path.join(data_path, dataset_path, attribute)))
    methods.extend(os.listdir(os.path.join(data_path, "test_set")))
    selected_method = ""
    for method in methods:
        if method in video:
            selected_method = method
            break
    return selected_method

def get_original_with_fakes(root_dir):
    pairs = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "FAKE":
                pairs.append((original[:-4], k[:-4] ))

    return pairs


def get_originals_and_fakes(root_dir):
    originals = []
    fakes = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            if v["label"] == "FAKE":
                fakes.append(k[:-4])
            else:
                originals.append(k[:-4])

    return originals, fakes

