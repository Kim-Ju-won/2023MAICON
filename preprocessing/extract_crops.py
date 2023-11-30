import argparse
import json
import os
from os import cpu_count
from pathlib import Path

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from functools import partial
from glob import glob
from multiprocessing.pool import Pool


import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm

from utils import get_video_paths, get_method, get_method_from_name

def extract_video(video, root_dir, dataset, output_path):
    try:
        bboxes_path = os.path.join(root_dir, "boxes", os.path.splitext(os.path.basename(video))[0]+".mp4" ,os.path.splitext(os.path.basename(video))[0] + ".json")
        
        if not os.path.exists(bboxes_path) or not os.path.exists(video):
            return
        with open(bboxes_path, "r") as bbox_f:
            bboxes_dict = json.load(bbox_f)

        capture = cv2.VideoCapture(video)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        counter = 0
        count = 0
        for i in range(frames_num):
            if i >= 58 and count < 32 :
                capture.grab()
                #if i % 2 != 0:
                #    continue
                success, frame = capture.retrieve()
                if not success or str(i) not in bboxes_dict:
                    continue
                id = os.path.splitext(os.path.basename(video))[0]
                crops = []
                bboxes = bboxes_dict[str(i)]
                if bboxes is None:
                    continue
                else:
                    counter += 1
                for bbox in bboxes:
                    xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                    w = xmax - xmin
                    h = ymax - ymin
                    p_h = 0
                    p_w = 0
                    

                    if h > w:
                        p_w = int((h-w)/2)
                    elif h < w:
                        p_h = int((w-h)/2)

                    crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                    crop = cv2.resize(crop, (224, 224))
                    h, w = crop.shape[:2]
                    crops.append(crop)
                os.makedirs(os.path.join(output_path, video.split('\\')[1],video.split('\\')[2],id), exist_ok=True)
                for j, crop in enumerate(crops):
                    cv2.imwrite(os.path.join(output_path, video.split('\\')[1],video.split('\\')[2],id, "{}_{}.png".format(i, j)), crop)
                count+=1
            if count >= 32 : 
                break
        if counter == 0:
            print(video, counter)
    except Exception as e:
        print("Error:", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="MAICON", type=str,
                        help='Dataset MAICON')
    parser.add_argument('--data_path', default='../sample_data', type=str,
                        help='Videos directory')
    parser.add_argument('--output_path', default='../dataset', type=str,
                        help='Output directory')

    opt = parser.parse_args()
    print(opt) 
    dataset = 1
    
    os.makedirs(opt.output_path, exist_ok=True)
    #excluded_videos = os.listdir(os.path.join(opt.output_dir)) # Useful to avoid to extract from already extracted videos
    excluded_videos = os.listdir(opt.output_path)
    paths = get_video_paths(opt.data_path, dataset)
    
    with Pool(processes=cpu_count()-2) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(extract_video, root_dir=opt.data_path, dataset=dataset, output_path=opt.output_path), paths):
                pbar.update()
