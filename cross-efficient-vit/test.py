import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import os
import cv2
import numpy as np
import torch
from torch import nn, einsum

from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from cross_efficient_vit import CrossEfficientViT
from utils import transform_frame
import glob
from os import cpu_count
import json
from multiprocessing.pool import Pool
from progress.bar import Bar
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
from utils import custom_round, custom_video_round
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate
from transforms.albu import IsotropicResize
import yaml
import argparse

#########################
####### CONSTANTS #######
#########################

BASE_DIR = '../'
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set")


TEST_LABELS_PATH = os.path.join(BASE_DIR, "dataset/dfdc_test_labels.csv")

#########################
####### UTILITIES #######
#########################

def save_confusion_matrix(confusion_matrix):
  fig, ax = plt.subplots()
  im = ax.imshow(confusion_matrix, cmap="Blues")

  threshold = im.norm(confusion_matrix.max())/2.
  textcolors=("black", "white")

  ax.set_xticks(np.arange(2))
  ax.set_yticks(np.arange(2))
  ax.set_xticklabels(["original", "fake"])
  ax.set_yticklabels(["original", "fake"])
  
  ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

  for i in range(2):
      for j in range(2):
          text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", 
                         fontsize=12, color=textcolors[int(im.norm(confusion_matrix[i, j]) > threshold)])

  fig.tight_layout()
  plt.savefig(os.path.join(BASE_DIR, "confusion.jpg"))
  

def save_roc_curves(correct_labels, preds, model_name, accuracy, loss, f1):
  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')

  fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

  model_auc = auc(fpr, tpr)


  plt.plot(fpr, tpr, label="Model_"+ model_name + ' (area = {:.3f})'.format(model_auc))

  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.savefig(os.path.join(BASE_DIR, model_name +  "_" + opt.dataset + "_acc" + str(accuracy*100) + "_loss"+str(loss)+"_f1"+str(f1)+".jpg"))
  plt.clf()


def read_frames(video_path, videos):

    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0,1):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)

    # Select N frames from the collected ones
    video = {}
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            image = cv2.imread(os.path.join(video_path, frame_image))
            if image is not None:
                if key in video:
                    video[key].append(image)
                else:
                    video[key] = [image]
    videos.append((video, 0, video_path))




# Main body
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path', default='./models/efficientnet_checkpoint0_MAICON', type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='MAICON', 
                        help="Which dataset to use MAICON")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, default='./configs/architecture.yaml',
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--frames_per_video', type=int, default=30, 
                        help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Batch size (default: 32)")
    
    opt = parser.parse_args()
    print(opt)
    
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 
        
    if os.path.exists(opt.model_path):
        model = CrossEfficientViT(config=config)
        model.load_state_dict(torch.load(opt.model_path))
        model.eval()
        model = model.cuda()
    else:
        print("No model found.")
        exit()

    model_name = os.path.basename(opt.model_path)


    #########################
    ####### EXECUTION #######
    #########################


    OUTPUT_DIR = os.path.join(BASE_DIR, opt.dataset)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    NUM_CLASSES = 1
    preds = []

    mgr = Manager()
    paths = []
    videos = mgr.list()

    folders = [opt.dataset]

    # Read all videos paths
    for folder in folders:  
        for index, video_folder in enumerate(os.listdir(TEST_DIR)):
            paths.append(os.path.join(TEST_DIR, video_folder, video_folder[:-3]))

    # Read faces
    with Pool(processes=cpu_count()-1) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, videos=videos),paths):
                pbar.update()

    video_names = np.asarray([row[2] for row in videos])
    correct_test_labels = np.asarray([row[1] for row in videos])
    videos = np.asarray([row[0] for row in videos])
    preds = []
    print(video_names)
    print(correct_test_labels)

    # Perform prediction
    bar = Bar('Predicting', max=len(videos))

    videos_names = []


    for index, video in enumerate(videos):
        video_faces_preds = []
        video_name = video_names[index].split('\\')[2]
        videos_names.append(video_name)
        for key in video:
            faces_preds = []
            video_faces = video[key]
            for i in range(0, len(video_faces), opt.batch_size):
                faces = video_faces[i:i+opt.batch_size]
                faces = torch.tensor(np.asarray(faces))
                if faces.shape[0] == 0:
                    continue
                faces = np.transpose(faces, (0, 3, 1, 2))
                faces = faces.cuda().float()
                
                pred = model(faces)
                
                scaled_pred = []
                for idx, p in enumerate(pred):
                    scaled_pred.append(torch.sigmoid(p))
                faces_preds.extend(scaled_pred)
                
            current_faces_pred = sum(faces_preds)/len(faces_preds)
            face_pred = current_faces_pred.cpu().detach().numpy()[0]
            video_faces_preds.append(face_pred)
        bar.next()
        if len(video_faces_preds) > 1:
            video_pred = custom_video_round(video_faces_preds)
        else:
            video_pred = video_faces_preds[0]
        preds.append('fake' if video_pred >= 0.5 else 'real')
        
    bar.finish()

    answer = pd.DataFrame({
        'path': videos_names,
        'label':preds
    })

    answer.to_csv('MAICON_ANSWER.csv',index=False)

    sample_submission = pd.read_csv('../sample_submission.csv')
    sample_submissionf = pd.merge(sample_submission, answer, on='path', how='left')
    sample_submissionf['label_x'].fillna(sample_submission['label_y'], inplace=True)
    sample_submission.drop(['label_y'], axis=1, inplace=True)
    sample_submission.rename(columns={'label_x': 'label'}, inplace=True)
    sample_submission.to_csv('../sample_submission.csv')