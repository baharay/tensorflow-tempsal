import argparse
import os

import cv2

import numpy as np

from loss import *

parser = argparse.ArgumentParser()
# Evaluate the model on C:/Users/devcloud/bahar_projects/git-tf/tensorflow-tempsal/tf-test/dataset/ExtractedFolder/salicon/images/val/
parser.add_argument('--predictions_dir', default="../my_predictions/", type=str)

parser.add_argument('--dataset_dir', default="C:/Users/devcloud/bahar_projects/git-tf/tensorflow-tempsal/tf-test/dataset/ExtractedFolder/salicon/maps/val/",
                    type=str)

parser.add_argument('--time_slices', default=1, type=int)

parser.add_argument('--no_workers', default=4, type=int)

parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--results_dir', default="../data/predictions/", type=str)

args = parser.parse_args()

# Define the paths to the folders containing ground truths and predictions

gt_folder = args.dataset_dir

pred_folder = args.predictions_dir

# Initialize lists to store the metric values for each image

kldiv_values = []

cc_values = []

sim_values = []

# Load and compare images
a=0
for image_name in os.listdir(gt_folder):
    # Load ground truth image
    try:
        

        gt_path = os.path.join(gt_folder, image_name)

        gt = cv2.imread(gt_path, 0)

        # Load prediction image

        pred_path = os.path.join(pred_folder, image_name)

        pred = cv2.imread(pred_path, 0)

        # Compare ground truth and prediction using the metrics

        m1 = kldiv(pred, gt)

        m2 = cc(pred, gt)

        m3 = similarity(pred, gt)

        # Append metric values to respective lists

        kldiv_values.append(m1)

        cc_values.append(m2)

        sim_values.append(m3)
        
    except:
        continue
    a+=1

# Calculate mean values for each metric

mean_kldiv = np.mean(kldiv_values)

mean_cc = np.mean(cc_values)

mean_sim = np.mean(sim_values)

# Display mean values for each metric

print("Mean kldiv ", mean_kldiv)

print("Mean cc ", mean_cc)

print("Mean sim: ", mean_sim)