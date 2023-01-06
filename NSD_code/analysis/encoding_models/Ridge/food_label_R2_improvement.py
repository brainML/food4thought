"""
This file runs ridge regression encoding model on all labels and all non-food labels and outputs the improvement in R2 by including food labels.

Example run: 
python food_label_R2_improvement.py -s 1 -v voxel_path.npy -t stim_id_path.npy -c coco_prop_path.npy -o output.npy

-s : subject
-v : path to preprocessed voxels in the format of images x cortical voxels
-t : path to list of stimulus coco ids in the order of voxel_path.npy
-c : path to table of coco proportions organized by images x coco labels, where images are in same order as stim_id_path.npy
-o : desired output path

"""

import csv
import numpy as np
import pandas as pd
import random
import scipy
import matplotlib.pyplot as plt
from ridge_tools import *
import sys
import argparse


cocolabels_path = 'coco_labels.txt'

coco_labels = []
with open(cocolabels_path, "r") as f:
    for line in f:
        coco_labels.extend(line.split('\n'))
coco_labels = [i for i in coco_labels if len(i) >= 1]


def getImprovedR2(subj, voxels_path, stimulus_path, coco_proportions_path, output_path):

    voxels_subj = np.load(voxels_path)
    stimulus = np.load(stimulus_path)
    coco_proportions = np.load(coco_proportions_path)

    food_labels = ['cup', 'fork', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'donut', 'pizza', 'cake']
    food_label_ids = [coco_labels.index(i) for i in food_labels]
    non_food_label_ids = [i for i in range(80) if not(i in food_label_ids)]

    coco_max_proportions = np.max(coco_proportions[:, food_label_ids], axis=1)
    threshold = 0.15
    coco_max_proportions_ids = np.where(coco_max_proportions > threshold)[0]

    min_threshold = 0.1
    coco_non_food_proportions_ids = np.where(coco_max_proportions <= min_threshold)[0]

    coco_rand_non_food_proportions_ids = np.random.choice(len(coco_non_food_proportions_ids), size=len(coco_max_proportions_ids))
    coco_food_and_nonfood_ids = np.concatenate([coco_max_proportions_ids,coco_rand_non_food_proportions_ids])

    print("running on non food labels")
    X = coco_proportions[:, non_food_label_ids][coco_food_and_nonfood_ids]
    Y = voxels_subj[coco_food_and_nonfood_ids]
    res_non_food_labels = fit_predict(Y, X, n_splits=10,n_inner_splits=10)

    print("running on all labels")
    X = coco_proportions[coco_food_and_nonfood_ids]
    Y = voxels_subj[coco_food_and_nonfood_ids]
    res_all_labels = fit_predict(Y, X, n_splits=10,n_inner_splits=10)

    res_R2_diff = res_all_labels[2] - res_non_food_labels[2]

    with open(output_path, 'wb') as f:
        np.save(f, res_R2_diff)


if __name__ == "__main__":
    #Run one sided significance tests for given subject. 

    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', '-s', help="", type=int)
    parser.add_argument('--voxels_path', '-v', help="", type=str)
    parser.add_argument('--stim_path', '-t', help="", type=str)
    parser.add_argument('--coco_prop_path', '-c', help="", type=str)
    parser.add_argument('--out_path', '-o', help="", type=str)

    args = parser.parse_args()
    subj = args.subj
    voxels_path = args.voxels_path
    stimulus_coco_ids_subj_path = args.stim_path
    coco_proportions_path = args.coco_prop_path
    output_path = args.out_path

    print("SUBJECT INPUTTED: ", subj)
    getImprovedR2(subj, voxels_path, stimulus_path, coco_proportions_path, output_path)


    voxels_path = '../../../../yuanw3/project_outputs/NSD/output/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj0{0}.npy'.format(subj)
    stimulus_path = '../../../../yuanw3/project_outputs/NSD/output/coco_ID_of_repeats_subj0{0}.npy'.format(subj)
    coco_proportions_path = '../../../../yuanw3/project_outputs/NSD/features/subj{0}/cat.npy'.format(subj)

