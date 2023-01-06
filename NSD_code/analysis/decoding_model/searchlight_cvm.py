"""
This file runs SVM cross validation on searchlight data..

Example run: 
python food_label_R2_improvement.py -s 1 -v voxel_path.npy -t stim_id_path.npy -o output.npy

-s : subject
-v : path to searchlight voxels
-t : path to list of stimulus coco ids in the order of voxel_path.npy
-o : desired output path

"""

import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
import json 
import sys
import argparse
 
shared_cocoID_labels_set = set(np.load('/data/shared_1000_cocoId.npy'))
f = open('/data/shared_1000_cross_val_food_face_id_lists.json')
food_labels_dict = json.load(f)
food_ids_set = set(food_labels_dict['food'])
no_food_ids_set = set(food_labels_dict['no_food'])



def getVoxels(subj, stimulus_coco_ids_subj_path, searchlight_voxels_path):
    print("Getting searchlight voxels")
    stimulus_coco_ids_subj = np.load(stimulus_coco_ids_subj_path)

    count = 0
    ids_to_use_food = []
    food_labels = []

    for i in range(len(stimulus_coco_ids_subj)):
        coco_id_i = stimulus_coco_ids_subj[i]
        if coco_id_i in shared_cocoID_labels_set:
            
            #food
            if coco_id_i in food_ids_set.union(no_food_ids_set):
                if coco_id_i in food_ids_set:
                    food_labels.append(1)
                else:
                    food_labels.append(0)
                ids_to_use_food.append(count)
                
            count += 1
        
    searchlight_voxels = np.load(searchlight_voxels_path)
    food_searchlight_voxels = searchlight_voxels[ids_to_use_food, :, :]
    
    return food_searchlight_voxels,food_labels


def getSVMScores(subj, stimulus_coco_ids_subj_path, searchlight_voxels_path, output_path):
    print("Getting cross val scores")
    food_searchlight_voxels,food_labels = getVoxels(subj, stimulus_coco_ids_subj_path, searchlight_voxels_path)

    clf = svm.SVC(kernel='linear', C=1, random_state=42)
    res = np.zeros((food_searchlight_voxels.shape[1],1))
    count = 0
    for i in range(food_searchlight_voxels.shape[1]):
        if i % 100 == 0:
            print("food, ", i)
        
        curr = food_searchlight_voxels[:,i,:]
        curr_mean = curr.mean(1)
        curr_mean = curr_mean.reshape((-1,1))
        curr_voxels_final = curr - curr_mean
        scores = cross_val_score(clf, curr_voxels_final, food_labels, cv=5)
        res[count] = np.mean(scores)
        count += 1

    print("saving output to: ", output_path)
    path = output_path
    np.save(path, res)


if __name__ == "__main__":
    #Run one sided significance tests for given subject. 

    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', '-s', help="", type=int)
    parser.add_argument('--voxels_path', '-v', help="", type=str)
    parser.add_argument('--stim_path', '-t', help="", type=str)
    parser.add_argument('--out_path', '-o', help="", type=str)

    args = parser.parse_args()
    subj = args.subj
    searchlight_voxels_path = args.voxels_path
    stimulus_coco_ids_subj_path = args.stim_path
    output_path = args.out_path

    stimulus_coco_ids_subj_path = '../../../../yuanw3/project_outputs/NSD/output/coco_ID_of_repeats_subj0{0}.npy'.format(subj)
    searchlight_voxels_path = '../data/searchlight_voxel_vals_subj{0}.npy'.format(subj)
    output_path = '../data/test_searchlight2_scores_food_submean_s{0}.npy'.format(subj)

    getSVMScores(subj, stimulus_coco_ids_subj_path, searchlight_voxels_path, output_path)


    
    
