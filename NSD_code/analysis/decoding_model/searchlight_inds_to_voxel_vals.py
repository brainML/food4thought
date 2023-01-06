"""
This file runs gets searchlight voxel values for the relevant shared data.

Example run: 
python searchlight_inds_to_voxel_vals.py -s 1 -v voxel_path.npy -t stim_id_path.npy -i searchlight_inds.npy -o output.npy

-s : subject
-v : path to preprocessed voxels in the format of images x cortical voxels
-t : path to list of stimulus coco ids in the order of voxel_path.npy
-i : path to searchlight indices for each voxel
-o : desired output path

"""

import csv
import numpy as np
import sys
import argparse

shared_cocoID_labels_set = set(np.load('/data/shared_1000_cocoId.npy'))

def getVoxels(subj, voxels_subj_path, stimulus_coco_ids_subj_path, searchlight_inds_path, output_path):
    voxels_subj = np.load(voxels_subj_path)
    stimulus_coco_ids_subj = np.load(stimulus_coco_ids_subj_path)
    print("loaded data")
    
    searchlight_inds = np.load(searchlight_inds_path)
    
    searchlight_voxels = np.zeros((len(shared_cocoID_labels_set), searchlight_inds.shape[0], searchlight_inds.shape[1]))
    searchlight_valid_inds = np.sign(np.sign(searchlight_inds) + 1)
    searchlight_fin_inds = np.multiply(searchlight_inds,searchlight_valid_inds).astype(int)
    
    count = 0
    for i in range(len(stimulus_coco_ids_subj)):
        if stimulus_coco_ids_subj[i] in shared_cocoID_labels_set:
            print(count)
            searchlight_voxels[count] = voxels_subj[i][searchlight_fin_inds]
            count += 1
    
    print('done computation. saving')
    searchlight_voxels_path = output_path 
    np.save(searchlight_voxels_path, searchlight_voxels)
    return searchlight_voxels

if __name__ == "__main__":
    #Get voxel values corresponding to relevant shared image data and input indices.  

    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', '-s', help="", type=int)
    parser.add_argument('--voxels_path', '-v', help="", type=str)
    parser.add_argument('--stim_path', '-t', help="", type=str)
    parser.add_argument('--search_inds_path', '-i', help="", type=str)
    parser.add_argument('--out_path', '-o', help="", type=str)

    args = parser.parse_args()
    subj = args.subj
    voxels_subj_path = args.voxels_path
    stimulus_coco_ids_subj_path = args.stim_path
    searchlight_inds_path = args.search_inds_path
    output_path = args.out_path

    voxels_subj_path = '../../../../yuanw3/project_outputs/NSD/output/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj0{0}.npy'.format(subj)
    stimulus_coco_ids_subj_path = '../../../../yuanw3/project_outputs/NSD/output/coco_ID_of_repeats_subj0{0}.npy'.format(subj)
    searchlight_inds_path = '../data/searchlight_subj{0}.npy'.format(subj)
    output_path = '../data/code_test/encoding1/decoding_test.npy'.format(subj)

    print("SUBJECT INPUTTED: ", subj)
    getVoxels(subj, voxels_subj_path, stimulus_coco_ids_subj_path, searchlight_inds_path, output_path)


    
