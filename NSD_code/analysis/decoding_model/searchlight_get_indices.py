"""
This file gets each voxel's corresponding searchlight voxel indices and outputs a matrix of num_voxels * searchlight_size

Example run: 
python searchlight_get_indices.py -s 1 -m mask_path.npy -o output_path.npy

-s : subject
-m : cortical mask
-c : path to table of coco proportions organized by images x coco labels, where images are in same order as stim_id_path.npy
-o : desired output path

"""


import cortex
import numpy as np
import matplotlib.pyplot as plt

from cortex.options import config
from scipy.sparse import csr_matrix

import json
import sys
import argparse

#get searchlight indices 
def get_indices(subj, mask_path, output_path):
    surface = "subj0" + str(subj)
    xfm = "func1pt8_to_anat0pt8_autoFSbbr"
    
    mask = np.load(mask_path)

    mask_with_idx = np.zeros(mask.shape)
    mask_with_idx[mask==False] = np.nan
    mask_with_idx[mask==True] = np.arange(mask.sum())

    mapper = cortex.get_mapper(surface, xfm, 'trilinear', layers=32, recache=True)
    tmp = np.zeros(mask.shape)
    voxel_vol = cortex.Volume(np.swapaxes(tmp, 0, 2),surface,xfm)

    nc = mapper(voxel_vol).data.shape[0]
    nc1 = int(np.floor(nc/2)) # for efficiency
    nc2 = nc-nc1

    all_idx_1 = np.zeros((mask.sum(),nc1), dtype=bool) # (#voxel x nc1) 
    all_idx_2 = np.zeros((mask.sum(),nc2), dtype=bool) # (#voxel x nc2) 
    all_vox = np.zeros((mask.sum(),mask.sum())) # initiate a voxel adjacency matrix


    print("STARTING VOXEL ASSIGNMENT")
    for j in np.arange(mask.sum()): # for each voxel j
        tmp = np.zeros(mask.shape) # temporary 3d volume
        tmp[mask_with_idx==j] = 10 # set that voxel to be true in the volume 
        voxel_vol = cortex.Volume(np.swapaxes(tmp, 0, 2),surface,xfm) # make a volume object out of it
        tmp = mapper(voxel_vol).data > 0 # make a "binary" array where vertexes are on for that voxel 
        tmp = tmp.astype(bool)
        all_idx_1[j,tmp[:nc1]==True] = 1 # boolean 
        all_idx_2[j,tmp[nc1:]==True] = 1 
        if j%1000==0:
            print(j, tmp.sum())

    for j in np.arange(mask.sum()): # for each voxel
        non_zero_1 = all_idx_1[j]>0 # boolean array of non zero vertexs for that voxel
        sub_mat_1 = all_idx_1[:,non_zero_1].sum(1) # an array of voxel that are adjacent to those indexes
        all_vox[j,sub_mat_1>0] = 1 # set the adjacent voxels to positive in the matrix
        non_zero_2 = all_idx_2[j]>0 # repeat the same process for the second half
        sub_mat_2 = all_idx_2[:,non_zero_2].sum(1)
        all_vox[j,sub_mat_2>0] = 1
        if j%1000==0:
            print(j)

    max_n_neighbors = int(np.max(all_vox.sum(1)))+1 # max number of neighbors any voxel has
    print(max_n_neighbors)

    all_vox_neighbors = np.zeros((mask.sum(),max_n_neighbors))-1 # initiate a array of -1 to store neighbors

    print("STARTING NEIGHBORS ASSIGNMENT")
    for j in np.arange(mask.sum()): # for each voxel
        tmp = np.where(all_vox[j]>0)[0] # find the indexs of its neighboring voxels
        tmp = np.concatenate([tmp,np.array([j])]) # concatenate the indexes with itself
        all_vox_neighbors[j,0:len(tmp)] = tmp # store it in the array

    print("all vox neighbors size: ", all_vox_neighbors.shape)

    np.save(output_path,all_vox_neighbors)

if __name__ == "__main__":
    #process args
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', '-s', help="", type=int)
    parser.add_argument('--mask_path', '-m', help="", type=str)
    parser.add_argument('--out_path', '-o', help="", type=str)

    args = parser.parse_args()
    subj = args.subj
    mask_path = args.mask_path
    output_path = args.out_path

    get_indices(subj, mask_path, output_path)



