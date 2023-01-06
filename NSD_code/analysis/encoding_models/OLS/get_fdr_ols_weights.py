"""
This file runs 1 sided significance tests with various optional conditions. By default, this file runs food vs non-food significance test.

Example run: 
python get_fdr_ols_weights.py -s 1 -v voxel_path.npy -t stim_id_path.npy -o output.npy

-s : subject
-v : path to preprocessed voxels in the format of images x cortical voxels
-t : list of stimulus coco ids in the order of voxel_path.npy
-o : desired output path

To tweak optional conditions, look at main function. 
"""

import numpy as np 
import pandas as pd
from numpy.linalg import pinv
from sklearn.metrics import r2_score
from scipy import stats
from scipy.stats import t as tdistribution
import sys
import argparse


labels_path = '/data/shared_1000_all_labels_matrix.csv'
labels_data = pd.read_csv(labels_path)


def fdr_correction(p_values, alpha=0.05, method="by", axis=None):
    """
    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.
    Modified from the code at https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html
    Args:
        p_values: The p_values to correct.
        alpha: The error rate to correct the p-values with.
        method: one of by (for Benjamini/Yekutieli) or bh for Benjamini/Hochberg
        axis: Which axis of p_values to apply the correction along. If None, p_values is flattened.
    Returns:
        indicator_alternative: An boolean array with the same shape as p_values_corrected that is True where
            the null hypothesis should be rejected
        p_values_corrected: The p_values corrected for FDR. Same shape as p_values
    """
    p_values = np.asarray(p_values)

    shape = p_values.shape
    if axis is None:
        p_values = np.reshape(p_values, -1)
        axis = 0
    if axis < 0:
        axis += len(p_values.shape)
        if axis < 0:
            raise ValueError("axis out of bounds")

    indices_sorted = np.argsort(p_values, axis=axis)
    p_values = np.take_along_axis(p_values, indices_sorted, axis=axis)

    correction_factor = np.arange(1, p_values.shape[axis] + 1) / p_values.shape[axis]
    correction_factor_shape = [1] * len(p_values.shape)
    correction_factor_shape[axis] = len(correction_factor)
    correction_factor = np.reshape(correction_factor, correction_factor_shape)

    if method == "bh":
        pass
    elif method == "by":
        c_m = np.sum(1 / np.arange(1, p_values.shape[axis] + 1))
        correction_factor = correction_factor / c_m
    else:
        raise ValueError("Unrecognized method: {}".format(method))

    # set everything left of the maximum qualifying p-value
    indicator_alternative = p_values <= correction_factor * alpha
    indices_all = np.reshape(
        np.arange(indicator_alternative.shape[axis]),
        (1,) * axis
        + (indicator_alternative.shape[axis],)
        + (1,) * (len(indicator_alternative.shape) - 1 - axis),
    )
    indices_max = np.nanmax(
        np.where(indicator_alternative, indices_all, np.nan), axis=axis, keepdims=True
    ).astype(int)
    indicator_alternative = indices_all <= indices_max
    del indices_all

    p_values = np.clip(
        np.take(
            np.minimum.accumulate(
                np.take(
                    p_values / correction_factor,
                    np.arange(p_values.shape[axis] - 1, -1, -1),
                    axis=axis,
                ),
                axis=axis,
            ),
            np.arange(p_values.shape[axis] - 1, -1, -1),
            axis=axis,
        ),
        a_min=0,
        a_max=1,
    )

    indices_sorted = np.argsort(indices_sorted, axis=axis)
    p_values = np.take_along_axis(p_values, indices_sorted, axis=axis)
    indicator_alternative = np.take_along_axis(
        indicator_alternative, indices_sorted, axis=axis
    )

    return np.reshape(indicator_alternative, shape), np.reshape(p_values, shape)

def getDesignMatrixY(nsd_voxels,nsd_stimulus,labeled_data,exclude_img = False,exclude_img_label = ""):
    
    stimulus_coco_ids_indices = dict(zip(nsd_stimulus, [i for i in range(len(nsd_stimulus))]))    
    stim_labels_indices = [stimulus_coco_ids_indices[i] for i in list(labeled_data.cocoId) if list(labeled_data.cocoId)]

    if exclude_img:
        coco_ids = list(labeled_data.cocoId)
        exclude_label_vals = list(labeled_data[exclude_img_label])
        stim_labels_indices = [stimulus_coco_ids_indices[coco_ids[i]] for i in range(len(coco_ids)) if exclude_label_vals[i] == 0]
    
    nsd_ordered_stim = nsd_stimulus[stim_labels_indices]
    nsd_ordered_voxels = nsd_voxels[stim_labels_indices]
    
    if exclude_img:
        new_data = labeled_data[labeled_data[exclude_img_label] == 0].values
        new_data = new_data[:, 9::]
        X_design_matrix = np.array(new_data).astype('float64')
    else:
        X_design_matrix = labeled_data.iloc[:,-16:].values
    Y = nsd_ordered_voxels
    
    ind = np.abs(nsd_ordered_voxels).sum(1) == 0
    ind = ind==False
    X_design_matrix = X_design_matrix[ind]
    Y = Y[ind]
    print('retained {} images'.format(ind.sum))
    
    return (X_design_matrix,Y)

def getXTX(X):
    return pinv(np.dot(X.T,X))


def getW(X,Y, xTx_pinv):
    w = np.dot(xTx_pinv, np.dot(X.T, Y))
    return w

def getMSE(X,Y,w):
    pred_Y = np.dot(X, w)
    mse = np.mean(np.square(pred_Y - Y), axis=0)
    return mse 

def getPValue_preFDR(c,X,Y):
    xTx_pinv = getXTX(X)
    w = getW(X,Y,xTx_pinv)
    mse = getMSE(X,Y,w)
    
    t_stat_num = np.dot(c.T, w)
    t_stat_denom = np.multiply(np.abs(mse),np.sqrt(np.linalg.multi_dot([c.T, xTx_pinv, c])))
    t_stat = np.divide(t_stat_num, t_stat_denom)
    
    p_statistic = 1 - tdistribution.cdf(t_stat,1000)
    return p_statistic

def getTstat(c,X,Y):
    xTx_pinv = getXTX(X)
    w = getW(X,Y,xTx_pinv)
    mse = getMSE(X,Y,w)
    
    t_stat_num = np.dot(c.T, w)
    t_stat_denom = np.multiply(np.abs(mse),np.sqrt(np.linalg.multi_dot([c.T, xTx_pinv, c])))
    t_stat = np.divide(t_stat_num, t_stat_denom)
    return t_stat

def one_sided_t_test(subj, voxels_path, stimulus_coco_ids_subj_path, output_path, save_p_stat = True, label_v_all = "food", exclude_img = False, exclude_img_label = ""):
    """
    This function performs a one sided t_test (i.e. food vs non-food labels) and returns either FDR corrected p_values, or t_values, as specified. 
    Args:
        subj: The subject whos' data to perform the 1 sided test on.
        voxels_path: Path to subject's preprocessed beta response data in the format of (images seen x cortical surface voxels).
        stimulus_coco_ids_subj_path: Ordered list of stimulus IDs corresponding to voxel path
        output_path: Path to save output
        save_p_stat: Mark as true if interested in outputting corrected p_values, false if interested in t_values.
        label_v_all: Desired label x to run a one sided significance test of x vs non-x  
        exclude_img: Marked as true if you want to exclude images when running the test that are labeled 1 for a certain value
        exclude_img_label: For all images where this label name is marked as 1, the images won't be included in the test. exclude_img must be True for this to work
            
        Possible label names (for label_v_all and exclude_img_label): "indoor", "outdoor", "ambiguous-location", "plant", "human-face", "human-body", "animal-face",
             "animal-body", "food", "drink", "food-related", "faux-food", "zoom", "reach", "large-scale-scene", "object"
    Saves:
        FDR corrected p_values or t_values to output path file
    """
    voxels = np.load(voxels_path)
    stimulus_coco_ids_subj = np.load(stimulus_coco_ids_subj_path)

    voxels[np.where(voxels is None)] = 0
    voxels[np.where(voxels == np.nan)] = 0
    voxels[np.where(np.isnan(voxels))] = 0

    X,Y = getDesignMatrixY(voxels, stimulus_coco_ids_subj, labels_data, exclude_img, exclude_img_label)

    labels = ["indoor", "outdoor", "ambiguous-location", "plant", "human-face", "human-body", "animal-face",
             "animal-body", "food", "drink", "food-related", "faux-food", "zoom", "reach", "large-scale-scene", 
             "object"]
    label = label_v_all
    labels_i = dict([(v,i) for i,v in enumerate(labels)])
    c = np.array([-1/(len(labels)-1) for i in range(len(labels))])

    c[labels_i[label]] = 1

    if save_p_stat: 
        p_stat = getPValue_preFDR(c,X,Y)
        p_stat[np.where(np.isnan(p_stat))] = 0.9999
        correct_p_stat = fdr_correction(p_stat, alpha=0.05, method="bh", axis=0)[1]
        with open(output_path, 'wb') as f:
            print(output_path)
            np.save(f, correct_p_stat)
            print("Saved p-values to ", output_path)
    else:
        t_stat = getTstat(c,X,Y)
        with open(output_path, 'wb') as f:
            print(output_path)
            np.save(f, t_stat)
            print("Saved t-values to ", output_path)





if __name__ == "__main__":
    #Run one sided significance tests for given subject. 

    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', '-s', help="", type=int)
    parser.add_argument('--voxels_path', '-v', help="", type=str)
    parser.add_argument('--stim_path', '-t', help="", type=str)
    parser.add_argument('--out_path', '-o', help="", type=str)

    args = parser.parse_args()
    subj = args.subj
    voxels_path = args.voxels_path
    stimulus_coco_ids_subj_path = args.stim_path
    output_path = args.out_path

    one_sided_t_test(subj, voxels_path, stimulus_coco_ids_subj_path, output_path, save_p_stat = False, label_v_all = "food", exclude_img = False, exclude_img_label = "")