
import numpy as np 
from numpy.linalg import inv
import cortex
import nibabel
import csv
from scipy.stats import zscore
from utils import get_hrf, hrf_convolve
from utils import load_and_process
from utils import get_contrast_and_Tstat, getPValue_preFDR, fdr_correction

data_path = ('../data/')


total_time = dict(BW = 240, icons = 174, natural = 510)

def load_data_and_stim(surface, xfm, stim_path, data_name, data_runs, run_map, cond_map, cname, 
                       SMOOTHING = 0, DETREND = True, START_TRIM_NUM = 6, END_TRIM_NUM = 6, 
                       start_delay = 12, TR = 1.5):
    
    params = dict(surface = surface, 
                  xfm = xfm,
                  cname = cname,
                  SMOOTHING = SMOOTHING,
                  DETREND = DETREND,
                  START_TRIM_NUM = START_TRIM_NUM,
                  END_TRIM_NUM = END_TRIM_NUM,
                  start_delay = start_delay,
                  TR = TR)

    # define stim times and names
    stim_time = dict()
    stim_name = dict()

    for run_type, runs in run_map.items():
        stim_time[run_type] = dict()
        stim_name[run_type] = dict()
        for i_run, name_run in enumerate(runs):
            stim_time[run_type][i_run] = []
            stim_name[run_type][i_run] = []
            with open('{}{}_{}.csv'.format(stim_path,run_type,name_run), newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    stim_time[run_type][i_run].append(float(row['onset'])+start_delay)
                    stim_name[run_type][i_run].append(row['name'])

    stim_name_food = dict()

    for c, cnames in stim_name.items():
        stim_name_food[c] = dict()
        for r, rnames in cnames.items():
            stim_name_food[c][r] = []
            for s in rnames:
                for key, val in cond_map[c].items():
                    if key in s:
                        break
                stim_name_food[c][r].append(val)


    # simplify a bit
    task_names = ['BW']
    cond_names = dict([(t,sorted(set(cond_map[t].values()))) for t in task_names])

    print(cond_names)

    design_matrix_05 = dict()
    times = dict([(k,np.arange(0,total_time[k],0.5)) for k in task_names])

    for c, cnames in stim_name_food.items():
        design_matrix_05[c] = dict()
        for r, rnames in cnames.items():
            design_matrix_05[c][r] = np.zeros((len(times[c]),len(cond_names[c])))
            for i_stim, onset in enumerate(stim_time[c][r]):
                idx = np.where(times[c] == onset)[0]
                assert len(idx)==1
                design_matrix_05[c][r][idx,cond_names[c].index(rnames[i_stim])] = 1
    
    # get hrf 
    
    t_hrf, hrf = get_hrf(tr=TR)


    # get convolved design


    convolved_design = dict()
    sampled_design = dict()
    times_TR = dict([(k,np.arange(0,total_time[k],TR)) for k in task_names])

    for c, cdesign in design_matrix_05.items():
        convolved_design[c] = dict()
        sampled_design[c] = dict()
        for r, rdesign in cdesign.items():
            sampled_design[c][r] = np.zeros((len(times_TR[c]),rdesign.shape[1]))
            for itime, ttime in enumerate(times_TR[c]):
                idx = np.where( (times[c]>=ttime) * (times[c]<ttime+TR) )[0]
                assert (len(idx)==3) + (itime == len(times_TR[c])-1)
                sampled_design[c][r][itime] = np.mean(rdesign[idx],axis=0)
            convolved_design[c][r] = hrf_convolve(sampled_design[c][r], hrf) 


    # load data finally!

    func_files = dict()
    for c,runs in data_runs.items():
        func_files[c] = dict([(idx, data_name.format(c,r)) for idx,r in enumerate(data_runs[c])])



    data = dict([(run_name,load_and_process(file, start_trim = START_TRIM_NUM, end_trim = END_TRIM_NUM,
                                     do_detrend= DETREND, smoothing_factor = SMOOTHING,
                                     do_zscore = False))
                      for run_name, file in func_files[cname].items()])

    run_length = dict([(run_name,d.shape[0]) for run_name,d in data.items()])
    original_length = dict([(run_name,d.shape[0]+START_TRIM_NUM+END_TRIM_NUM) for run_name,d in data.items()])

    print("data shape: ",[(run_length,d.shape) for run_length,d in data.items()])
    print("runs length:", run_length)

    if END_TRIM_NUM>0:
        stim_X = dict([(r,convolved_design[cname][r][START_TRIM_NUM:-END_TRIM_NUM]) 
                       for r in convolved_design[cname].keys()])
    else:
        stim_X = dict([(r,convolved_design[cname][r][START_TRIM_NUM:]) 
                       for r in convolved_design[cname].keys()])


    print("stim shape: ",[(run_length,d.shape) for run_length,d in stim_X.items()])
    
    mask = cortex.db.get_mask(surface , xfm, 'thick')

    return params, data, stim_X, cond_names[cname], mask


def run_analysis(contrasts, params, data, stim_X, cond_names, mask, load_NSD=True, mask_MNI=0.5, do_mni=True,
                p_val_threshold=0.05):

    run_keys = sorted(stim_X.keys())
    surface = params['surface']
    xfm = params['xfm']

    masked_data = np.vstack([np.nan_to_num(zscore(data[k][:,mask])) for k in run_keys])
    X = np.vstack([stim_X[k] for k in run_keys])
    contrast = dict()

    for contrast_name, c in contrasts.items():
        
        __, contrast['tval_{}'.format(contrast_name)] = get_contrast_and_Tstat(c,X,masked_data)
        p_stat = getPValue_preFDR(c,X,masked_data)
        corrected_p_stat = fdr_correction(p_stat, alpha=p_val_threshold, method="bh", axis=0)[1]
        contrast['pval_{}'.format(contrast_name)] = (corrected_p_stat<p_val_threshold)*1.0
        print('ran contrast {}'.format(contrast_name))

    if do_mni:
        
        vols=dict()
        for s,t in contrast.items():
            vols[s] = cortex.Volume(t,surface,xfm)
    
        mni_transform_food = cortex.db.get_mnixfm(subject=surface, xfm=xfm)
    
        for s,t in vols.items():
            print(s)
            mni_vols = cortex.mni.transform_to_mni(t,mni_transform_food)
            contrast['mni_{}'.format(s)] = mni_vols.get_data().T
        
    
    return contrast, params






