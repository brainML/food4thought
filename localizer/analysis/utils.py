import numpy as np 
from scipy.stats import gamma
import nibabel
from scipy.stats import zscore
from scipy import signal # for detrending
from scipy.ndimage import gaussian_filter # for smoothing
from numpy.linalg import pinv
from scipy import stats
from scipy.stats import t as tdistribution



def hrf_convolve(X, hrf):
    """Convolve a design matrix with an appropriate HRF"""
    m, n = X.shape
    Xhrf = np.zeros_like(X)
    for i in range(n):
        tmp = np.convolve(X[:,i], hrf, mode='full')[:m]
        Xhrf[:,i] = tmp
    return Xhrf

def get_hrf(shape='twogamma', tr=1, pttp=5, nttp=15, pos_neg_ratio=6, onset=0, pdsp=1, ndsp=1, t=None):
    """Create canonical hemodynamic response filter
    
    Parameters
    ----------
    shape : string, {'twogamma'|'boynton'}
        HRF general shape {'twogamma' [, 'boynton']}
    tr : scalar
        HRF sample frequency, in seconds (default = 2)
    pttp : scalar
        time to positive (response) peak in seconds (default = 5)
    nttp : scalar
        Time to negative (undershoot) peak in seconds (default = 15)
    pos_neg_ratio : scalar
        Positive-to-negative ratio (default: 6, OK: [1 .. Inf])
    onset : 
        Onset of the HRF (default: 0 secs, OK: [-5 .. 5])
    pdsp : 
        Dispersion of positive gamma PDF (default: 1)
    ndsp : 
        Dispersion of negative gamma PDF (default: 1)
    t : vector | None
        Sampling range (default: [0, onset + 2 * (nttp + 1)])
    
    Returns
    -------
    h : HRF function given within [0 .. onset + 2*nttp]
    t : HRF sample points
    
    Notes
    -----
    The pttp and nttp parameters are increased by 1 before given
    as parameters into the scipy.stats.gamma.pdf function (which is a property
    of the gamma PDF!)

    Based on hrf function in matlab toolbox `BVQXtools`; converted to python and simplified by ML 
    Version:  v0.7f
    Build:    8110521
    Date:     Nov-05 2008, 9:00 PM CET
    Author:   Jochen Weber, SCAN Unit, Columbia University, NYC, NY, USA
    URL/Info: http://wiki.brainvoyager.com/BVQXtools
    """

    # Input checks
    if not shape.lower() in ('twogamma', 'boynton'):
        warnings.warn('Shape can only be "twogamma" or "boynton"')
        shape = 'twogamma'
    if t is None:
        t = np.arange(0, (onset + 2 * (nttp + 1)), tr) - onset
    else:
        t = np.arange(np.min(t), np.max(t), tr) - onset;

    # Create filter
    h = np.zeros((len(t), ))
    if shape.lower()=='boynton':
        # boynton (single-gamma) HRF
        h = scipy.stats.gamma.pdf(t, pttp + 1, pdsp)
    elif shape.lower()=='twogamma':
        gpos = gamma.pdf(t, pttp + 1, pdsp)
        gneg = gamma.pdf(t, nttp + 1, ndsp) / pos_neg_ratio
        h =  gpos-gneg 
    h /= np.sum(h)
    return t, h



def smooth_run_not_masked(data,smooth_factor):
    """Spatial smoothing function, not essential at this point
       Smoothes data without masking
    """
    smoothed_data = np.zeros_like(data)
    for i,d in enumerate(data):
        smoothed_data[i] = gaussian_filter(data[i], sigma=smooth_factor, order=0, output=None, 
                 mode='reflect', cval=0.0, truncate=4.0)
    return smoothed_data



def load_and_process(file,start_trim = 20, end_trim = 15, do_detrend=False, smoothing_factor = 0,
                     do_zscore = True): 
    """ Returns the data matrix corresponding to the filename provided
            Arguments:
                        file: name of the run file to load
                  start_trim: how many TRs to remove from the beginning of each run
                              the same number should be removed from the feature matrices
                    end_trim: how many TRs to remove from the end of each run
                              the same number should be removed from the feature matrices
                  do_detrend: temporal detrending of each voxel
            smoothing_factor: how much spatial smoothing to do. 0 = no smoothing
                              smoothing has a required zscore step
                   do_zscore: zscore (0-mean 1-std) the run
    
    """
    print("Loading {} ...".format(file))
    dat = nibabel.load(file).get_data()
    # very important to transpose otherwise data and brain surface don't match
    dat = dat.T 
    print("Loaded array with shape {}".format(dat.shape))
    
    #trimming
    if end_trim>0:
        dat = dat[start_trim:-end_trim]
    else: # to avoid empty error when end_trim = 0
        dat = dat[start_trim:]
    
    # detrending
    if do_detrend:
        dat = signal.detrend(np.nan_to_num(dat),axis =0)
    
    # smoothing
    if smoothing_factor>0:
        # need to zscore before smoothing
        dat = np.nan_to_num(zscore(dat))
        dat = smooth_run_not_masked(dat, smoothing_factor)
        
    # zscore
    if do_zscore:
        dat = np.nan_to_num(zscore(dat))
        
    return dat






def fdr_correction(p_values, alpha=0.05, method="bh", axis=None):
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

def get_contrast_and_Tstat(c,X,Y):
    xTx_pinv = getXTX(X)
    w = getW(X,Y,xTx_pinv)
    mse = getMSE(X,Y,w)
    
    t_stat_num = np.dot(c.T, w)
    t_stat_denom = np.multiply(np.abs(mse),np.sqrt(np.linalg.multi_dot([c.T, xTx_pinv, c])))
    t_stat = np.divide(t_stat_num, t_stat_denom)
    return t_stat_num, t_stat




