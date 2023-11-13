import numpy as np

# add the path to my packages to system paths so they can be imported
import sys
# sys.path.append('/home/yasamanparhizkar/Documents/yorku/01_thesis/simgraph/code/my_packages')
sys.path.append('F:/Users/yasam/Documents/GitHub/simgraph/code/my_packages')
# sys.path.append('/home/yasamanparhizkar/Documents/thesis/code/my_packages')

import dataprocess.data_handler_03 as dh


#################################
# global variables for sampling rates
sr_mnistsift = 4
sr_slowfast = 100
sr_sift3d = 2
sr_soenet = 5
#################################

def load_grouped_spikes(spikes_dp):
    binned_data = np.load(spikes_dp)
    binned_data = binned_data.reshape(binned_data.shape[0], 1141, 113)
    binned_data = binned_data * 2 - 1     # turn labels from 0,1 to -1,1

    I_order_10 = [54, 35, 10, 60, 74, 9, 61, 56, 91, 104]

    # group all neurons together
    grouped_data = np.zeros((297, 1141, 1))
    for trial in range(297):
        for frame in range(1141):
            grouped_data[trial, frame, :] = 2 * int((binned_data[trial, frame, :] == 1).any()) - 1
    
    return grouped_data

def get_mnist_labels(data_params):
    return np.loadtxt(data_params['features_dp']+'lbls.csv')

def transform_mnistsift(fv):
    return fv[::sr_mnistsift]

def transform_slowfast(fv):
    """
    Transform to be applied on feature vectors.
    
    Input: fv
    fv - 1xDf torch tensor representing a feature vector
    
    Output: fvv
    fvv - 1xDf' torch tensor representing the transformed feature vector
    """
    
    # for faster run and less memory usage
    fvv = fv[::sr_slowfast]
    
    # for numerical stability during GD
    # fvv = fvv * 10
    
    return fvv

def transform_sift3d(fv):
    return fv[::sr_sift3d]

def transform_soenet(fv):
    return fv[::sr_soenet]

def get_data_params(feature_id, grouped_data):
    if feature_id == 'mnist-sift':
        data_params = {'func': dh.datapoint_sift, 'lbl_func': get_mnist_labels, 'features_dp': '../../../data/fe_exp/mnist-sift/', \
                       'spike_data': None, 'group_id': None, 'transform': transform_mnistsift, \
                       'ind_min': 0, 'ind_max': 13203,'feature_id':'mnist-sift'}

    elif feature_id == 'slowfast':      
        data_params = {'func': dh.datapoint_numpy, 'lbl_func': dh.get_labels, \
                       'features_dp': '../../data/features/slowfast/slowfast_4732_numpy/', \
                       'spike_data': grouped_data, 'group_id': 0, 'transform': transform_slowfast, \
                       'ind_min': 1*1141+0, 'ind_max': 2*1141-1, 'feature_id':'slowfast'}

    elif feature_id == 'sift3d':
        data_params = {'func': dh.datapoint_numpy, 'lbl_func': dh.get_labels, \
                       'features_dp': '../../data/features/sift3d/fvs_s1_with_kp/desc/', \
                       'spike_data': grouped_data, 'group_id': 0, 'transform': transform_sift3d, \
                       'ind_min': 1*1141+0, 'ind_max': 2*1141-1, 'feature_id':'sift3d'}

    elif feature_id == 'soenet':
        data_params = {'func': dh.datapoint_numpy, 'lbl_func': dh.get_labels, \
                       'features_dp': '../../data/features/soenet/soenet3/features_2layer/', \
                       'spike_data': grouped_data, 'group_id': 0, 'transform': transform_soenet, \
                       'ind_min': 1*1141+41, 'ind_max': 2*1141-1, 'feature_id':'soenet'}
    
    print('**Warning: Check the sampling rate.**\n')
    return data_params