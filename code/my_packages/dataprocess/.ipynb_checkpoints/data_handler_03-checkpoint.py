"""
Added benefits compared to the previous version:
This version puts more parameters in the 'data_params' dictionary and further modularizes the code.
Moreover, you can load data features and labels (must be in {-1, 1}) from disc. 

Author: Yasaman
Last modified: May 12, 2023
"""

import numpy as np
import torch

def datapoint_torch(index, data_params):
    """
    Return a single datapoint consisting of (feature vector, label) 
    based on the extended index system of the whole dataset (297 repeats of a 1141-frame movie); 
    for example, the 6th frame of the 7th repeat is indexed 7*1141+5. 
    In this system, indices only move forward after repeats, so they represent time in a sense.
    Acceptable index range is batch_sz-1 to 1141*297-1.
      
    Inputs: index, data_params
    index - chosen datapoint's index
    data_params   -
        func - funtion which returns a datapoint (fv, lbl) based on its index
        features_dp - path to where feature vectors are stored
        spike_data - (297 x 1141 x m)-shaped array where m is the number of subgroups of neurons.
        group_id - index of the chosen subgroup of neurons which is being considered
        transform - func. applied to the original feature vector (defult: None, no transform is applied)
            
    
    Output: fv, lbl
    fv  - torch tensor representing the selected time bin's feature vector
    lbl - the selected time bin's label
    """
    # unpack params
    features_dp = data_params['features_dp']
    lbl_func = data_params['lbl_func']
    transform = data_params['transform'] if 'transform' in data_params else None
    
    # feature vector
    # trial = index//1141
    frame = index%1141
    fv = torch.load(features_dp+'fv_'+str(frame)+'.pt')
    if transform is not None:
        fv = transform(fv)

    # label  
    lbls = lbl_func(data_params)
    lbl = lbls[index]
    
    return fv, lbl

def datapoint_numpy(index, data_params):
    """
    Return a single datapoint consisting of (feature vector, label) 
    based on the extended index system of the whole dataset (297 repeats of a 1141-frame movie); 
    for example, the 6th frame of the 7th repeat is indexed 7*1141+5. 
    In this system, indices only move forward after repeats, so they represent time in a sense.
    Acceptable index range is batch_sz-1 to 1141*297-1.
      
    Inputs: index, features_dp, spike_data, group_id
    index - chosen datapoint's index
    data_params   -
        func - funtion which returns a datapoint (fv, lbl) based on its index
        features_dp - path to where feature vectors are stored
        spike_data - (297 x 1141 x m)-shaped array where m is the number of subgroups of neurons.
        group_id - index of the chosen subgroup of neurons which is being considered
        transform - func. applied to the original feature vector (defult: None, no transform is applied)
            
    
    Output: fv, lbl
    fv  - Dfx1 vector representing the selected time bin's feature vector
    lbl - the selected time bin's label
    """
    # unpack params
    features_dp = data_params['features_dp']
    lbl_func = data_params['lbl_func']
    transform = data_params['transform'] if 'transform' in data_params else None
    
    # feature vector
    # trial = index//1141
    frame = index%1141
    fv = np.loadtxt(features_dp+'fv_'+str(frame)+'.csv')
    if transform is not None:
        fv = transform(fv)

    # label  
    lbls = lbl_func(data_params)
    lbl = lbls[index]  
    
    return fv, lbl

def datapoint_sift(index, data_params):
    """
    Only for use of mnist-sift features
    """
    # unpack params
    features_dp = data_params['features_dp']
    lbl_func = data_params['lbl_func']
    transform = data_params['transform'] if 'transform' in data_params else None
    
    # feature vector
    fv = np.loadtxt(features_dp+'fv_'+str(index)+'.csv')
    if transform is not None:
        fv = transform(fv)

    # label  
    lbls = lbl_func(data_params)
    lbl = lbls[index]  
    
    return fv, lbl

def update_set(samples, data_params):
    """
    Update a set (either training or validation) based on a new list of datapoints' indices.
    
    Input: samples, data_params
    samples - list of datapoints' indices (original indices representing time)
    data_params   -
        func - funtion which returns a datapoint (fv, lbl) based on its index
        features_dp - path to where feature vectors are stored
        spike_data - (297 x 1141 x m)-shaped array where m is the number of subgroups of neurons.
        group_id - index of the chosen subgroup of neurons which is being considered
        transform - func. applied to the original feature vector (defult: None, no transform is applied)
        
    Output: dess, lbls
    dess - NxD matrix of feature vectors of N datapoints
    lbls - Nx1 vector of corresponding labels of said datapoints
    """
    
    datapoint = data_params['func']
    
    dess = []
    lbls = []
    for index in samples:
        fv, lbl = datapoint(index, data_params)
        dess.append(fv)
        lbls.append(lbl)
    
    if datapoint == datapoint_torch:
        dess = torch.cat(dess)
        dess = dess.detach().numpy()
    elif datapoint == datapoint_numpy:
        dess = np.array(dess)
    else:
        dess = np.array(dess)
    lbls = np.array(lbls)
    
    return dess, lbls

def random_train_val(train_num, val_num, data_params, seed=None):
    """
    Choose random datapoints to form training and validation datasets. The two sets do not overlap.
    Note: since datapoints are selected randomly, their new indices do NOT represent time anymore.
    
    Input: train_num, val_num, ind_min, ind_max, data_params, seed
    train_num     - size of the training datase
    val_num       - size of the validation dataset
    ind_min       - minimum possible datapoint index (acceptable >= batch_sz-1)
    ind_max       - maximum possible datapoint index (acceptable < 297*1141)
    data_params   -
        func - funtion which returns a datapoint (fv, lbl) based on its index
        features_dp - path to where feature vectors are stored
        spike_data - (297 x 1141 x m)-shaped array where m is the number of subgroups of neurons.
        group_id - index of the chosen subgroup of neurons which is being considered
        transform - func. applied to the original feature vector (defult: None, no transform is applied)
    seed - for random selection of datapoints (default: None, machine chosen seed is used)
    
    Output: train_num, val_num, train_data, val_data
    train_num  - number of training datapoints
    val_num    - number of validation datapoints
    train_data - 
        des   - NxD numpy array of feature vectors
        lbls  - Nx1 numpy array of corresponding labels
        smpls - list of indices of chosen datapoints, original indices which represent time
    val_data  - 
        des   - NxD numpy array of feature vectors
        lbls  - Nx1 numpy array of corresponding labels
        smpls - list of indices of chosen datapoints, original indices which represent time
    """
    
    ind_min = data_params['ind_min']
    ind_max = data_params['ind_max']
    data_num = ind_max - ind_min + 1
    train_num = min(train_num, data_num)
    val_num   = min(val_num, data_num-train_num)
    
    # select indices of datapoints randomly
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.arange(ind_min, ind_max+1), size=(train_num+val_num), replace=False)
    train_smpls = samples[:train_num]
    val_smpls   = samples[train_num:]
    
    # get feature vectors and labels corresponding to chosen indices
    train_dess, train_lbls = update_set(train_smpls, data_params)
    val_dess, val_lbls = update_set(val_smpls, data_params)
    
    train_data = {'des': train_dess, 'lbls': train_lbls, 'smpls': train_smpls}
    val_data   = {'des': val_dess, 'lbls': val_lbls, 'smpls': val_smpls}
    
    return train_num, val_num, train_data, val_data

def update_indices(num, ind_min, ind_max, minus_set, seed=None):
    """
    Update the choice of datapoints for a specific set (either training or validation).
    Do not consider indices in the minus_set as options; 
    this prevents overlap between the previous and the new sets, or between the training and validation sets.
    
    Input: num, ind_min, ind_max, minus_set, seed=None
    num - number of datapoints in the final set
    ind_min - minimum possible datapoint index (acceptable >= batch_sz-1)
    ind_max - maximum possible datapoint index (acceptable < 297*1141)
    minus_set - set of discarded indices (AKA overlapping indices)
    seed - for random selection of datapoints (default: None, machine chosen seed is used)
    
    Output: num, samples
    num - number of chosen datapoints
    samples - list of chosen datapoints' indices
    """
    
    # remove overlapping indices from options
    options = np.arange(ind_min, ind_max+1)
    keeplist = [True] * len(options)
    for index in minus_set:
        keeplist = np.logical_and(keeplist, (options != index))
    options = options[keeplist]
    
    # user error: more datapoints are requested that available options
    num = min(num, len(options))
    
    # select randomly
    rng = np.random.default_rng(seed)
    samples = rng.choice(options, size=num, replace=False)
    
    return num, samples

def get_labels(data_params):
    spike_data = data_params['spike_data']
    group_id = data_params['group_id']
    lbls = spike_data[:,:,group_id].reshape(-1)

    return lbls

def random_train_val_balanced(train_num, val_num, data_params, seed=None):
    """
    Choose random datapoints to form training and validation datasets. The two sets do not overlap.
    Note: since datapoints are selected randomly, their new indices do NOT represent time anymore.
    
    Input: train_num, val_num, ind_min, ind_max, data_params, seed
    train_num     - size of the training datase
    val_num       - size of the validation dataset
    ind_min       - minimum possible datapoint index (acceptable >= batch_sz-1)
    ind_max       - maximum possible datapoint index (acceptable < 297*1141)
    data_params   -
        func - funtion which returns a datapoint (fv, lbl) based on its index
        features_dp - path to where feature vectors are stored
        spike_data - (297 x 1141 x m)-shaped array where m is the number of subgroups of neurons.
        group_id - index of the chosen subgroup of neurons which is being considered
        transform - func. applied to the original feature vector (defult: None, no transform is applied)
    seed - for random selection of datapoints (default: None, machine chosen seed is used)
    
    Output: train_num, val_num, train_data, val_data
    train_num  - number of training datapoints
    val_num    - number of validation datapoints
    train_data - 
        des   - NxD numpy array of feature vectors
        lbls  - Nx1 numpy array of corresponding labels
        smpls - list of indices of chosen datapoints, original indices which represent time
    val_data  - 
        des   - NxD numpy array of feature vectors
        lbls  - Nx1 numpy array of corresponding labels
        smpls - list of indices of chosen datapoints, original indices which represent time
    """

    # unpack params
    ind_min = data_params['ind_min']
    ind_max = data_params['ind_max']
    
    # select indices of datapoints randomly
    lbl_func = data_params['lbl_func']
    lbls = lbl_func(data_params)
    train_num, train_smpls = update_indices_balanced(train_num, ind_min, ind_max, [], lbls, seed)
    val_num, val_smpls = update_indices_balanced(val_num, ind_min, ind_max, train_smpls, lbls, seed)
    
    # get feature vectors and labels corresponding to chosen indices
    train_dess, train_lbls = update_set(train_smpls, data_params)
    val_dess, val_lbls = update_set(val_smpls, data_params)
    
    train_data = {'des': train_dess, 'lbls': train_lbls, 'smpls': train_smpls}
    val_data   = {'des': val_dess, 'lbls': val_lbls, 'smpls': val_smpls}
    
    return train_num, val_num, train_data, val_data


def update_indices_balanced(num, ind_min, ind_max, minus_set, lbls, seed=None):
    """
    Update the choice of datapoints for a specific set (either training or validation).
    Do not consider indices in the minus_set as options; 
    this prevents overlap between the previous and the new sets, or between the training and validation sets.
    
    Input: num, ind_min, ind_max, minus_set, seed=None
    num - number of datapoints in the final set
    ind_min - minimum possible datapoint index (acceptable >= batch_sz-1)
    ind_max - maximum possible datapoint index (acceptable < 297*1141)
    minus_set - set of discarded indices (AKA overlapping indices)
    lbls - Nx1 vector of datapoint labels, indices must correspond with ind_min and ind_max (lables in {-1, 1})
    seed - for random selection of datapoints (default: None, machine chosen seed is used)
    
    Output: num, samples
    num - number of chosen datapoints
    samples - list of chosen datapoints' indices
    """
        
    # remove overlapping indices from options
    options = np.arange(ind_min, ind_max+1)
    keeplist = [True] * len(options)
    for index in minus_set:
        keeplist = np.logical_and(keeplist, (options != index))
    options = options[keeplist]

    # calculate size of each cluster (-1 and 1) in the dataset
    lbls = lbls[options]
    num_spk = sum(lbls == 1)
    num_nospk = len(lbls)- num_spk  
    dnum_spk = min(num//2, num_spk, num_nospk+1)
    dnum_nospk = min(num-num//2, num_nospk, num_spk+1)
    num = dnum_spk + dnum_nospk # final set size
    
    # select randomly
    rng = np.random.default_rng(seed)
    samples = rng.choice(options[lbls == 1], size=dnum_spk, replace=False)
    samples = np.append(samples, rng.choice(options[lbls != 1], size=dnum_nospk, replace=False))
    
    return num, samples

def normalize(dess, feature_nrm=1, node_nrm=1):
    """
    Normalize feature vectors.
    Inputs: dess, feature_nrm, node_nrm
    dess - NxD array of features for all datapoints.
    feature_nrm - final norm by feature (columns of dess)
    node_nrm - final norm by datapoint/node (rows of dess)
    
    Outputs: dess_nrm
    dess_nrm - NxD array of normalized features.
    """
    # method 2: double normalization
    # step 1 - feature-wise: subtract mean and divide by standard deviation of each feature.
    dess_mean = np.mean(dess, axis=1, keepdims=True)
    dess_std = np.std(dess, axis=1, keepdims=True)

    dess_nrm = dess - dess_mean
    dess_nrm = dess_nrm * feature_nrm / (dess_std + 0.01)


    # step 2 - smaple-wise: normalize l2-norm of each vector to a certain value.
    ideal_norm = 30
    dess_norm = np.linalg.norm(dess_nrm, axis=0, keepdims=True)
    dess_nrm = dess_nrm * node_nrm / (dess_norm + 0.01)
    
    return dess_nrm

###########################
# Visualization Utilities #
###########################

def class_percentages(lbls, classes):
    """
    Receive class labels. Calculate what percentage of data belongs to each class.

    Input: lbls, classes
    lbls - Nx1 vector of class labels for all datapoints
    classes - list of all possible labels

    Output: pers
    pers - list of percentages corresponding to labels in 'classes'
    """

    N = len(lbls)
    pers = classes.copy()

    for i in range(len(classes)):
        pers[i] = np.sum(lbls == classes[i])*100/N

    return pers


