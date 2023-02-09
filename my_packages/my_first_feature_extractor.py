"""
With this package you can: 
1. download an OpenML dataset
2. make it a binary classification set by separating out only 0, 1 classe
3. split it into training and validation datasets 
4. extract SIFT keypoints and descriptors from images
5. discard of images with fewer keypoints than a certain min, and discard extra keypoints in images with more keypoints than a certain max
6. concatenate SIFT descriptors to form singular feature vectors for images
7. normalize feature vectors
8. save and load feature vectors and image labels
9. measure running times of all funtions

Autorun and some visualization utilities are also provided. Autorun function is named 'xsift_from_data'.
Used as a block in Yasaman's master's thesis.

Author: Yasaman
Last modified: Nov 15, 2022
"""

import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml # to download OpenML data

train_times = {'download': None, 'binarize': None, 'split_data': [None, None], 'convert_2D_U8': None, 'sift': None, \
    'detangle_kps': None, 'kpnum_floor': None, 'kpnum_ceil': None, 'normalize_dess': None, 'save_data': None, \
    'load_data': None, 'make_train_val_data': None, 'xsift_from_imgs': None, 'combine_sift_dess': None, 'xsift_from_data': None, \
    'class_percentages': None, 'display_image': None, 'vis_random_imgs': None, 'kp_pers': None, 'visualize_kps': None}

val_times = {'download': None, 'binarize': None, 'split_data': [None, None], 'convert_2D_U8': None, 'sift': None, \
    'detangle_kps': None, 'kpnum_floor': None, 'kpnum_ceil': None, 'normalize_dess': None, 'save_data': None, \
    'load_data': None, 'make_train_val_data': None, 'xsift_from_imgs': None, 'combine_sift_dess': None, 'xsift_from_data': None, \
    'class_percentages': None, 'display_image': None, 'vis_random_imgs': None, 'kp_pers': None, 'visualize_kps': None}



def download(data_set = 'mnist', version='active', data_id=None, times = train_times):
    """
    Download given data_set

    Input: data_set='mnist', version='active', data_id=None
    data_set - name of the dataset in OpenML repository
    version - version of the dataset
    data_id - most exact way of distinguishing a dataset in OpenML repository

    Output: raw_data_x, raw_data_y
    raw_data_x - NaxD matrix of numeric data. Each row is a datapoint.
    raw_data_y - Nax1 vector of corresponding class labels.
    """

    start_t = time.time()
    
    if data_id is not None:
        raw_data = fetch_openml(data_id = data_id)
        
    elif data_set == 'mnist':
        raw_data = fetch_openml('mnist_784')
        
    else:
        try:
            raw_data = fetch_openml(name = data_set, version = version)
        except:
            print('Dataset does not exist in OpenML repository.')

    # separate numeric data from classification target
    raw_data_x = np.array((raw_data['data']+0.5)/256.0)
    raw_data_y = np.array(raw_data['target'].astype('int8'))
    data_dim = raw_data_x.shape[1]

    times['download'] = time.time() - start_t
    
    return raw_data_x, raw_data_y, data_dim


def binarize(raw_data_x, raw_data_y, times = train_times):
    """
    From any classification dataset, pick out 0 and 1 labeled datapoints and create a new binary classification dataset with them.

    Input: raw_data_x, raw_data_y
    raw_data_x - NaxD matrix of numeric data. Each row is a datapoint.
    raw_data_y - Nax1 vector of corresponding class labels.

    Output: data_x, data_y, num_classes
    data_x - NbxD matrix of numeric data. Each row is a datapoint.
    data_y - Nbx1 vector of corresponding class labels; possible classes are 0 and 1.
    num_classes - 2
    """

    start_t = time.time()

    accptbl_pts = (raw_data_y == 0) | (raw_data_y == 1)
    data_x = raw_data_x[accptbl_pts]
    data_y = raw_data_y[accptbl_pts]
    num_classes = 2

    times['binarize'] = time.time() - start_t
    
    return data_x, data_y, num_classes
    

def split_data(x, y, fracs=[0.8,0.2], seed=0, time_i = 0, times = train_times):
    """
    Randomly split data into two sets.

    Input: x, y, fracs=[0.8, 0.2], seed=0
    x - NxD matrix of x data (e.g. images)
    y - Nx1 vector of y data (e.g. labels)
    fracs - split fractions determining sizes of set one and set two.
    seed - random seed. 'None' disables the use of a new seed.

    Output: x1, y1, x2, y2
    x1 - (fracs[0]*N)xD matrix of x data of set 1
    y1 - (fracs[0]*N)x1 vector of y data of set 1
    x2 - (fracs[1]*N)xD matrix of x data of set 2
    y2 - (fracs[1]*N)x1 vector of y data of set 2
    """

    start_t = time.time()

    if seed is not None:
        np.random.seed(seed)
    N = x.shape[0]
    rp = np.random.permutation(N)

    N1 = int(fracs[0]*N)
    N2 = min(N-N1,int(fracs[1]*N))

    # split the data into two parts
    x1 = x[rp[:N1]]
    y1 = y[rp[:N1]]
    x2 = x[rp[N1:(N1+N2)]]
    y2 = y[rp[N1:(N1+N2)]]

    times['split_data'][time_i] = time.time() - start_t

    return x1,y1,x2,y2


def convert_2D_U8(data_x, img_sz, times = train_times): 
    """
    Grayscale images are sometimes represented as 1D vectors of [0,1] fractional entries.
    This function converts the representation to 2D matrices of [0,255] integer entries.

    Input: data_x, img_sz
    data_x - NxD matrix of all images in the dataset. Each row is one image.
    img_sz - (length, width) of the images in pixels (e.g. (28,28))
    times - the dict to store running time. if None, does not store running time.

    Output: cnv_data_x
    cnv_data_x - (N x length x width) tensor of all images in the dataset with the new representation.
    """

    start_t = time.time()

    N = data_x.shape[0]
    length = img_sz[0]
    width = img_sz[1]

    cnv_data_x = data_x.reshape((N, length, width))
    cnv_data_x = (cnv_data_x*256).astype('uint8')

    if times is not None:
        times['convert_2D_U8'] = time.time() - start_t

    return cnv_data_x


def sift(images, sift_params, times = train_times):
    """
    Extracts SIFT keypoints and descriptors for all input images.

    Input: images, sift_params
    images - (N x length x width) tensor of N grayscale images with size (length, width) and integer entries between [0,255]
    sift_params - parameters of the SIFT algorithm

    Output: kp, des
    kp - list of tuples of keypoints of images. Each entry of the list is a tuple containing keypoints of the corresponding image.
    des - list of nx128 matrices depicting descriptors found in the corresponding image.
    """

    start_t = time.time()

    # define SIFT parameters. If no parameters are given, resort to default values.
    nfeatures = sift_params['nfeatures'] if 'nfeatures' in sift_params else 0
    nOctaveLayers = sift_params['nOctaveLayers'] if 'nOctaveLayers' in sift_params else 3
    contrastThreshold = sift_params['contrastThreshold'] if 'contrastThreshold' in sift_params else 0.04
    edgeThreshold = sift_params['edgeThreshold'] if 'edgeThreshold' in sift_params else 10
    sigma = sift_params['sigma'] if 'sigma' in sift_params else 1.6
        
    # create SIFT object with given parameters
    sift = cv.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
    
    # extract SIFT keypoints and descriptors for all images
    N = images.shape[0]
    kp = []
    des = []
    for i in range(N):
        _kp, _des = sift.detectAndCompute(images[i],None)
        kp.append(_kp)
        des.append(_des)

    times['sift'] = time.time() - start_t

    return kp, des


def detangle_kps(kp, des, times = train_times):
    """
    Receive SIFT keypoints and descriptors of a number of images. If multiple keyponits in an image are too close to each other, only keep one of them.

    Input: kp, des
    kp - list of tuples of keypoints of images.
    des - list of nx128 matrices of descriptors of corresponding images.

    Output: cl_kp, cl_des
    cl_kp - clean list of tuples of keypoints of images where each keypoint only appears once in a tuple.
    cl_des - list of nx128 matrices of descriptors of corresponding images.
    """

    start_t = time.time()
    
    print("To Be Implemented.")

    times['detangle_kps'] = time.time() - start_t

    return 0

def kpnum_floor(kp, des, cutoff, times = train_times):
    """
    Receive SIFT keypoints and descriptors of a number of images. Discard all images with less than a certain number of keypoints.

    Input: kp, des, cutoff
    kp - list of tuples of keypoints of images.
    des - list of nx128 matrices of descriptors of corresponding images.
    cutoff - minimum acceptable number of keypoints in each image.

    Output: ct_kp, ct_des, ct_ind
    ct_kp - list of tuples of keypoints of images with more than the minimum number of keypoints.
    ct_des - list of nx128 matrices of descriptors of corresponding images.
    ct_ind - list of indices of images with more than the minimum number of keypoints in the original dataset.
    """

    start_t = time.time()

    N = len(kp)
    ct_ind = [i for i in range(N) if len(kp[i])>=cutoff]
    ct_kp = [kp[i] for i in ct_ind]
    ct_des = [des[i] for i in ct_ind]

    times['kpnum_floor'] = time.time() - start_t

    return ct_kp, ct_des, ct_ind


def kpnum_ceil(kp, des, cutoff, times = train_times):
    """
    Receive SIFT keypoints and descriptors of a number of images.
    For images with more than 'cutoff' number of keypoints, keep only 'cutoff' keypoints based on their proximity to the top-left corner.
    For example, if cutoff is 2, keep only 1 keypoint closest to and 1 keypoint farthest from the top-left corner.
    This strategy hopefully leads to the widest possible coverage of the image by the keypoints' descriptors.
    -----
    IMPORTANT Note: If you are using this function alongside kpnum_floor, you should use kpnum_floor before this function.

    Input: kp, des, cutoff
    kp - list of tuples of keypoints of images.
    des - list of nx128 matrices of descriptors of corresponding images.
    cutoff - maximum acceptable number of keypoints in each image.

    Output: ct_kp, ct_des
    ct_kp - list of tuples of keypoints of images, now having fewer equal keypoints to the given maximum.
    ct_des - list of (n*128)x1 vectors of concatenated descriptors of all keypoints of corresponding images.
    """

    start_t = time.time()

    assert cutoff > 0

    ct_kp = []
    ct_des = []
    N = len(kp)

    if cutoff == 2:
        for i in range(N):
            # calculate keypoints' distances from the top-left corner
            dists = [np.sum(np.array(key.pt)**2) for key in kp[i]]

            # choose keypoints closest and farthest from the top-left corner
            left_pt = np.argmin(dists)
            right_pt = np.argmax(dists)

            # create new lists with chosen keypoints and the concatenation of their descriptors
            kept_kps = (kp[i][left_pt], kp[i][right_pt])
            cat_des = np.append(des[i][left_pt], des[i][right_pt], axis=0)
            ct_kp.append(kept_kps)
            ct_des.append(cat_des)

    else:
        nr = int(cutoff/2) # number of keypoints farthest from the top-left corner
        nl = cutoff - nr   # number of keypoints closest to the top-left corner

        for i in range(N):
            # calculate keypoints' distances from the top-left corner
            n = len(kp[i])
            dists = [(j, np.sum(np.array(kp[i][j].pt)**2)) for j in range(n)]

            # sort keypoints by their distances
            sorted_dists = sorted(dists, key=lambda x:x[1])

            # keep the right number of keypoints closes to and farthest from the top-left corner
            raw_ind = [j if j < nl else -1*(j-nl+1) for j in range(cutoff)]
            kp_ind = [sorted_dists[ind][0] for ind in raw_ind]

            kept_kps = tuple([kp[i][ind] for ind in kp_ind])
            cat_des = des[i][kp_ind,:].reshape(-1)

            ct_kp.append(kept_kps)
            ct_des.append(cat_des)

    times['kpnum_ceil'] = time.time() - start_t

    return ct_kp, ct_des
    

def normalize_dess(descriptors, times = train_times):
    """
    Receive feature vectors describing a number of images (e.g. concatenated SIFT descriptors). Normalize the vectors according to below algorithm.
    There are two steps in this normalization: feature-wise and sample-wise (aka. image-wise).
        1. feature-wise: subtract mean and divide by standard deviation of each feature for all images.
        2. sample-wise: normalize l2-norm of each vector to 1.

    Input: descriptors
    descriptors - NxDf matrix of vectors describing N images.

    Output: nrm_dess
    nrm_dess - NxDf matrix of normalized vectors describing N images.
    """

    start_t = time.time()

    # feature-wise normalization
    means = np.mean(descriptors, axis=0, keepdims=True)
    stds = np.std(descriptors, axis=0, keepdims=True)
    nrm_dess = descriptors - means
    nrm_dess = nrm_dess / (stds + 0.01)

    # sample-wise normalization
    norms = np.linalg.norm(nrm_dess, axis=1, keepdims=True)
    nrm_dess = nrm_dess / (norms + 0.01)

    times['normalize_dess'] = time.time() - start_t

    return nrm_dess


def save_data(imgs, lbls, path = '/', times = train_times):
    """
    Save feature vectors describing images and their corresponding class labels to a specified path. 
    Note: both images and their labels should be stored in numpy arrays.

    Input: imgs, lbls, path
    imgs - NxDf matrix of feature vectors describing images. each row is describing one image.
    lbls - Nx1 vector of corresponding class labels.
    path - path to which to save the data (default: '/')

    Output:
    """

    start_t = time.time()

    assert type(imgs) == np.ndarray
    assert type(lbls) == np.ndarray

    np.savetxt(path+'imgs.txt', imgs)
    np.savetxt(path+'lbls.txt', lbls)

    times['save_data'] = time.time() - start_t


def load_data(path, times = train_times):
    """
    Load a numpy array (e.g. the feature vectors describing images or their corresponding class labels) from a specified path. 

    Input: path
    path - path from which to load the data

    Output: data
    data - loaded numpy array
    """

    start_t = time.time()

    data = np.loadtxt(path)

    times['load_data'] = time.time() - start_t

    return data


def make_train_val_data(data_params, allocation_params, img_sz):
    """
    Download required OpenML dataset and organize it in training and validation datasets as requested. Autorun of some functions in this package.

    Input: data_params, allocation_params, img_sz
    -----
    data_params - dict of info about the dataset to be downloaded. Only OpenML datasets are acceptable:
        'name': name of the dataset (default: 'mnist')
        'version': which version to download (default: 'active')
        'id': best way to distinguish an OpenML dataset. can be used instead of 'name' and 'version'. otherwise, it is 'None'.
    allocation_params - dict of parameters determining sizes of training and validation datasets:
        'portion' - what portion of the dataset to be used for both datasets together (default: 0.01).
        'fracs' - fractions of training and validation sizes applied to the portion chosen above (default: [0.8, 0.2]).
        'seed' - seed for random choice of training and validation data points (default: None, in which case no new seed is used).
    img_sz - tuple of (length, width) of images (default: None, in which case img_sz=(data_dim/2, data_dim/2)).


    Output: train_x, train_y, val_x, val_y, img_sz
    -----
    train_x - NtxD matrix of training images. each row is an image.
    train_y - Ntx1 vector of class labels of corresponding training images.
    val_x - NvxD matrix of validation images. each row is an image.
    val_y - Nvx1 vector of class labels of corresponding validation images.
    img_sz - tuple of (length, width) of images.
    """

    start_t = time.time()

    # parameters determining the dataset to be downloaded
    data_name = data_params['name'] if 'name' in data_params else 'mnist'
    data_version = data_params['version'] if 'version' in data_params else 'active'
    data_id = data_params['id'] if 'id' in data_params else None
    # parameters determining sizes of training and validation sets
    portion = allocation_params['portion'] if 'portion' in allocation_params else 0.01
    fracs = allocation_params['fracs'] if 'fracs' in allocation_params else [0.8, 0.2]
    seed = allocation_params['seed'] if 'seed' in allocation_params else None

    # download the data
    raw_data_x, raw_data_y, data_dim = download(data_name, data_version, data_id, times = train_times)
    val_times['download'] = train_times['download']

    # determine image size
    if img_sz is None:
        img_sz = (data_dim//2, data_dim - data_dim//2)

    # separate only 0, 1 classes to make a binary classification set
    data_x, data_y, num_classes = binarize(raw_data_x, raw_data_y, times = train_times)
    val_times['binarize'] = train_times['binarize']

    # make training and validation sets
    train_x, train_y, test_x, test_y = split_data(data_x, data_y, fracs=[portion,portion], seed=seed, time_i=0, times = train_times)
    train_x, train_y,  val_x, val_y = split_data(train_x, train_y, fracs=fracs, seed=seed, time_i=1, times = train_times)
    val_times['split_data'] = train_times['split_data']

    train_times['make_train_val_data'] = time.time() - start_t
    val_times['make_train_val_data'] = train_times['make_train_val_data']

    return train_x, train_y, val_x, val_y, img_sz


def xsfit_from_imgs(images, img_sz, sift_params, times = train_times):
    """
    Receive a number of images and extract SIFT keypoints and descriptors from them. Autorun of some functions in this package.

    Input: images, img_sz, sift_params
    images - NxD matrix of N images represetend as rows.
    img_sz - tuple of (length, width) of images
    sift_params - dict of parameters for the SIFT algorithm:
        'nfeatures': (default: 0)
        'nOctaveLayers': (default: 3)
        'contrastThreshold': (default: 0.04)
        'edgeThreshold': (default: 10)
        'sigma': (default: 1.6)

    Output: kp, des
    kp - list of tuples of keypoints of images
    des - list of nx128 matricies of descriptors of corresponding keypoints.
    """

    start_t = time.time()

    # represent images in the required way by the SIFT algorithm
    cnv_images = convert_2D_U8(images, img_sz, times = times)

    # extract SIFT keypoints and descriptors
    kp, des = sift(cnv_images, sift_params, times = times)

    times['xsift_from_imgs'] = time.time() - start_t
    
    return kp, des


def combine_sift_dess(kp, des, kpnum_params, normalize, times = train_times):
    """
    Receive keypoints and descriptors of a number of images. Keep the required number of keypoints in images. 
    Concatenate the corresponding descriptors to form a single feature vector for each image. Autorun of some functions in this package.

    Input: kp, des, kpnum_params
    kp - list of tuples of keypoints of images
    des - list of nx128 matrices of decriptors of corresponding keypoints
    kpnum_params - dict of parameters determining number of keypoints in images.
        'min': minimum number of keypoints (default: 2)
        'max': maximum number of keypoints (default: 2)
    normalize - whether or not to normalize the feature vectors

    Output: kps, dess, ind
    kps - list of tuples of keypoints of images, now having the required number of keypoints for all images
    dess - NxDf matrix of feature vectors describing images. each row is describing one image.
    ind - list of indices of the kept images (in the original set) which had more equal number of keypoints to the given min.
    """

    start_t = time.time()

    # parameters determining number of keypoints in images
    kpnum_min = kpnum_params['min'] if 'min' in kpnum_params else 2
    kpnum_max = kpnum_params['max'] if 'max' in kpnum_params else 2

    # discard images with fewer keypoints than the requested min
    kps, dess, ind = kpnum_floor(kp, des, kpnum_min, times = times)

    # choose the correct number of keypoints in images according to the given max
    kps, dess = kpnum_ceil(kps, dess, kpnum_max, times = times)

    # normalize feature vectors
    if normalize:
        dess = normalize_dess(np.array(dess), times = times)
    else:
        dess = np.array(dess)

    times['combine_sift_dess'] = time.time() - start_t

    return kps, dess, ind



def xsift_from_data(data_params, allocation_params, sift_params, kpnum_params, \
    normalize = True, return_imgs = False, save = False, savepath = '/', img_sz = None):
    """
    Autorun of all functions in this package.

    Input: data_params, allocation_params, img_sz, sift_params, kpnum_params, normalize, return_imgs, save, savepath
    -----   
    data_params - dict of info about the dataset to be downloaded. Only OpenML datasets are acceptable:
        'name': name of the dataset (default: 'mnist')
        'version': which version to download (default: 'active')
        'id': best way to distinguish an OpenML dataset. can be used instead of 'name' and 'version'. otherwise, it is 'None'.
    allocation_params - dict of parameters determining sizes of training and validation datasets:
        'portion' - what portion of the dataset to be used for both datasets together (default: 0.01).
        'fracs' - fractions of training and validation sizes applied to the portion chosen above (default: [0.8, 0.2]).
        'seed' - seed for random choice of training and validation data points (default: None, in which case no new seed is used).
    sift_params - dict of parameters for the SIFT algorithm:
        'nfeatures': (default: 0)
        'nOctaveLayers': (default: 3)
        'contrastThreshold': (default: 0.04)
        'edgeThreshold': (default: 10)
        'sigma': (default: 1.6)
    kpnum_params - dict of parameters determining number of keypoints in images.
        'min': minimum number of keypoints (default: 2)
        'max': maximum number of keypoints (default: 2)
    normalize - whether or not to normalize the feature vectors (default: True)
    return_imgs - wheter or not to return the original images from which features were extracted (default: False)
    save - wheter or not to save the descriptors and class labels as files (default: False)
    savepath - path to save the data (default: '/', in which case it saves in the active directory)
    img_sz - tuple of (length, width) of images (default: None, in which case img_sz=(data_dim/2, data_dim/2)).


    Output: num_train, num_val, train_data, val_data
    -----
    num_train - number of training data points
    num_val - number of validation data points
    if return_imgs:
    train_data - dict of info about training data:
        'des' - NtxDf matrix of descriptors of training images. descriptors correspond to keypoints in 'kp'
        'lbls' - Ntx1 vector of class labels of training images.
        'imgs' - NtxD matrix of original training data before feature extraction
        'kp' - list of tuples of keypoints of training images.
    val_data - dict of info about validation data
        'des' - NvxDf matrix of descriptors of validation images. descriptors correspond to keypoints in 'kp'
        'lbls' - Nvx1 vector of class labels of validation images.
        'imgs' - NtxD matrix of original training data before feature extraction
        'kp' - list of tuples of keypoints of validation images.
    else:
    train_data - dict of info about training data:
        'des' - NtxDf matrix of descriptors of training images. descriptors correspond to keypoints in 'kp'
        'lbls' - Ntx1 vector of class labels of training images.
    val_data - dict of info about validation data
        'des' - NvxDf matrix of descriptors of validation images. descriptors correspond to keypoints in 'kp'
        'lbls' - Nvx1 vector of class labels of validation images.
    """ 

    start_t = time.time()

    # prepare training and validation datasets as requested
    train_x, train_y, val_x, val_y, img_sz = make_train_val_data(data_params, allocation_params, img_sz)
    
    # get number of training and validation data points
    num_train = len(train_y)
    num_val = len(val_y)

    # extract SIFT keypoints and descriptors 
    train_kp, train_des = xsfit_from_imgs(train_x, img_sz, sift_params, times = train_times)
    val_kp, val_des = xsfit_from_imgs(val_x, img_sz, sift_params, times = val_times)

    # make a single feature vector for each image by combining a certain number of SIFT descriptors
    train_kp, train_des, train_ind = combine_sift_dess(train_kp, train_des, kpnum_params, normalize, times = train_times)
    val_kp, val_des, val_ind = combine_sift_dess(val_kp, val_des, kpnum_params, normalize, times = val_times)

    # discard images and their labels which have fewer keypoints than the given min, as done in the previous step
    train_x = train_x[train_ind]
    train_y = train_y[train_ind]
    val_x = val_x[val_ind]
    val_y = val_y[val_ind]
    num_train = len(train_y)
    num_val = len(val_y)

    # save the data
    if save:   
        save_data(train_des, train_y, savepath+'train_', times = train_times)
        save_data(val_des, val_y, savepath+'val_', times = val_times)   

    # create dictionaries to be returned
    if return_imgs:
        train_data = {'des': train_des, 'lbls': train_y, 'imgs': train_x, 'kp': train_kp}
        val_data = {'des': val_des, 'lbls': val_y, 'imgs': val_x, 'kp': val_kp}
    else:
        train_data = {'des': train_des, 'lbls': train_y}
        val_data = {'des': val_des, 'lbls': val_y}

    train_times['xsift_from_data'] = time.time() - start_t
    val_times['xsift_from_data'] = train_times['xsift_from_data']

    return num_train, num_val, train_data, val_data


def xsift_no_download(data_x, data_y, portion, fracs, seed, img_sz, sift_params, kpnum_params, normalize):
    """
    Extract SIFT descriptors from a set of already downloaded images.
    Autorun of some functions in this package.

    Input: data_x, data_y, portion, fracs, seed, img_sz, sift_params, kpnum_params, normalize

    Output: num_train, num_val, train_data, val_data
    """

    # split data into train and val data sets
    train_x, train_y, test_x, test_y = split_data(data_x, data_y, fracs=[portion,portion], seed=seed, time_i=0, times = train_times)
    train_x, train_y,  val_x, val_y = split_data(train_x, train_y, fracs=fracs, seed=seed, time_i=1, times = train_times)
    val_times['split_data'] = train_times['split_data']
    
    # extract SIFT keypoints and descriptors 
    train_kp, train_des = xsfit_from_imgs(train_x, img_sz, sift_params, times = train_times)
    val_kp, val_des = xsfit_from_imgs(val_x, img_sz, sift_params, times = val_times)

    # make a single feature vector for each image by combining a certain number of SIFT descriptors
    train_kp, train_des, train_ind = combine_sift_dess(train_kp, train_des, kpnum_params, normalize, times = train_times)
    val_kp, val_des, val_ind = combine_sift_dess(val_kp, val_des, kpnum_params, normalize, times = val_times)

    # discard images and their labels which have fewer keypoints than the given min, as done in the previous step
    train_y = train_y[train_ind]
    val_y = val_y[val_ind]
    num_train = len(train_y)
    num_val = len(val_y)

    # create dictionaries to be returned
    train_data = {'des': train_des, 'lbls': train_y}
    val_data = {'des': val_des, 'lbls': val_y}

    return num_train, num_val, train_data, val_data



###########################
# Visualization Utilities #
###########################

def class_percentages(lbls, classes, times = train_times):
    """
    Receive class labels. Calculate what percentage of data belongs to each class.

    Input: lbls, classes
    lbls - Nx1 vector of class labels for all data points
    classes - list of all possible labels

    Output: pers
    pers - list of percentages corresponding to labels in 'classes'
    """

    start_t = time.time()

    N = len(lbls)
    pers = classes.copy()

    for i in range(len(classes)):
        pers[i] = np.sum(lbls == classes[i])*100/N

    times['class_percentages'] = time.time() - start_t

    return pers

def display_image(img, img_sz = None, title = None, times = train_times):
    """
    Display an image represented as a vector, matrix or 3D tensor (with 3 channels for rgb).
    In case of vector representation, image length and width should be given.

    Input: img, img_sz = None, title = None
    img - vector, matrix of 3D tensor depicting an image
    img_sz - tuple of (length, width) of the image. used for vector representation only.
    title - (optional) title of the image.

    Output:
    """

    start_t = time.time()

    if len(img.shape) == 3 or len(img.shape) == 2:
        plt.imshow(img)
    elif len(img.shape) == 1:
        plt.imshow(img.reshape(img_sz))
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)

    times['display_image'] = time.time() - start_t

def vis_random_imgs(imgs, lbls, num_cols = 10, num_rows = 3, seed = None, img_sz = None, times = train_times):
    """
    Visualize random images and their labels.

    Input: imgs, lbls, num_cols, num_rows, seed
    imgs - NxD matrix of images. each row is an image.
    lbls - Nx1 vector of corresponding labels.
    num_cols - number of columns in the figure (default: 10)
    num_rows - number of rows in the figure (default: 3)
    seed - random seed (default: None, in which case no new seed is used)
    img_sz - tuple of (length, width) of the image (default: None, in which case img_sz=(D/2, D/2))

    Output:
    """

    start_t = time.time()

    N, D = imgs.shape

    if img_sz is None:
        img_sz = (D//2, D - D//2)   

    if seed is not None:
        np.random.seed(seed)

    plt.figure(figsize=(num_cols,(num_rows*5)//3))
    for i in range(num_rows*num_cols):
        ind = np.random.randint(0,N)
        plt.subplot(num_rows,num_cols,i+1)
        display_image(imgs[ind,:], img_sz, lbls[ind], times = times)

    times['vis_random_imgs'] = time.time() - start_t


def kp_pers(kp, times = train_times):
    """
    Receive list of SIFT keypoints for a number of images. Find what percentage of images have how many keypoints.

    Input: kp
    kp - list of tuples of keypoints of images

    Output: kpnums, kppers
    kpnums - list of all possible numbers of keypoints found in images
    kppers - list of percentages of images having a certain number of keypoints, corresponding to elements of kpnums
    """

    start_t = time.time()

    N = len(kp)
    kplens = [len(keys) for keys in kp]
    kpnums = list(dict.fromkeys(kplens))
    kppers = kpnums.copy()
    for i in range(len(kpnums)):
        kppers[i] = np.sum(np.array(kplens) == kpnums[i])*100/N

    times['kp_pers'] = time.time() - start_t
    
    return kpnums, kppers

def visualize_kps(imgs, kp, lbls = None, num_cols = 5, num_rows = 3, seed = None, img_sz = None, rich = False, times = train_times):
    """
    Draw SIFT keypoints on images and optionally show their class labels as well.

    Input: imgs, kp, lbls, num_cols, num_rows, seed e, img_sz, rich
    imgs - NxD matrix of images. each row is an image.
    kp - list of tuples of keypoints of images.
    lbls - Nx1 vector of corresponding labels.
    num_cols - number of columns in the figure (default: 5)
    num_rows - number of rows in the figure (default: 3)
    seed - random seed (default: None, in which case no new seed is used)
    img_sz - tuple of (length, width) of images (default: None, in which case img_sz=(D/2, D/2))
    rich - whether or not to draw rich version of keypoints with matches

    Output:
    """

    start_t = time.time()

    N, D = imgs.shape

    if img_sz is None:
            img_sz = (D//2, D - D//2)

    if seed is not None:
        np.random.seed(seed)

    if lbls is None:
        lbls = [None] * N

    # convert images to the required representation by the SIFT object
    cnv_imgs = convert_2D_U8(imgs, img_sz, times = None)

    plt.figure(figsize=(num_cols*2,(num_rows*5)//2))
    for i in range(num_rows*num_cols):
        ind = np.random.randint(0,N)
        plt.subplot(num_rows,num_cols,i+1)

        if rich:
            img=cv.drawKeypoints(cnv_imgs[ind],kp[ind],0,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            img=cv.drawKeypoints(cnv_imgs[ind],kp[ind],0)

        display_image(img, img_sz, lbls[ind], times = times)

    times['visualize_kps'] = time.time() - start_t

def get_times():
    """
    Return dictionaries containing running times of functions in this package.

    Input:

    Output: times
    times - dict of {train_times, val_times} where each element is a dict containing times for each fucntion.
    """

    times = {'train_times': train_times, 'val_times': val_times}

    return times




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    