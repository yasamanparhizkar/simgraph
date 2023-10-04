"""
Classify digits 0 and 1 from the mnist dataset.
Use SIFT (Scale-Invariant Feature Transform) to extract features from images. Each image has two and only two keypoints whose descriptors are concatenated to form the image's feature vector.
Create a similarity graph where each node represents an image, and nodes with the same label are strongly connected to each other while nodes with different labels are weakly connected.
Similarity graph is learned by minimizing x^T L(M) x + \mu tr(M) over M, where M acts as a key yielding the weight of an edge given both ends' feature vectors.
Validation images' labels are estimated by minimizing GLR in a larger graph containing val images as well as training images.
Accuracy on validation set is calculated and returned, alongside the learned M, number of training images and number of validation images.
For more information, refer to 'sift_on_mnist_05.ipynb'

Author: Yasaman
Last modified: August 26 2022
"""

import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml # to download mnist data


def learn_graph(raw_data_x, raw_data_y, portion, fracs, img_sz=(28,28), show_msgs=False):
    """
    Take downloaded mnist data as input.
    Randomly select a specific portion of it and divide that into training and validation sets.
    Use SIFT to extract features of images. 
    Use gradient descent to train a similarity graph minimizing GLR (+ a penalty term). 
    Estimate labels of validation images using the same graph and GLR minimization. 
    Return accuracy.

    Inputs: raw_data_x, raw_data_y, portion, fracs
        raw_data_x - NtxD tensor of raw input images to be classified into two classes
        raw_data_y - Ntx1 vector of corresponding labels for images; described as 0 and 1
        portion - the portion of raw data to be used in the algorithm (whole dataset might be too large)
        fracs - fractions of split between training and validation data (e.g. [0.6, 0.4])

    Outputs: num_train, num_val, acc, M
        num_train - number of images used to learn the similarity graph
        num_validation - number of images used to validate the similarity graph
        acc - obtained accuracy after learning the graph on training data and validating on validation data
        M - DfxDf learned metric matrix of the similarity graph (Df = dimention of feature vectors = 256 in this case)
    """

    # split data into training and validation datasets
    train_x,train_y,test_x,test_y = split_data(raw_data_x,raw_data_y,fracs=[portion,portion], seed=None)
    train_x,train_y,  val_x,val_y = split_data(train_x,train_y,fracs=fracs, seed=None)
    
    num_train = len(train_y)
    num_val = len(val_y)
    
    # convert train and validation images from 1D to 2D with 3 channels of [0,255] pixel values
    train_x_sift = ((train_x.reshape((train_x.shape[0], img_sz[0], img_sz[1], 1)))*256).astype('uint8')
    train_x_sift = np.broadcast_to(train_x_sift, (train_x.shape[0], img_sz[0], img_sz[1], 3))

    val_x_sift = ((val_x.reshape((val_x.shape[0], img_sz[0], img_sz[1], 1)))*256).astype('uint8')
    val_x_sift = np.broadcast_to(val_x_sift, (val_x.shape[0], img_sz[0], img_sz[1], 3))
    
    
    # SIFT default parameter values are:
    nfeatures = 0
    nOctaveLayers = 3
    contrastThreshold = 0.04
    edgeThreshold = 10
    sigma = 1.6
    
    sift = cv.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
    
    # extract SIFT descriptors for all training and validation images
    gray = np.array([cv.cvtColor(train_x_sift[0],cv.COLOR_BGR2GRAY)])
    for i in range(1, train_x_sift.shape[0]):
        gray = np.append(gray, [cv.cvtColor(train_x_sift[i],cv.COLOR_BGR2GRAY)], axis=0)
    for i in range(val_x_sift.shape[0]):
        gray = np.append(gray, [cv.cvtColor(val_x_sift[i],cv.COLOR_BGR2GRAY)], axis=0)
    kp = []
    des = []
    for i in range(gray.shape[0]):
        _kp, _des = sift.detectAndCompute(gray[i],None)
        kp.append(_kp)
        des.append(_des)
    
    # discard all images that have less than 2 keypoints
    num_keypoints = [len(kp[i]) for i in range(len(kp))]
    to_discard_ind = [i for i in range(len(num_keypoints)) if (num_keypoints[i]==0 or num_keypoints[i]==1)]
    x_sift_mod = np.delete(np.append(train_x_sift, val_x_sift, axis=0), to_discard_ind, axis=0)
    y_mod = np.delete(np.append(train_y, val_y, axis=0), to_discard_ind, axis=0)
    gray_mod = np.delete(gray, to_discard_ind, axis=0)
    kp_mod = [kp[i] for i in range(len(kp)) if i not in to_discard_ind]
    des_mod = [des[i] for i in range(len(des)) if i not in to_discard_ind]

    # find new number of train & val images
    num_train_mod = num_train - np.sum(np.array(to_discard_ind) < num_train)
    num_val_mod = num_val - np.sum(np.array(to_discard_ind) >= num_train)
    
    # select only two keypoints in all images with more than 2 keypoints
    for i in range(len(kp_mod)): # for each training or validation image
        # calculate keypoint's distance from the top-left and bottom-right corners
        left_dists = [(kp_mod[i][j], np.sum(np.array(kp_mod[i][j].pt)**2), des_mod[i][j]) for j in range(len(kp_mod[i]))]
        right_dists = [(kp_mod[i][j], np.sum(np.array([img_sz[0]-kp_mod[i][j].pt[0], img_sz[1]-kp_mod[i][j].pt[1]])**2), \
                        des_mod[i][j]) for j in range(len(kp_mod[i]))]

        # select closest keypoints to the top-left and bottom-right corners, discard all other keypoints
        kp_mod[i] = (min(left_dists, key=lambda x:x[1])[0], min(right_dists, key=lambda x:x[1])[0])
        des_mod[i] = np.array([min(left_dists, key=lambda x:x[1])[2], min(right_dists, key=lambda x:x[1])[2]])

        # concatenate two descriptors
        des_mod[i] = des_mod[i].reshape(-1)
        
    
    # Normalize descriptors by method 2: double normalization
    # step 1 - feature-wise: subtract mean and divide by standard deviation of each feature.
    des_mod_nrm = np.array(des_mod)
    des_mod_mean = np.mean(des_mod_nrm, axis=0, keepdims=True)
    des_mod_std = np.std(des_mod_nrm, axis=0, keepdims=True)
    des_mod_nrm = des_mod_nrm - des_mod_mean
    des_mod_nrm = des_mod_nrm / (des_mod_std + 0.01)
    # step 2 - smaple-wise: normalize l2-norm of each vector to 1.
    des_mod_norm = np.linalg.norm(des_mod_nrm, axis=1, keepdims=True)
    des_mod_nrm = des_mod_nrm / (des_mod_norm + 0.01)
    

    # fit similarity graph
    f_sz = des_mod_nrm.shape[1]
    #B0 = 1*np.ones((f_sz, f_sz)) #uncomment if you want to set Theta0: B0 in next line
    opt_params = { 'epsilon0':1, 'epsilon_decay':0.5, 'epsilon_jump': 3,
                   'num_its':11, 'check_freq':1, 'print_checks':show_msgs, 'Theta0':None, 'seed':None }
    B, stats = fit_graph_with_rayleigh_tr(x_sift_mod[:num_train_mod], y_mod[:num_train_mod], \
                                      des_mod_nrm[:num_train_mod], opt_params)
    
    # find learned M from learned B
    M = B.T @ B
    if show_msgs:       
        print('M_max = '+str(np.max(M))+'\tM_min = '+str(np.min(M)))
        hist_of_entries(M, 500, lx=None, lbl='M', zeroline=True, peakline=False)
    
    
    # find unknown labels using learned metric B, and measure accuracy
    f = des_mod_nrm.T
    y_val_est, acc = extrapolate(f, B, y_mod[:num_train_mod], y_mod[num_train_mod:])
    
    
    return num_train_mod, num_val_mod, acc, M


def split_data(x,y,fracs=[0.8,0.2],seed=0):
    """
    Randomly splits data into two sets.

    Inputs: x, y, fracs=[0.8, 0.2], seed=0
    x - NxD matrix of x data (e.g. images)
    y - Nx1 vector of y data (e.g. labels)
    fracs - split fractions determining sizes of set one and set two.
    seed - random seed. 'None' disables the use of a new seed.

    Outputs: x1, y1, x2, y2
    x1 - (fracs[0]*N)xD matrix of x data of set 1
    y1 - (fracs[0]*N)x1 vector of y data of set 1
    x2 - (fracs[1]*N)xD matrix of x data of set 2
    y2 - (fracs[1]*N)x1 vector of y data of set 2
    """

    if seed is not None:
        np.random.seed(seed)
    N = x.shape[0]
    rp = np.random.permutation(N)

    N1 = int(fracs[0]*N)
    N2 = min(N-N1,int(fracs[1]*N))

    # Split the data into two parts
    x1 = x[rp[:N1]]
    y1 = y[rp[:N1]]
    x2 = x[rp[N1:(N1+N2)]]
    y2 = y[rp[N1:(N1+N2)]]

    return x1,y1,x2,y2


def prepare_data(data_set='mnist'):
    """
    Download mnist data. Keep only 0 and 1 images.

    Inputs: data_set
    data_set - name of the dataset to be downloaded. The only option for now is 'mnist'.

    Outputs: raw_data_x, raw_data_y, num_classes, data_dim, img_sz
    raw_data_x - NxD matrix of x data (e.g. images to be classified)
    raw_data_y - Nx1 vector of y data (e.g. labels)
    num_classes - number of classes
    data_dim - dimension of each data point a.k.a. D
    img_sz - tuple of (x,y) sizes of images
    """

    # download data
    if data_set == 'mnist':
        raw_data = fetch_openml('mnist_784')
    else:
        assert False

    # embelishments
    raw_data_x = np.array((raw_data['data']+0.5)/256.0)
    raw_data_y = np.array(raw_data['target'].astype('int8'))

    # separate classes 0,1 into a new dataset
    accptbl_cats = (raw_data_y == 0) | (raw_data_y == 1)
    raw_data_x = raw_data_x[accptbl_cats]
    raw_data_y = raw_data_y[accptbl_cats]

    # some variables describing the data
    num_classes = 2
    img_sz = (28,28)
    data_dim = raw_data_x.shape[1]

    return raw_data_x, raw_data_y, num_classes, data_dim, img_sz


def rayleigh_tr(B, f, x, mu=1, deriv=False):
    """ 
    Compute loss function and its derivative w.r.t. B
    Loss = x.T @ L @ x + mu * tr(M)
    Assuming M = B^T @ B
    
    Input: B, f, x, mu, deriv
    B - NxN matrix of parameters determining the covariance matrix (M=B.T @ B)
    f - CxN matrix of feature vectors (Cx1) for all N training images
    x - Nx1 vector of image labels (1 for 1; -1 for 0)
    mu - scalar parameter of the loss function
    deriv - whether or not to compute and do the derivative
    
    Output: if deriv: E, dE
                else: E
    E - loss value with current B
    dE - derivative of loss function w.r.t. B at current B
    """
    
    # Some additional matrices:
    # F - NxNxC matrix of difference of feature vectors for each pair of training images
    # M - CxC matrix of covariance of each pair of training images; M = B.T @ B
    # W - NxN adjacency matrix of the graph; w_ij = exp(- F_ij.T @ M @ F_ij)
    # D - NxN degree matrix of graph; D = diag(W @ 1)
    # L - NxN graph laplacian matrix; L = D - W
    # X - NxN auxiliary matrix to compute dr
    
    # create F
    f_sz, num_train = f.shape
    Fj = np.broadcast_to([f.T], (num_train, num_train, f_sz))
    Fi = np.transpose(Fj, (1,0,2))
    F = Fi - Fj
    
    # create M
    M = B.T @ B

    # create W
    W = np.zeros((num_train, num_train))
    for i in range(num_train):
        for j in range(i+1, num_train):
            W[i][j] = np.exp(-1*(F[i][j].T @ M @ F[i][j]))
            W[j][i] = W[i][j]
            
    # create L
    D = np.diag(W @ np.ones(num_train))
    L = D - W
    
    # calculate r
    r = x.T @ L @ x
    
    # calculate E
    E = r + mu * np.trace(M)
    
    if deriv:
        
        # create X
        X = np.broadcast_to([x], (len(x), len(x))).T
        X = ((X - X.T)**2)
        
        # calculate drdM - TAKES TOO LONG
        XW = np.multiply(X, W)
        drdM = np.zeros((f_sz, f_sz))
        for s in range(f_sz):
            for t in range(f_sz):
                Fst = np.multiply(F[:,:,s], F[:,:,t])
                drdM[s][t] = np.sum(np.multiply(XW, Fst))
                
        # calculate dr w.r.t. B
        drdB = B @ (drdM + drdM.T)
        
        # calculate dE w.r.t. B
        dE = drdB + 2 * mu * B
        
        return E,dE
    else:
        return E


def gradient_descent(loss_func, opt_params):
    """
    Learn a set of parameters using gradient descent.

    Inputs: loss_func, opt_params
    loss_func - loss function to use for training; it should only take one input which is optimization parameter
    opt_params - parameters of the training algorithm (see below)

    Outputs: Theta, stats
    Theta - parameters at the end of optimization
    stats - dictionary of various statistics computed during training to be used
            for visualization and analysis
    """

    # Optimization parameters in opt_params
    epsilon0 = opt_params['epsilon0'] # starting learning rate for GD
    epsilon_decay = opt_params['epsilon_decay'] # decay factor for GD learning rate
    epsilon_jump = opt_params['epsilon_jump'] # decay factor for GD learning rate
    num_its = opt_params['num_its'] # number of iterations to run
    Theta0 = opt_params['Theta0'] # initial value for the parameters
    check_freq = opt_params['check_freq'] # how frequently to compute and print out statistics of learning
    print_checks = opt_params['print_checks'] # print info out when checkpointing
    seed = opt_params['seed'] if 'seed' in opt_params else 0

    if seed is not None:
        np.random.seed(seed)

    check_its = []
    check_times = []
    check_Thetas = []
    train_losss = []
    it_times = []
    epsilon = epsilon0
    start_t = time.time()
    Theta = Theta0
    for it in range(num_its):
                
        # Compute loss and its derivative with current parameter values
        E, dEdTheta = loss_func(Theta, deriv=True)

        # Find epsilon which decreases train loss
        epsilon *= epsilon_jump
        
        
        new_E = np.inf
        while new_E > E:
            # Update parameters with the GD update
            Theta1 = Theta - epsilon * dEdTheta
            new_E = loss_func(Theta1, deriv=False)
            epsilon *= epsilon_decay


        # Replace old value of Theta with new one
        Theta = Theta1
        
        # Compute the norm of the entire gradient to monitor
        nrmsq_dEdTheta = np.sum(dEdTheta**2)
        
        # Restore epsilon's working value 
        epsilon /= epsilon_decay

        if it%check_freq == 0 or it+1 == num_its:
            # Periodically compute the training loss/accuracy on the _full_ dataset
            # for reference.  Note this is rarely done in practice because it isn't
            # possible or is hugely impractical.  We're just doing it here to see 
            # how it relates to the values computed with a mini-batch
            E = loss_func(Theta, deriv=False)

            check_Thetas.append(Theta)
            check_its.append(it)
            check_times.append(time.time() - start_t)
            train_losss.append(E)

            if print_checks:
                print("{:4}: eps = {:.2e};"
                      "  train loss (E) = {:5.2f};"
                      "  ||dEdTheta|| = {:5.2f}".format(it, epsilon,
                                                        E, 
                                                        np.sqrt(nrmsq_dEdTheta)))
        it_times.append(time.time() - start_t)

    stats = { 'check_its':check_its, # Iteration numbers of checkpoints
            'check_times':check_times, # wall clock time of checkpoints
            'check_Thetas':check_Thetas, # Theta values at checkpoints
            'it_times':it_times, # wall clock time of each iteration
            'train_losss':train_losss} # loss of full training set at checkpoint iterations
    
    return Theta, stats

    
def fit_graph_with_rayleigh_tr(train_x, train_y, des, opt_params, show_L=False):  
    """
    Fit a similarity graph to training data using the rayleigh_tr loss function. 
    
    Inputs: train_x, train_y, des, opt_params, show_L=False
    train_x - NxD matrix of training x data (e.g. images)
    train_y - Nx1 vector of training y data (e.g. labels)
    des - NxDf matrix of feature vectors of training images
    opt_params - dict of gradient descent parameters covering 'epsilon0', 'epsilon_decay', 'epsilon_jump', 'num_its', 'check_freq', 'print_checks', 'Theta0'
    show_L - wether to return the graph Laplacian or the actual optimization variable B

    Outputs: if show_L: L, stats
             else: B, stats
    L - Similarity graph's graph laplacian of size NxN
    B - Learned optimization variable (M = B^T @ B) of size DfxDf
    stats - dict of gradient descent's resutls covering 'check_its', 'check_times', 'check_Thetas', 'it_times', 'train_losss'
    """


    B0 = opt_params['Theta0']

    if B0 is None:
        num_train, f_sz = des.shape
        B0 = 0.0001*np.random.random_sample((f_sz,f_sz))
        opt_params['Theta0'] = B0
    
    if opt_params['print_checks']:
        display_matrix(B0, 'B0')

    def rayleigh_tr_wrap(B, deriv=False):
        x = train_y*2-1
        f = des.T
        mu = 1
        return rayleigh_tr(B, f, x, mu, deriv)

    B, stats = gradient_descent(rayleigh_tr_wrap, opt_params)

    if show_L:
        L = B.T @ B
        Bs = stats['check_Thetas']
        Ls = [(b.T @ b) for b in Bs]
        stats['check_Thetas'] = Ls

        return L, stats
    else:
        return B, stats


def display_matrix(M, lbl=None):
    """
    Display a color map of a matrix's entries.
    
    Inputs: M, lbl=None
    M - matrix to be displayed
    lbl - title of the figure
    """

    plt.figure(figsize=(6,6))
    plt.imshow(M)
    plt.colorbar()
    if lbl is not None:
        plt.title(lbl)


def hist_of_entries(M, num_bins, lx=None, lbl=None, zeroline=False, peakline=False):
    """
    Draw a histogram of a given matrix's entries.

    Inputs: M, num_bins, lx=None, lbl=None, zeroline=False, peakline=False
    M - matrix whose histogram of entries is draw
    num_bins - number of bins in the histogram
    lx - defines interval of x-axis to be shown [-lx, lx]. 'None' to disable x-axis limiting.
    lbl - title of the histogram
    zeroline - wether or not to draw a vertical line at x=0
    peakline - wether or not to draw a vertical line at the histogram's peak

    Outputs: 
    """
    
    plt.figure()
    histvals, bins, _ = plt.hist(M.reshape(-1),num_bins)
    
    if lx is not None:
        plt.xlim([-lx, lx])
    
    if lbl is not None:
        _ = plt.title(lbl)
        
    if zeroline:
        plt.axvline(x = 0, color = 'm', label = 'zero line')
        
    if peakline:
        peak_x = (bins[np.argmax(histvals)] + bins[np.argmax(histvals)+1])/2
        plt.axvline(x = peak_x, color = 'r', label = 'peak line')
        return peak_x, np.max(histvals) # most repeated M value and the number of its repeats


def extrapolate(f, B, gt_train, gt_val):
    """
    Estimate labels of unknown vertices based on learned metric B (or M), then calculate estimation accuracy.
    
    Inputs: f, B, gt_train, gt_val
    f - Cx(Nt+Nv) matrix of feature vectors (Cx1) for all training and validation images
    B - CxC matrix of parameters determining the covariance matrix (M=B.T @ B)
    gt_train - (Nt)x1 vector of ground truth labels of training vertices
    gt_val - (Nv)x1 vector of ground truth labels of validation vertices
    
    Output: y_est, acc
    y_est - (Nv)x1 vector of estimated labels
    acc - accuracy a.k.a. percentage of correctly estimated labels over all validation vertices
    """
    
    # find number of validation and training images
    num_train = gt_train.shape[0]
    num_val = gt_val.shape[0]
    
    # create F
    f_sz, num_imgs = f.shape
    assert num_imgs == num_train+num_val
    Fj = np.broadcast_to([f.T], (num_imgs, num_imgs, f_sz))
    Fi = np.transpose(Fj, (1,0,2))
    F = Fi - Fj
    
    # create M
    M = B.T @ B
    
    # create W
    W = np.zeros((num_imgs, num_imgs))
    for i in range(num_imgs):
        for j in range(i+1, num_imgs):
            W[i][j] = np.exp(-1*(F[i][j].T @ M @ F[i][j]))
            W[j][i] = W[i][j]
            
    # create L21, L22
    D = np.diag(W @ np.ones(num_train+num_val))
    L = D - W
    L21 = L[num_train:, :num_train]
    L22 = L[num_train:, num_train:]
    
    # estimate unknown labels via graph filtering
    x_est = - np.linalg.inv(L22) @ L21 @ (gt_train*2-1)
    y_est = (x_est > 0).astype(int)
    
    # calculate accuracy
    acc = np.sum(y_est == gt_val)/num_val
    
    return y_est, acc



    