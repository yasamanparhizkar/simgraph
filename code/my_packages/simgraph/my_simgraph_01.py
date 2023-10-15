"""
With this package you can:
1. Use gradient descent algorithm to minimize a given loss function.
2. Evaluate a custom loss function (called 'rayleigh_tr') and its derivate.
3. Use GD and a custom loss fucntion to fit a similarity graph to any data represented by vectors and their corresponding class labels.
4. Use the optimized similarity graph (or more exactly, the optimized metric matrix M) to predict labels of validation data points.
5. Measure validation accuracy.
6. Measure running times of all gradient descent iterations. Three different time intervals are measured: 
        1. time to execute a whole iteration
        2. time to evaluate loss and its derivative inside each iteration
        3. time to find an optimized step size in each iteration
7. Measure the time it takes to predict validation labels.

Autorun and some visualization utilities are also provided. 
Autorun functions are named 'fit_graph' (for learning phase) and 'get_acc' (for validation phase).
Used as a block in Yasaman's master's thesis.

Author: Yasaman
Last modified: Nov 17, 2022
"""

import numpy as np
import time
import matplotlib.pyplot as plt


Fg = None       # NxNxC matrix of difference of feature vectors for each pair of training images.
drdMaux = None  # CxCxNxN tensor of fixed parts to compute drdM. drdMaux[s][t] = X . F[:,:,s] . F[:,:,t]
lbls = None     # Nx1 vector of labels of data points

def set_Fg(temp_Fg):
    """
    Set global matrix Fg.

    Input: temp_Fg
    temp_Fg - NxNxC matrix of difference of feature vectors for each pair of training images

    Output:
    """
    global Fg
    Fg = temp_Fg

def set_drdMaux(temp_drdMaux):
    """
    Set global variable drdMaux.

    Input: temp_drdMaux
    temp_drdMaux - value of drdMaux to be set

    Output:
    """
    global drdMaux
    drdMaux = temp_drdMaux

def set_lbls(temp_lbls):
    """
    Set global variable lbls.

    Input: temp_lbls
    temp_lbls - value of lbls to be set

    Output:
    """
    global lbls
    lbls = temp_lbls

def create_F(f):
    """
    Create matrix F out of matrix f.
    
    Input: f
    f - CxN matrix of feature vectors (Cx1) for all N training images
    
    Output: F
    F - NxNxC matrix of difference of feature vectors for each pair of training images
    """
    
    f_sz, num_train = f.shape
    Fj = np.broadcast_to([f.T], (num_train, num_train, f_sz))
    Fi = np.transpose(Fj, (1,0,2))
    F = Fi - Fj
    
    return F

def create_X(x):
    """
    Create matrix X out of vector x containing training image labels.
    Used as an auxiliary function with 'rayleigh_tr'.
    
    Input: x
    x - Nx1 vector of image labels (1 for 1; -1 for 0)
    
    Output: X
    X - NxN auxiliary matrix to compute dr; X[i][j] = (x_i - x_j)**2 where x_i is the label of image i.
    """
    
    num_train = len(x)
    X = np.broadcast_to([x], (num_train, num_train))
    X = (X.T - X)**2
    
    return X

def create_drdMaux(f, x):
    """
    Create the fixed part of drdM. For any value of B, this part gets multiplied with W(B) to yield drdM for that B.
    Used as an auxiliary function with 'rayleigh_tr'.

    Input: f, x
    f - CxN matrix of feature vectors (Cx1) for all N training images
    x - Nx1 vector of image labels (1 for 1; -1 for 0)

    Output: drdMaux
    drdMaux - CxCxNxN tensor of fixed parts to compute drdM. drdMaux[s][t] = X . F[:,:,s] . F[:,:,t]
    """

    f_sz, num_train = f.shape
    F = create_F(f)
    X = create_X(x)

    drdMaux = np.zeros((f_sz, f_sz, num_train, num_train))
    for s in range(f_sz):
        for t in range(s, f_sz):
            FsFt = np.multiply(F[:,:,s], F[:,:,t])
            drdMaux[s][t] = np.multiply(X, FsFt)
            drdMaux[t][s] = drdMaux[s][t]

    return drdMaux
            

# FACTOR OUT SOME CALCULATIONS TO INCREASE SPEED; USE SYMMETRY OF MATRICES, AND TENSOR MULTIPLICATION
def rayleigh_tr(B, deriv=False, mu = 1, x = None, F = None, drdMx = None):
    """ 
    Compute loss function and its derivative w.r.t. B
    Loss = x.T @ L @ x + mu * tr(M)
    Assuming M = B^T @ B
    
    Input: B, deriv, mu, x, F, drdMx
    B - CxC matrix of parameters determining the metric matrix (M=B.T @ B)
    deriv - whether or not to compute and do the derivative (default: False)
    mu - scalar parameter of the loss function (default: 1)
    x - Nx1 vector of image labels (1 for 1; -1 for 0). By default uses the global variable 'lbls'.
    F - NxNxC matrix of difference of feature vectors for each pair of training images (default: Fg which is a global variable)
    drdMx - CxCxNxN tensor of fixed parts to compute drdM. drdMx[s][t] = X . F[:,:,s] . F[:,:,t]
            where F is an NxNxC matrix of difference of feature vectors for each pair of training images,
            and X is an NxN marix such that X[i][j] = (x_i - x_j)**2 where x_i is the label of image i.
            By default is the global variable 'drdMaux'.
    
    Output: if deriv: E, dE
                else: E
    E - loss value with current B
    dE - derivative of loss function w.r.t. B at current B
    """

    global lbls
    global Fg
    global drdMaux
    
    if x is None:
        x = lbls
    if F is None:
        F = Fg
    if drdMx is None:
        drdMx = drdMaux
    
    # Some additional matrices:
    # M - CxC matrix of covariance of each pair of training images; M = B.T @ B
    # W - NxN adjacency matrix of the graph; w_ij = exp(- F_ij.T @ M @ F_ij)
    # D - NxN degree matrix of graph; D = diag(W @ 1)
    # L - NxN graph laplacian matrix; L = D - W
    
    f_sz = drdMx.shape[0]
    num_train = drdMx.shape[3]

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
        # calculate drdM - TAKES TOO LONG
        drdM = np.zeros((f_sz, f_sz))
        for s in range(f_sz):
            for t in range(s, f_sz):
                WXFsFt = np.multiply(W, drdMx[s][t])
                drdM[s][t] = -0.5 * np.sum(WXFsFt)
                drdM[t][s] = drdM[s][t]
                
        # calculate dr w.r.t. B
        drdB = 2 * (B @ drdM)
        
        # calculate dE w.r.t. B
        dE = drdB + 2 * mu * B
        
        return E,dE
    else:
        return E


def gradient_descent(loss_func, opt_params, show_nrmdE = False):
    """
    Learn a set of parameters using gradient descent.

    Inputs: loss_func, opt_params
    loss_func - loss function to use for training; it should only take one input which is optimization parameter
    opt_params - parameters of the training algorithm:
        'epsilon0': starting learning rate for GD
        'epsilon_decay': decay factor for GD learning rate
        'epsilon_jump': jump factor for GD learning rate
        'Theta0': initial value for the parameters
        'num_its': maximum number of iterations to run
        'check_freq': how frequently to compute and print out statistics of learning
        'print_checks': print info out when checkpointing
        'force_all_its': wether or not to run the maximum number of iterations (default: False)
        'threshod': ratio which stops the process if the loss improvment becomes less than (default: 0.01)
    show_nrmdE - whether or not to print norm of dE

    Outputs: Theta, stats
    Theta - parameters at the end of optimization
    stats - dictionary of various statistics computed during training to be used
            for visualization and analysis
    """

    # Optimization parameters in opt_params
    epsilon0 = opt_params['epsilon0']
    epsilon_decay = opt_params['epsilon_decay']
    epsilon_jump = opt_params['epsilon_jump']
    Theta0 = opt_params['Theta0']
    num_its = opt_params['num_its']
    check_freq = opt_params['check_freq']
    print_checks = opt_params['print_checks']
    force_all_its = opt_params['force_all_its'] if 'force_all_its' in opt_params else False
    threshold = opt_params['threshold'] if 'threshold' in opt_params else 0.01
    
    check_its = []
    # check_times = []
    # check_Thetas = []
    train_losss = []
    it_times = []
    stepsizeloop_times = []
    eval_times = []
    epsilon = epsilon0
    start_t = time.time()
    Theta = Theta0
    it = 0
    new_E = 0
    E = 1
    for it in range(num_its):
  
        # if the loss is not significantly improved, end the algorithm
        if (not force_all_its) and (E-new_E)/E <= threshold:
            break
                
        # Compute loss and its derivative with current parameter values
        s_start_t = time.time()
        E, dEdTheta = loss_func(Theta, deriv=True)
        eval_times.append(time.time() - s_start_t)

        # Find epsilon which decreases train loss
        s_start_t = time.time()
        epsilon *= epsilon_jump
        
        new_E = np.inf
        while new_E > E:
            # Update parameters with the GD update
            Theta1 = Theta - epsilon * dEdTheta
            new_E = loss_func(Theta1, deriv=False)
            epsilon *= epsilon_decay
            
        stepsizeloop_times.append(time.time() - s_start_t)
            
        # Replace old value of Theta with new one
        Theta = Theta1
        
        # Restore epsilon's working value 
        epsilon /= epsilon_decay

        if it%check_freq == 0 or it+1 == num_its or (E-new_E)/E <= threshold:

            # check_Thetas.append(Theta)
            check_its.append(it)
            # check_times.append(time.time() - start_t)
            train_losss.append(new_E)

            # Compute the norm of the entire gradient to monitor
            if show_nrmdE:
                new_E, dEdTheta = loss_func(Theta, deriv=True)
                nrmsq_dEdTheta = np.sum(dEdTheta**2)

            if print_checks and show_nrmdE:
                print("{:4}: eps = {:.2e};"
                      "  train loss (E) = {:5.2f};"
                      "  ||dEdTheta|| = {:5.2f}".format(it, epsilon,
                                                        new_E, 
                                                        np.sqrt(nrmsq_dEdTheta)))
            elif print_checks:
                print("{:4}: eps = {:.2e};"
                      "  train loss (E) = {:5.2f}".format(it, epsilon, new_E))

        it_times.append(time.time() - start_t)

    stats = {'check_its':check_its, # Iteration numbers of checkpoints
            # 'check_times':check_times, # wall clock time of checkpoints
            # 'check_Thetas':check_Thetas, # Theta values at checkpoints
            'it_times':it_times, # wall clock time of each iteration
            'stepsizeloop_times':stepsizeloop_times, # time to find the best stepsize in each iteration
            'eval_times':eval_times, # time to evaluate loss and its derivative at the start of each iteration
            'train_losss':train_losss} # loss of full training set at checkpoint iterations
    
    return Theta, stats


def fit_graph(dess, lbls, opt_params, seed = None):  
    """
    Fit a similarity graph to training data using the rayleigh_tr loss function. 
    Autorun of some functions of this package.
    
    Inputs: dess, lbls, opt_params, show_L=False
    dess - NxDf matrix of feature vectors of training data ponits (e.g. images)
    lbls - Nx1 vector of training labels of corresponding data points    
    opt_params - dict of parameters for gradient descent:
        'epsilon0': starting learning rate for GD
        'epsilon_decay': decay factor for GD learning rate
        'epsilon_jump': jump factor for GD learning rate
        'Theta0': initial value for the parameters. if None, it will be initialized randomly.
        'num_its': maximum number of iterations to run
        'check_freq': how frequently to compute and print out statistics of learning
        'print_checks': print info out when checkpointing
        'force_all_its': wether or not to run the maximum number of iterations (default: False)
        'threshod': ratio which stops the process if the loss improvment becomes less than (default: 0.01)
    seed - seed for random initialization of optimization parameters if it is not given (default: None, in which case no new seed is used)

    Outputs: B, stats
    B - DfxDf matrix of optimized parameters defining the metric matrix (M = B^T @ B)
    stats - dict of gradient descent's statistics
    """

    if seed is not None:
        np.random.seed(seed)

    if opt_params['Theta0'] is None:
        num_train, f_sz = dess.shape
        opt_params['Theta0'] = 0.0001 * np.random.random_sample((f_sz,f_sz))

    x = lbls * 2 - 1
    f = dess.T
    set_lbls(x)
    set_Fg(create_F(f))
    set_drdMaux(create_drdMaux(f, x))

    B, stats = gradient_descent(rayleigh_tr, opt_params)

    return B, stats


def create_W(B, F = None):
    """
    Create the graph's adjacency matrix from B. Used as an auxiliary function for 'rayleigh_tr'.
    
    Input: B, F
    B - CxC matrix of parameters determining the metric matrix (M = B.T @ B)
    F - NxNxC matrix of difference of feature vectors for each pair of training images (default: None, in which case global Fg is used)
    
    Output: W
    W - NxN adjacency matrix of the similarity graph    
    """

    global Fg
    if F is None:
        F = Fg

    num_train = F.shape[0]
      
    M = B.T @ B
    
    W = np.zeros((num_train, num_train))
    for i in range(num_train):
        for j in range(i+1, num_train):
            W[i][j] = np.exp(-1*(F[i][j].T @ M @ F[i][j]))
            W[j][i] = W[i][j]
                
    return W

def create_L(W):
    """
    Create the graph laplacian from its adjacency matrix. L = diag(W 1) - W

    Input: W
    W - NxN adjacency matrix of a graph

    Output: L
    L - NxN graph laplacian
    """

    num_train = W.shape[0]
    D = np.diag(W @ np.ones(num_train))
    L = D - W

    return L

def in_order(W, x):
    """
    Sorts the adjacency matrix w.r.t. labels of nodes. All nodes of the same label are gathered in a block.
    Labels are sorted in ascending order.
    
    Input: W, x
    W - NxN adjacency matrix of the similarity graph
    x - Nx1 vector of labels of corresponding nodes
    
    Output: W_ord, order
    W_ord - NxN adjacency matrix where all same labeled nodes come together
    origin - Nx1 vector depicting the new order of nodes; origin[i] = index of image #i in the original random permutation
    cats - all possible labels in the order they come in W_ord
    """
    
    # get all possible labels in ascending order
    cats = sorted(list(dict.fromkeys(x)))

    # original indices of data points
    orig_i = np.arange(x.shape[0])

    # sort rows (along axis 0)
    W1 = W[x == cats[0], :]
    for i in range(1, len(cats)):
        W1 = np.append(W1, W[x == cats[i], :], axis=0)
    
    # sort columns (along axis 1)
    W_ord = W1[:, x == cats[0]]
    for i in range(1, len(cats)):
        W_ord = np.append(W_ord, W1[:, x == cats[i]], axis=1)

    # get new places of data points
    origin = orig_i[x == cats[0]]
    for i in range(1, len(cats)):
        origin = np.append(origin, orig_i[x == cats[i]], axis=0)   
    
    return W_ord, origin, cats


def extrapolate(B, train_des, val_des, train_gt):
    """
    Estimate labels of unknown vertices based on learned metric B (M = B.T @ B).
    
    Inputs: B, train_des, val_des, train_gt
    B - CxC matrix of parameters determining the metric matrix (M = B.T @ B)
    train_des - NtxC matrix of descriptors of training data points (e.g. images)
    val_des - NvxC matrix of descriptors of validation data points (e.g. images)
    train_gt - Ntx1 vector of labels of corresponding training data points (1 for 1, and -1 for 0)
    
    Output: y_est
    y_est - Nvx1 vector of estimated labels (1 for 1, and -1 for 0)
    t - time it took to estimate labels in seconds
    """

    start_t = time.time()
    
    # find number of validation and training images
    num_train = train_des.shape[0]
    num_val = val_des.shape[0]
    f_sz = B.shape[0]

    # concatenate descriptors of training and validation data points
    data_des = np.append(train_des, val_des, axis=0)
    num_data = data_des.shape[0]
    assert num_data == num_train + num_val
    assert data_des.shape[1] == f_sz
    
    # create F
    F = create_F(data_des.T)

    # create W
    W = create_W(B, F)
            
    # create L21, L22
    L = create_L(W)
    L21 = L[num_train:, :num_train]
    L22 = L[num_train:, num_train:]
    
    # estimate unknown labels via graph filtering
    y_est = -1 * (np.linalg.inv(L22) @ L21 @ train_gt)

    t = time.time() - start_t
    
    return y_est, t


def get_acc(B, train_des, train_y, val_des, val_y):
    """
    Estimate labels of unknown vertices based on learned metric B (M = B.T @ B), and calculate accuracy.
    Autorun of 'extrapolate' function in this package.
    
    Inputs: B, train_des, train_y, val_des, val_y
    B - CxC matrix of parameters determining the metric matrix (M = B.T @ B)
    train_des - NtxC matrix of descriptors of training data points (e.g. images)
    train_y - Ntx1 vector of labels of corresponding training data points
    val_des - NvxC matrix of descriptors of validation data points (e.g. images)
    val_y - Ntx1 vector of labels of corresponding training data points

    Output: y_est, acc
    acc - accuracy a.k.a. percentage of correctly estimated labels over all validation vertices
    y_est - Nvx1 vector of estimated labels
    t - time it took to estimate validation labels in seconds
    """

    train_gt = train_y * 2 - 1
    y_est, t = extrapolate(B, train_des, val_des, train_gt)

    y_th = (y_est > 0).astype(int)
    acc = np.sum(y_th == val_y)/len(val_y)

    return acc, y_est, t










###########################
# Visualization Utilities #
###########################

def display_matrix(M, title=None):
    """
    Display a color map of a matrix's entries.
    
    Inputs: M, title=None
    M - matrix to be displayed
    title - title of the figure
    """
    
    plt.figure(figsize=(4,4))
    plt.imshow(M)
    plt.colorbar()
    if title is not None:
        plt.title(title)


def hist_of_entries(M, num_bins, lx=None, lbl=None, zeroline=False, peakline=False):
    """
    Draw a histogram of a given matrix's entries.

    Inputs: M, num_bins, lx=None, lbl=None, zeroline=False, peakline=False
    M - matrix whose histogram of entries is drawn
    num_bins - number of bins in the histogram
    lx - defines interval of x-axis to be shown [-lx, lx]. 'None' to disable x-axis limiting.
    lbl - title of the histogram
    zeroline - wether or not to draw a vertical line at x=0
    peakline - wether or not to draw a vertical line at the histogram's peak

    Outputs: 
    """
    
    plt.figure(figsize=(4,4))
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



