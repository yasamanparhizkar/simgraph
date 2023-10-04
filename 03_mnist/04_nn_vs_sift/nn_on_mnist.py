"""
Classify digits 0 and 1 from the mnist dataset.
Use a neural network with one hidden layer consisting of 5 nodes.
Activation function: relu + softmax for the last layer
Loss function: cross-entropy
Optimization method: gradient descent
For more information, refer to 'nn_on_mnist_03.ipynb'

Author: Yasaman
Last modified: August 26 2022
"""

import numpy as np
import scipy as sp
import time
from sklearn.datasets import fetch_openml # to download mnist data

def learn_nn(raw_data_x, raw_data_y, portion, fracs, img_sz=(28,28), num_classes=2, show_msgs=False):
    """
    Receive downloaded mnist data as input.
    Randomly select a specific portion of the data and split it into training and validation datasets.
    Fit a simple one-layer neural network to training data via gradient descent.
    (Refer to the code for details of nn's architecture.)
    Calculate accuracy using the validation data.
    
    Inputs: raw_data_x, raw_data_y, portion, fracs, img_sz, show_msgs
    raw_data_x - NtxD tensor of raw input images to be classified into two classes
    raw_data_y - Ntx1 vector of corresponding labels for images; described as 0 and 1
    portion - the portion of raw data to be used in the algorithm (whole dataset might be too large)
    fracs - fractions of split between training and validation data (e.g. [0.6, 0.4])
    
    Outputs: num_train, num_val, acc, Theta
    num_train - number of training images
    num_val - number of validation images
    acc - accuracy obtained on validation set
    Theta - final optimized parameters of the neural network
    """
    
    # split data into training and validation datasets
    train_x,train_y,test_x,test_y = split_data(raw_data_x,raw_data_y,fracs=[portion,portion],seed=None)
    train_x,train_y,  val_x,val_y = split_data(train_x,train_y,fracs=fracs,seed=None)
    
    # define number of training and validation images
    num_train = len(train_y)
    num_val = len(val_y)
    
    # define gradient descent parameters
    opt_params = { 'epsilon0':1, 'epsilon_decay':0.5, 'epsilon_jump': 1.5,
               'num_its':10, 'check_freq':1, 'print_checks':show_msgs, 'Theta0':None, 'seed':None }
    
    # define neural network architecture
    arch_params = { 'num_hidden_layers':1, 'num_hidden_units':5, 'num_outs':num_classes,
                    'act_func':relu, 'out_func':softmax, 'loss_func':crossent }

    # fit neural network to the training images and measure accuracy on validation set
    Theta, stats = nn_fit(train_x, train_y, val_x, val_y, opt_params, arch_params)
    
    # accuracy obtained in the final iteration of gradient descent
    acc = stats['val_accs'][-1]
    
    return num_train, num_val, acc, Theta  


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


def onehot_encode(y,num_classes=2):
    """
    Convert an Nx1 vector of integer class labels from 0 to C-1 into 
    an NxC matrix where each row is a one-hot encoding of the 
    """
    return np.eye(num_classes)[y.reshape(-1)]


def relu(a,deriv=False):
    """
    relu activation function max(a,0) and its derivative

    Input: a - NxC matrix of values where each row of a corresponds to a single input to the current layer with C units.
    Output: r,dr
        r - NxC matrix of function values at the values in a
        dr - NxC matrix of the derivatives of the relu evaluated at the values of a
    """
    r = np.maximum(a,0)
    if deriv:
        dr = (r == a)*1

        return r, dr
    else:
        return r


def softmax(a,params=None,deriv=False):
    """
    softmax function s = exp(a)/sum(exp(a)) and its Jacobian where J[i,j] = ds[i]/da[j] = -(s[i]*s[j]) if i!=j 
    and (s[i]*(1-s[i])) if i==j.
    
    Input: a - NxC matrix of values where each row of a corresponds to a single input to the softmax function
    Output: s,ds
        s - NxC matrix of softmax function values at the values in a
        ds - NxCxC tensor of Jacobians of the softmax function of each input
     """
     
    N,C = a.shape

    c = np.amax(a,1).reshape(N,1)
    s = np.exp(a-c)/np.sum(np.exp(a-c), 1).reshape(N,1)
    if deriv:
        ds = np.array([np.diag(s[i]) - np.outer(s[i],s[i]) for i in range(N)]) # NEEDS BETTER EFFICIENCY!!!!!!!

        return s, ds
    else:
        return s


def crossent(y,yhat, num_classes=2, deriv=False):
    """
    cross entropy loss L = -\sum_c (y==c) log(p(y==c)).
    
    Input: y,yhat
        y - Nx1 vector of class labels for N inputs
        yhat - NxC matrix of probabilities of each class c for each input i
            i.e., yhat[i,c] is p(y[i]==c)
    
    Output: l,dl
        l - Nx1 vector of cross entropy values
        dl - Nx1xC matrix of derivatives
     """

    N,C = yhat.shape
    assert y.shape[0] == N

    p_correctlbl = (yhat[np.arange(N), y]).reshape(N,1)
    l = -np.log(p_correctlbl)
    if deriv:
        dl = onehot_encode(y, num_classes)*(-1/p_correctlbl)
        dl.reshape(N,1,C)

        return l,dl
    else:
        return l


def nn_forward(x,acts,Theta):
    """
    Compute the forward pass of a neural network model for a set of inputs.

    Input: x,loss,acts,Theta,deriv
    x - NxD matrix of input vectors
    acts - list of length L of activation functions for each layer
    Theta - list of length L of the parameters of each layer.  In general, 
            Theta[i] = (Wi,bi) where Wi and bi are the weight matrix and bias 
            vector for layer i.

    Output: z
    z - an NxC matrix where C is the dimensionality of the output
    """
    N = x.shape[0]
    num_layers = len(acts)
    z = x

    for i in range(num_layers):
        act = acts[i] # activation function at this layer
        Wi,bi = Theta[i] # parameters at this layer 

        ai = (Wi @ z.T + bi).T
        z = act(ai,deriv=False)

    return z

def nn_loss(x,y,loss,acts,Theta,deriv=False):
    """
    Compute the objective function (and gradients) of the loss function for a set
    of data using a neural network model.

    Input: x,y,loss,acts,Theta,deriv
    x - NxD matrix of input vectors
    y - Nx1 vector of ground truth labels
    loss - function handle for the loss function
    acts - list of length L of activation functions for each layer
    Theta - list of length L of the parameters of each layer.  In general, 
            Theta[i] = (Wi,bi) where Wi and bi are the weight matrix and bias 
            vector for layer i.
    deriv - whether or not to compute and do the derivative

    Output: if deriv: E, dEdTheta, acc
              else: E, acc
    E - value of the loss function
    dEdTheta - the derivatives of the loss function for the parameters in Theta
               it has the same layout as Theta, specifically, 
                 dEdTheta[i] = (dEdWi,dEdbi) is the
               derivative of E with respect to the weight matrix and bias vector
               of layer i.
    acc - accuracy of predictions on the given data
    """
    N = x.shape[0]
    num_layers = len(acts)
    z = x

    zs = [] # input and output of each layer
    dzi_dais = [] # derivative of activation function of each layer
    for i in range(num_layers):
        act = acts[i] # activation function at this layer
        Wi,bi = Theta[i] # parameters at this layer 

        ai = (Wi @ z.T + bi).T
        if deriv:
            zs.append(z) # store input to the layer, needed for derivative wrt W

            z, dzi_dai = act(ai,deriv=True)
            dzi_dais.append(dzi_dai) # store derivative of activation function
        else:
            z = act(ai,deriv=False)

    maxc = np.argmax(z,axis=1) # Index of class with maximum probability
    acc = np.sum(maxc == y)/N

    if deriv:
        L, dLdz = loss(y,z,deriv=True)
        E = np.mean(L)

        dEdTheta = [None]*len(acts)

        # Initialize backprop with the gradients of the loss values wrt to the
        # output of the last layer
        # 
        # dLdzi has shape Nx1xDi where Di is the dimensionality of layer i because
        # we are treating it as the Jacobian of a function with 1 output
        dLdzi = (1.0/N)*dLdz

        # Backward pass
        for i in reversed(range(num_layers)):
            act = acts[i] # activation function at this layer
            Wi,bi = Theta[i] # parameters at this layer 
            zi_1 = zs[i] # input to the current layer
            dzi_dai = dzi_dais[i] # derivatives of layer activation function

            Di, Di_1 = Wi.shape
            # TODO: Backpropogate the derivative dLdzi through the activation function 
            # to compute dLdai.
            if dzi_dai.ndim == 2:
                # If dzi_dai only has two dimensions, then this activation function is 
                # element wise, so we only need to multiply each element of dLdzi with
                # it's corresponding derivative in dzi_dai.
                dLdai = (dLdzi * dzi_dai).reshape(N,1,Di)
            else:
                # If dzi_dai only has three dimensions, then this activation function is 
                # not elementwise and has a full Jacobian so we need to multiply the
                # jacobians dLzi[i] (a 1xDi matrix) and dzi_dai[i] (a DixDi matrix) for
                # each data point i.  For efficiency try to avoid looping by using the
                # @ operator and/or the np.matmul function.
                dLdai = (dLdzi.reshape(N,1,Di)) @ dzi_dai

            # TODO: Backpropogate the derivative dLdai through the linear
            # transformation to compute dLdzi for the next layer.
            dLdzi = (dLdai @ Wi).reshape(N,Di_1)

            # Derivatives of the parameters of the linear layer
            dEdW = np.sum(dLdai.reshape((N,Di,1)) * zi_1.reshape((N,1,Di_1)),axis=0)
            dEdb = np.sum(dLdai.reshape((N,Di)),axis=0).reshape((Di,1))
            dEdTheta[i] = (dEdW,dEdb)

        return E, dEdTheta, acc
    else:
        L = loss(y,z,deriv=False)
        E = np.mean(L)
        return E, acc


def nn_fit(train_x, train_y, val_x, val_y,
           opt_params, arch_params):
    """
    Train a neural network using gradient descent.

    Inputs: train_x, train_y, val_x, val_y, opt_params, arch_params
    train_x - NxD matrix of full training set with N samples and D dimensions
    train_y - Nx1 vector of training output values
    val_x, val_y - a validation set of samples to monitor performance
    opt_params - parameters of the training algorithm (see below)
    arch_params - parameters of the neural network architecture (see below)

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
    Theta0 = opt_params['Theta0'] # initial value for the parameters, can be None to randomly initialize
    check_freq = opt_params['check_freq'] # how frequently to compute and print out statistics of learning
    print_checks = opt_params['print_checks'] # print info out when checkpointing
    seed = opt_params['seed'] if 'seed' in opt_params else 0

    if seed is not None:
        np.random.seed(seed)

    # Neural Network architecture parameters in arch_params
    num_hidden_layers = arch_params['num_hidden_layers'] # number of hidden layers
    num_hidden_units = arch_params['num_hidden_units'] # number of units per hidden layer (all layers are assumed to have the same number)
    num_outs = arch_params['num_outs'] # number of outputs, for classification the number of classes
    act_func = arch_params['act_func'] # hidden layer activation function
    out_func = arch_params['out_func'] # output layer activation function
    loss_func = arch_params['loss_func'] # loss function to use for training

    train_N, data_dim = train_x.shape

    assert num_hidden_layers >= 0
    acts = [act_func]*num_hidden_layers + [out_func]
    if Theta0 is None:
        # If no initial value of parameters is given, this randomly generates one.
        if num_hidden_layers > 1:
            Theta0 = [    (0.1*np.random.randn(num_hidden_units,data_dim),      np.zeros((num_hidden_units,1))) ] + \
                   [ (0.1*np.random.randn(num_hidden_units,num_hidden_units), np.zeros((num_hidden_units,1))) ]*(num_hidden_layers-1) + \
                   [    (0.1*np.random.randn(num_outs,num_hidden_units),              np.zeros((num_outs,1))) ]
        elif num_hidden_layers == 1:
            Theta0 = [ (0.1*np.random.randn(num_hidden_units,data_dim), np.zeros((num_hidden_units,1))) ] + \
                   [ (0.1*np.random.randn(num_outs,num_hidden_units),         np.zeros((num_outs,1))) ]
        elif num_hidden_layers == 0:
            Theta0 = [ (0.1*np.random.randn(num_outs,data_dim),   np.zeros((num_outs,1))) ]

    check_its = []
    check_times = []
    it_times = []
    data_pts = []
    check_data_pts = []
    check_Thetas = []
    train_losss = []
    train_accs = []
    val_losss = []
    val_accs = []
    epsilon = epsilon0
    start_t = time.time()
    num_data_proc = 0
    Theta = Theta0
    for it in range(num_its):
        # Compute loss and its derivative with current parameter values
        train_loss, dEdTheta, train_acc = nn_loss(train_x,train_y,loss_func,acts,Theta,deriv=True)
        num_data_proc += train_N

        # Find epsilon which decreases train loss
        epsilon *= epsilon_jump
        new_train_loss = np.inf
        while new_train_loss > train_loss:
            nrmsq_dEdTheta = 0
            Theta1 = []
            # Loop over all layers and update their parameters with the GD update
            for (Wi,bi),(dEdWi,dEdbi) in zip(Theta,dEdTheta):
                # TODO: Wi1 and bi1 should be the value of the parameters after the
                # gradient update based on the derivatives dEdWi,dEdbi, previous
                # parameter values Wi,bi and stepsize epsilon
                Wi1 = Wi - epsilon * dEdWi
                bi1 = bi - epsilon * dEdbi

                Theta1.append((Wi1,bi1))

                # Compute the norm of the entire gradient to monitor
                nrmsq_dEdTheta += np.sum(dEdWi**2) + np.sum(dEdbi**2)
            
            new_train_loss, new_train_acc = nn_loss(train_x,train_y,loss_func,acts,Theta,deriv=False)
            epsilon *= epsilon_decay

        # Replace old value of Theta with new one
        Theta = Theta1
        

        if it%check_freq == 0 or it+1 == num_its:
            # Periodically compute the training loss/accuracy on the _full_ dataset
            # for reference.  Note this is rarely done in practice because it isn't
            # possible or is hugely impractical.  We're just doing it here to see 
            # how it relates to the values computed with a mini-batch
            train_loss, train_acc = nn_loss(train_x,train_y,
                                       loss_func,acts,Theta,deriv=False)


            # Periodically compute the validation loss/accuracy for reference.
            val_loss, val_acc = nn_loss(val_x,val_y,
                                   loss_func,acts,Theta,deriv=False)

            check_Thetas.append(Theta)
            check_its.append(it)
            check_data_pts.append(num_data_proc)
            check_times.append(time.time() - start_t)
            train_losss.append(train_loss)
            train_accs.append(train_acc)
            val_losss.append(val_loss)
            val_accs.append(val_acc)
            if print_checks:
                print("{:4}: eps = {:.2e};"
                      "  train: loss = {:5.2f}, acc = {:.2f};"
                      "  val: loss = {:5.2f}, acc = {:.2f};"
                      "  ||dEdTheta|| = {:5.2f}".format(it, epsilon,
                                                        train_loss, train_acc,
                                                        val_loss, val_acc,
                                                        np.sqrt(nrmsq_dEdTheta)))
        data_pts.append(num_data_proc)
        it_times.append(time.time() - start_t)

    stats = { 'check_its':check_its, # Iteration numbers of checkpoints
            'check_times':check_times, # wall clock time of checkpoints
            'check_data_pts':check_data_pts, # number of training samples processed at check points
            'check_Thetas':check_Thetas, # Theta values at checkpoints
            'it_times':it_times, # wall clock time of each iteration
            'data_pts':data_pts, # number of training samples processed at each iteration
            'train_losss':train_losss, 'train_accs':train_accs, # loss and accuracy of full training set at checkpoint iterations
            'val_losss':val_losss, 'val_accs':val_accs } # loss and accuracy of validation set at checkpoint iterations
    return Theta, stats






