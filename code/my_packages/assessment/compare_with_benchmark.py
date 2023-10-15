"""
With this package you can:
1. Create a simple neural network, plus the backpropagation alg. to compute its derivative.
2. Train a given neural network using gradient descent, and measure its accuracy using a validation dataset.
3. Measure running times of all gradient descent iterations. Three different time intervals are measured: 
        1. time to execute a whole iteration
        2. time to evaluate loss and its derivative inside each iteration
        3. time to find an optimized step size in each iteration
4. Compare performance and running times of another model with a given neural network as a benchmark.

Autorun and some visualization utilities are also provided. 
Used as a block in Yasaman's master's thesis.

Author: Yasaman
Last modified: Nov 17, 2022
"""

import numpy as np
import time


def onehot_encode(y,num_classes=2):
    """
    Convert an Nx1 vector of integer class labels from 0 to C-1 into 
    an NxC matrix where each row is a one-hot encoding of the corresponding class label.
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

    p_correctlbl = (yhat[np.arange(N), y]).reshape(N,1) + 0.001
    # print(p_correctlbl)
    l = -np.log(p_correctlbl)
    if deriv:
        dl = onehot_encode(y, num_classes)*(-1/p_correctlbl)
        dl.reshape(N,1,C)

        return l,dl
    else:
        return l


def forward(x,acts,Theta):
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

def nn_loss(x,y,loss,acts,Theta,deriv=False,estimates=False):
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
            # Backpropogate the derivative dLdzi through the activation function 
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
        if estimates:
            return E, acc, maxc
        else:
            return E, acc


def fit(train_x, train_y, opt_params, arch_params, show_nrmdE = False):
    """
    Train a neural network using gradient descent.

    Inputs: train_x, train_y, opt_params, arch_params, show_nrmdE
    train_x - NxD matrix of full training set with N samples and D dimensions
    train_y - Nx1 vector of training output values
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
        'seed': seed to randomly initialize the parameters if needed (default: None, in which case no new seed is used)
    arch_params - parameters of the neural network architecture:
        'num_hidden_layers': number of hidden layers
        'num_hidden_units': number of units per hidden layer (all layers are assumed to have the same number)
        'num_outs': number of outputs, for classification the number of classes
        'act_func': hidden layer activation function
        'out_func': output layer activation function
        'loss_func': loss function to use for training
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
    num_its = opt_params['num_its']
    Theta0 = opt_params['Theta0']
    check_freq = opt_params['check_freq']
    print_checks = opt_params['print_checks']
    force_all_its = opt_params['force_all_its'] if 'force_all_its' in opt_params else False
    threshold = opt_params['threshold'] if 'threshold' in opt_params else 0.01
    seed = opt_params['seed'] if 'seed' in opt_params else None

    if seed is not None:
        np.random.seed(seed)

    # Neural Network architecture parameters in arch_params
    num_hidden_layers = arch_params['num_hidden_layers']
    num_hidden_units = arch_params['num_hidden_units']
    num_outs = arch_params['num_outs']
    act_func = arch_params['act_func']
    out_func = arch_params['out_func']
    loss_func = arch_params['loss_func']

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
        if (not force_all_its) and (E-new_E+1e-2)/(E+1e-2) <= threshold:
            break

        # Compute loss and its derivative with current parameter values
        s_start_t = time.time()
        E, dEdTheta, train_acc = nn_loss(train_x,train_y,loss_func,acts,Theta,deriv=True)
        eval_times.append(time.time() - s_start_t)

        # Find epsilon which decreases train loss
        s_start_t = time.time()
        epsilon *= epsilon_jump

        new_E = np.inf
        while new_E > E:
            Theta1 = []
            # Loop over all layers and update their parameters with the GD update
            for (Wi,bi),(dEdWi,dEdbi) in zip(Theta,dEdTheta):
                Wi1 = Wi - epsilon * dEdWi
                bi1 = bi - epsilon * dEdbi
                Theta1.append((Wi1,bi1))

            new_E, new_train_acc = nn_loss(train_x,train_y,loss_func,acts,Theta1,deriv=False)
            epsilon *= epsilon_decay

        stepsizeloop_times.append(time.time() - s_start_t)

        # Replace old value of Theta with new one
        Theta = Theta1

        # Restore epsilon's working value 
        epsilon /= epsilon_decay
        
        if it%check_freq == 0 or it+1 == num_its or (E-new_E+1e-2)/(E+1e-2) <= threshold:
            check_its.append(it)
            train_losss.append(new_E)

            # Compute the norm of the entire gradient to monitor
            if show_nrmdE:
                new_E, dEdTheta, new_train_acc = nn_loss(train_x,train_y,loss_func,acts,Theta,deriv=True)
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

    stats = { 'check_its':check_its, # Iteration numbers of checkpoints
            'it_times':it_times, # wall clock time of each iteration
            'stepsizeloop_times':stepsizeloop_times, # time to find the best stepsize in each iteration
            'eval_times':eval_times, # time to evaluate loss and its derivative at the start of each iteration
            'train_losss':train_losss} # loss and accuracy of full training set at checkpoint iterations

    return Theta, stats


def get_acc(val_x, val_y, arch_params, Theta):
    """
    Get validation loss and accuracy of a given neural network.

    Input: val_x, val_y, arch_params, Theta
    val_x - NxD matrix of full validation set with N samples and D dimensions
    val_y - Nx1 vector of validation output values
    arch_params - parameters of the neural network architecture:
        'num_hidden_layers': number of hidden layers
        'num_hidden_units': number of units per hidden layer (all layers are assumed to have the same number)
        'num_outs': number of outputs, for classification the number of classes
        'act_func': hidden layer activation function
        'out_func': output layer activation function
        'loss_func': loss function to use for training
    Theta - optimized parameters of the neural network

    Output: val_acc, val_loss
    val_acc - validation accuracy
    val_loss - validation loss computed with given loss function
    """

    # Neural Network architecture parameters in arch_params
    num_hidden_layers = arch_params['num_hidden_layers']
    act_func = arch_params['act_func']
    out_func = arch_params['out_func']
    loss_func = arch_params['loss_func']

    assert num_hidden_layers >= 0
    acts = [act_func]*num_hidden_layers + [out_func]

    val_loss, val_acc, y_est = nn_loss(val_x,val_y,loss_func,acts,Theta,deriv=False,estimates=True)

    return val_acc, y_est


def assess(get_data, get_acc, data_sz, rpts, show_msgs = True):
    """
    Assess performance and running times of a classification method.

    Input: get_data, get_acc, data_sz, rpts
    get_data - function to get train and val data sets
    get_acc - function to train the model and validate it
    data_sz - list of all data sizes to test. each data size corresponds with one assessment point in this function's return values.
    rpts - list determining how many times to run the model for each data size.

    Output: nums, accs, times
    nums - dict depicting number of training and validation data points used for each assessment of the model
        'train_nums' - average number of training data points used to evaluate one assessment point
        'train_nums_std' - standard deviation of number of training points used to evaluate one assessment point
        'val_nums' - average number of validation data points used to evaluate one assessment point
        'val_nums_std' - standard deviation of number of validation points used to evaluate one assessment point
    accs - dict depicting obtained accuracies for each assessment of the model
        'val' - average accuracies for each assessment point
        'std' - standard deviation of accuracies for each assessment point
    times - dict depicting elapsed time for each assessment of the model and each preparation of data
        data_t - average elapsed time of each preparation of training and validation data sets (includes feature extraction)
        data_t_std - standard deviation of elapsed time of each preparation of training and validation data sets
        run_t - average elapsed time of each run of the classification method for each assessment point
        run_t_std - standard deviation of elapsed times to run the classification method for each assessment point
    """

    assert len(data_sz) == len(rpts)

    train_nums = []
    train_nums_std = []
    val_nums = []
    val_nums_std = []
    accs_val = []
    accs_std = []
    data_t = []
    data_t_std = []
    run_t = []
    run_t_std = []

    if show_msgs:
        print('{: ^10.10} {: ^10.10} {: ^10.10} {: ^10.10} {: ^10.10} {: ^10.10} {: ^10.10}'\
            .format('data size', 'iteration', 'train_num', 'val_num', 'acc(%)', 'dtime(ms)', 'rtime(ms)'))
        print('----------------------------------------------------------------------------')

    for i in range(len(data_sz)):
        train_N = []
        val_N = []
        accs = []
        dtimes = []
        rtimes  = []

        for j in range(rpts[i]):
            # get train and val data
            start_t = time.time()
            train_num, val_num, train_data, val_data = get_data(data_sz[i])
            dtime = time.time() - start_t
            dtimes.append(dtime)
            train_N.append(train_num)
            val_N.append(val_num)

            # train and validate model
            start_t = time.time()
            acc = get_acc(train_data, val_data)
            rtime = time.time() - start_t
            rtimes.append(rtime)
            accs.append(acc)

            if show_msgs:
                print('{: ^10} {: ^10} {: ^10} {: ^10} {: ^10.4} {: ^10.4} {: ^10.4}'\
                    .format(data_sz[i], j, train_num, val_num, acc*100, dtime*1000, rtime*1000))

        train_nums.append(np.mean(train_N))
        train_nums_std.append(np.std(train_N))
        val_nums.append(np.mean(val_N))
        val_nums_std.append(np.std(val_N))
        accs_val.append(np.mean(accs))
        accs_std.append(np.std(accs))
        data_t.append(np.mean(dtimes))
        data_t_std.append(np.std(dtimes))
        run_t.append(np.mean(rtimes))
        run_t_std.append(np.std(rtimes))

        if show_msgs:
            print('{: ^10} {: ^10.10} {: ^10.5} {: ^10.5} {: ^10.4} {: ^10.4} {: ^10.4}\n'\
                .format(data_sz[i], 'Average', train_nums[-1], val_nums[-1], accs_val[-1]*100, data_t[-1]*1000, run_t[-1]*1000))

    # create dictionaries to reutrn
    nums = {'train_nums': train_nums, 'train_nums_std': train_nums_std, 'val_nums': val_nums, 'val_nums_std': val_nums_std}
    accs = {'val': accs_val, 'std': accs_std}
    times = {'data_t': data_t, 'data_t_std': data_t_std, 'run_t': run_t, 'run_t_std': run_t_std}

    return nums, accs, times