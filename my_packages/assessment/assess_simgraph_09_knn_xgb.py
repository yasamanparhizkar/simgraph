"""
* Branched from assess_simgraph_08_cvxlmnn_factobj1.py

Last Modified: May 19, 2023
"""

from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import dataprocess.data_handler_03 as dh

def get_valset(train_data, val_num, ind_min, ind_max, data_params):
    minus_set = train_data['smpls']
    lbl_func = data_params['lbl_func']
    val_num, val_smpls = dh.update_indices_balanced(val_num, ind_min, ind_max, minus_set, lbl_func(data_params), seed=None)
    val_dess, val_lbls = dh.update_set(val_smpls, data_params)
    val_data   = {'des': val_dess, 'lbls': val_lbls, 'smpls': val_smpls}
    
    return val_num, val_data

# def visualize_M(M, fig_params, idstr, res_path):
#     # unpack params
#     rmark_th = fig_params['rmark_th']
#     xloc = fig_params['xloc']
#     yloc = fig_params['yloc']


#     sg.display_matrix(M, None)
#     # mark prominent elements          
#     lim = (rmark_th/100) * np.max(M) # marker threshold                
#     plt.plot(xloc[M > lim],yloc[M > lim], marker='o', markersize=3, color='r', linestyle='')
#     plt.title('M - marked above {}%'.format(rmark_th))
#     # save figure
#     plt.savefig(res_path+'figures/finalM_'+idstr+'.png')
#     plt.close()
    
def assessment_quantities(val_data, val_num, y_est, val_acc):
    nospk_per = np.sum(val_data['lbls']==-1)/val_num
    min_acc = max(nospk_per, 1-nospk_per)
    if sum(val_data['lbls']==1) == 0:
        missed = 0
    else:
        missed = sum(np.logical_and(val_data['lbls']==1, y_est < 0))/sum(val_data['lbls']==1)

    if sum(val_data['lbls']==-1) == 0:
        false_alarm = 0
    else:
        false_alarm = sum(np.logical_and(val_data['lbls']==-1, y_est > 0))/sum(val_data['lbls']==-1)
        
    assess_qs = {'min_acc': min_acc, 'val_acc': val_acc, 'missed': missed, 'false_alarm': false_alarm}
        
    return assess_qs

def make_line(head, train_num, val_num, res_dict, index):
    line = '{:^10} | {:^10} | {:^10} | {:^10.2f} | {:^10.2f} | {:^17.2f} | {:^17.2f} \n'\
           .format(head, train_num, val_num, \
                   res_dict['min_acc'][index]*100, \
                   res_dict['val_acc'][index]*100, \
                   res_dict['missed'][index]*100, \
                   res_dict['false_alarm'][index]*100)
    return line

# def get_edges_tt_maxdeg_cheng(x, D, seed):
#     """
#     Refer to documentation for 'get_edges_tt'
#     """

#     if seed is not None:
#         np.random.seed(seed)

#     edges = []
#     N = x.shape[0]
#     degree = np.zeros(N)
#     for i in range(N):
#         candids = []
#         for j in range(i+1, N):
#             if degree[j] < D:
#                 candids.append(j)
#         candids = np.array(candids)
        
#         temp = np.array([],dtype=np.int64)
#         comp = D-degree[i]
#         if comp < len(candids):
#             if comp > 0 and any(x[candids]==x[i]):
#                 chosen = candids[x[candids]==x[i]][0]
#                 temp = np.append(temp, chosen)
#                 comp = comp - 1
#                 candids = candids[candids != chosen]
#             if comp > 0 and any(x[candids]!=x[i]):
#                 chosen = candids[x[candids]!=x[i]][0]
#                 temp = np.append(temp, chosen)
#                 comp = comp - 1
#                 candids = candids[candids != chosen]
#             temp = np.append(temp, np.random.choice(candids, size=int(comp), replace=False))
#         else:
#             temp = np.array(candids)

#         temp = [(i,j) for j in temp] 
#         edges = edges + temp
#         degree[i] += len(temp)
#         for (i,j) in temp:
#             degree[j] += 1

#     return edges

def take_train_step(knn, xgb_params, train_comb, train_num, val_num, data_params, res_path_1, res_path_2, seed):
    # create training set
    train_num, _, train_data, _ = dh.random_train_val_balanced(train_num, 1, data_params, seed)

    # fit knn model
    knn = knn.fit(train_data['des'], train_data['lbls'])

    # fit XGboost
    dtrain = xgb.DMatrix(train_data['des'], label=(train_data['lbls']+1)//2)
    evals = [(dtrain, "train")]
    xgb_model = xgb.train(params=xgb_params['params'],dtrain=dtrain,num_boost_round=xgb_params['num_rounds'],evals=evals,verbose_eval=xgb_params['veval'], early_stopping_rounds=xgb_params['early_stop'])

    return train_num, train_data, knn, xgb_model

def take_val_step(train_data, val_num, data_params, knn, xgb_model, seed, show_edges=False):
    # unpack params
    ind_min = data_params['ind_min']
    ind_max = data_params['ind_max']

    # create validation set, NO overlap with the training set
    val_num, val_data = get_valset(train_data, val_num, ind_min, ind_max, data_params)

    # validate knn
    val_acc_1 = knn.score(val_data['des'], val_data['lbls'])
    y_est_1 = knn.predict(val_data['des'])

    # validate xgb
    dval = xgb.DMatrix(val_data['des'], label=(val_data['lbls']+1)//2)
    preds = xgb_model.predict(dval)
    y_est_2 = 2* (preds > 0.5).astype(int) - 1
    val_acc_2 = np.sum(y_est_2 == val_data['lbls'])/val_num

    # compute several assessment quantities
    assess_qs_1 = assessment_quantities(val_data, val_num, y_est_1, val_acc_1)
    assess_qs_2 = assessment_quantities(val_data, val_num, y_est_2, val_acc_2)
    
    return val_num, val_data, assess_qs_1, y_est_1, assess_qs_2, y_est_2

def avg_and_log(next_dict, prev_dict, index, head, train_num, val_num, func, path):
    # compute averages over random combinations of validation sets
    for quantity in prev_dict:
        if func == 'mean':
            next_dict[quantity][index] = np.mean(prev_dict[quantity])
        elif func == 'std':
            next_dict[quantity][index] = np.std(prev_dict[quantity])
        else:
            assert False

    # save on file
    with open(path+'log.txt', 'a') as file:
        line = make_line(head, train_num, val_num, next_dict, index)
        file.write(line)
        
    return next_dict


def assess_sg_model(knn, xgb_params, data_params, rnd_params, res_path_1, res_path_2):
    time0 = time.time()

    # unpack params
    train_sizes = rnd_params['train_sizes']
    val_sizes = rnd_params['val_sizes']
    train_its = rnd_params['train_its']
    val_its = rnd_params['val_its']
    seed = rnd_params['seed'] if 'seed' in rnd_params else None

    ################################################################################
    # set up the result directories for the 1st approach
    if not os.path.exists(res_path_1):
        os.mkdir(res_path_1)
        os.mkdir(res_path_1+'curves/')
        os.mkdir(res_path_1+'matrices/')
        os.mkdir(res_path_1+'figures/')
    else:
        if not os.path.exists(res_path_1+'curves/'):
            os.mkdir(res_path_1+'curves/')
        if not os.path.exists(res_path_1+'matrices/'):
            os.mkdir(res_path_1+'matrices/')
        if not os.path.exists(res_path_1+'figures/'):
            os.mkdir(res_path_1+'figures/')

    # set up the log files for the 1st approach
    with open(res_path_1+'log.txt', 'w') as file:    
        arr = ('{:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^17} | {:^17} \n'\
               .format('i', 'train_num', 'val_num', 'min_acc(%)', 'val_acc(%)',\
                       'missed spks(%)', 'false alarms(%)'),'-'*101+'\n')
        file.writelines(arr)

    for i in range(len(val_sizes)):
        file = open(res_path_1+'curves/train_'+str(i)+'.txt', 'w')
        file.close()
        file = open(res_path_1+'curves/runtime_'+str(i)+'.txt', 'w')
        file.close()
    file = open(res_path_1+'curves/val.txt', 'w')
    file.close()

    # create dictionaries to keep the desired assessment quantities for the 1st approach
    assess_qs_1 = {'min_acc': 0, 'val_acc': 0, 'missed': 0, 'false_alarm': 0}
    val_comb_res_1 = {}
    train_comb_res_1 = {}
    train_num_res_1 = {}
    train_num_err_1 = {}
    val_num_res_1 = {}
    val_num_err_1 = {}
    for quantity in assess_qs_1:
        val_comb_res_1[quantity] = np.zeros(val_its)
        train_comb_res_1[quantity] = np.zeros(train_its)
        train_num_res_1[quantity] = np.zeros(len(train_sizes))
        train_num_err_1[quantity] = np.zeros(len(train_sizes))
        val_num_res_1[quantity] = np.zeros(len(val_sizes))
        val_num_err_1[quantity] = np.zeros(len(val_sizes))

    ##########################################################################
    # set up the result directories for the 2nd approach
    if not os.path.exists(res_path_2):
        os.mkdir(res_path_2)
        os.mkdir(res_path_2+'curves/')
        os.mkdir(res_path_2+'matrices/')
        os.mkdir(res_path_2+'figures/')
    else:
        if not os.path.exists(res_path_2+'curves/'):
            os.mkdir(res_path_2+'curves/')
        if not os.path.exists(res_path_2+'matrices/'):
            os.mkdir(res_path_2+'matrices/')
        if not os.path.exists(res_path_2+'figures/'):
            os.mkdir(res_path_2+'figures/')

    # set up the log files for the 2nd approach
    with open(res_path_2+'log.txt', 'w') as file:    
        arr = ('{:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^17} | {:^17} \n'\
               .format('i', 'train_num', 'val_num', 'min_acc(%)', 'val_acc(%)',\
                       'missed spks(%)', 'false alarms(%)'),'-'*101+'\n')
        file.writelines(arr)

    for i in range(len(val_sizes)):
        file = open(res_path_2+'curves/train_'+str(i)+'.txt', 'w')
        file.close()
        file = open(res_path_2+'curves/runtime_'+str(i)+'.txt', 'w')
        file.close()
    file = open(res_path_2+'curves/val.txt', 'w')
    file.close()

    # create dictionaries to keep the desired assessment quantities for the 2nd approach
    assess_qs_2 = {'min_acc': 0, 'val_acc': 0, 'missed': 0, 'false_alarm': 0}
    val_comb_res_2 = {}
    train_comb_res_2 = {}
    train_num_res_2 = {}
    train_num_err_2 = {}
    val_num_res_2 = {}
    val_num_err_2 = {}
    for quantity in assess_qs_2:
        val_comb_res_2[quantity] = np.zeros(val_its)
        train_comb_res_2[quantity] = np.zeros(train_its)
        train_num_res_2[quantity] = np.zeros(len(train_sizes))
        train_num_err_2[quantity] = np.zeros(len(train_sizes))
        val_num_res_2[quantity] = np.zeros(len(val_sizes))
        val_num_err_2[quantity] = np.zeros(len(val_sizes))

    ###########################################################################

    i = 0
    for val_num in val_sizes:
        j = 0
        for train_num in train_sizes:
            train_comb_time = []
            for train_comb in range(train_its):
                time3 = time.time()
                # train
                train_num, train_data, knn, xgb_model = \
                take_train_step(knn, xgb_params, train_comb, train_num, val_num, data_params, res_path_1, res_path_2, seed)

                for val_comb in range(val_its):
                    # validate
                    val_num, val_data, assess_qs_1, y_est_1, assess_qs_2, y_est_2 = \
                    take_val_step(train_data, val_num, data_params, knn, xgb_model, seed)
                    # log resutls
                    val_comb_res_1 = avg_and_log(val_comb_res_1, assess_qs_1, val_comb, str(val_comb), train_num, val_num, 'mean', res_path_1)
                    val_comb_res_2 = avg_and_log(val_comb_res_2, assess_qs_2, val_comb, str(val_comb), train_num, val_num, 'mean', res_path_2)

                # average over various validation set combinations and log
                train_comb_res_1 = avg_and_log(train_comb_res_1, val_comb_res_1, train_comb, '>'+str(train_comb), train_num, val_num, 'mean', res_path_1)
                train_comb_res_2 = avg_and_log(train_comb_res_2, val_comb_res_2, train_comb, '>'+str(train_comb), train_num, val_num, 'mean', res_path_2)
                # measure runtime
                train_comb_time.append(time.time() - time3)

            # average over various training and validation set combinations and log
            train_num_res_1 = avg_and_log(train_num_res_1, train_comb_res_1, j, '*t*', train_num, val_num, 'mean', res_path_1)
            train_num_err_1 = avg_and_log(train_num_err_1, train_comb_res_1, j, '*te*', train_num, val_num, 'std', res_path_1)
            train_num_res_2 = avg_and_log(train_num_res_2, train_comb_res_2, j, '*t*', train_num, val_num, 'mean', res_path_2)
            train_num_err_2 = avg_and_log(train_num_err_2, train_comb_res_2, j, '*te*', train_num, val_num, 'std', res_path_2)
            # save the curves in a separate file
            with open(res_path_1+'curves/train_'+str(i)+'.txt', 'a') as file:
                for quantity in assess_qs_1:
                    file.write(str(train_num_res_1[quantity][j])+'\n')
                    file.write(str(train_num_err_1[quantity][j])+'\n')
                file.write('\n')
            with open(res_path_2+'curves/train_'+str(i)+'.txt', 'a') as file:
                for quantity in assess_qs_2:
                    file.write(str(train_num_res_2[quantity][j])+'\n')
                    file.write(str(train_num_err_2[quantity][j])+'\n')
                file.write('\n')
            # save this iteration's runtime
            with open(res_path_1+'curves/runtime_'+str(i)+'.txt', 'a') as file:
                file.write(str(np.mean(train_comb_time))+'\n')
                file.write(str(np.std(train_comb_time))+'\n')
                file.write('\n')
            j += 1

        # average over various training set sizes and training and validation set combinations, and log
        val_num_res_1 = avg_and_log(val_num_res_1, train_num_res_1, i, '**v**', train_num, val_num, 'mean', res_path_1)
        val_num_err_1 = avg_and_log(val_num_err_1, train_num_res_1, i, '**ve**', train_num, val_num, 'std', res_path_1)
        val_num_res_2 = avg_and_log(val_num_res_2, train_num_res_2, i, '**v**', train_num, val_num, 'mean', res_path_2)
        val_num_err_2 = avg_and_log(val_num_err_2, train_num_res_2, i, '**ve**', train_num, val_num, 'std', res_path_2)
        # save train_num_res curves for this specific val_num
        with open(res_path_1+'curves/val.txt', 'a') as file:
            for quantity in assess_qs_1:
                file.write(str(val_num_res_1[quantity][i])+'\n')
                file.write(str(val_num_err_1[quantity][i])+'\n')
            file.write('\n')
        with open(res_path_2+'curves/val.txt', 'a') as file:
            for quantity in assess_qs_2:
                file.write(str(val_num_res_2[quantity][i])+'\n')
                file.write(str(val_num_err_2[quantity][i])+'\n')
            file.write('\n')
        i += 1

    # eng.quit()
    print('Done. Elapsed time: {:.3f} sec'.format(time.time()-time0))
        
    return val_num_res_1, val_num_err_1, val_num_res_2, val_num_err_2

def plot_curves(rnd_params, title, res_path):
    # unpack params
    train_sizes = rnd_params['train_sizes']
    val_sizes = rnd_params['val_sizes']
    train_its = rnd_params['train_its']
    val_its = rnd_params['val_its']
    assess_qs = ['min_acc', 'val_acc', 'missed', 'false_alarm']

    # read training curves
    curves = {}
    errors = {}
    for i in range(len(val_sizes)):
        curves_i = np.loadtxt(res_path+'curves/train_'+str(i)+'.txt')
        j = 0
        for quantity in assess_qs:
            if i==0:
                curves[quantity] = curves_i[j::8].reshape((1, -1))
                errors[quantity] = curves_i[j+1::8].reshape((1, -1))
            else:
                curves[quantity] = np.concatenate((curves[quantity], [curves_i[j::8]]), axis=0)
                errors[quantity] = np.concatenate((errors[quantity], [curves_i[j+1::8]]), axis=0)
            j += 2
    
    # plot training curves
    plt.figure()
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.8,hspace=0.8)
    for i in range(len(val_sizes)):
        plt.subplot(len(val_sizes), 1, i+1)
        for quantity in curves:
            plt.errorbar(train_sizes, curves[quantity][i], errors[quantity][i])
        plt.legend(curves.keys())
        plt.xlabel('training set size')
        plt.ylabel('{} val repeats x {} train repeats'.format(val_its, train_its))
        _ = plt.title(title)
    plt.savefig(res_path+'train_curves.png')
    plt.close()

    # read validation curves
    curves_i = np.loadtxt(res_path+'curves/val.txt')
    j = 0
    for quantity in assess_qs:
        curves[quantity] = curves_i[j::8]
        errors[quantity] = curves_i[j+1::8]
        j += 2
    
    # plot validation curves
    plt.figure()
    for quantity in assess_qs:
        plt.errorbar(val_sizes, curves[quantity], errors[quantity])
    plt.legend(curves.keys())
    plt.xlabel('validation set size')
    plt.ylabel('{} val repeats x {} train repeats x {} train set sizes'.format(val_its, train_its, len(train_sizes)))
    _ = plt.title(title)
    plt.savefig(res_path+'val_curves.png')
    plt.close()

    # read runtime curves
    for i in range(len(val_sizes)):
        curves_i = np.loadtxt(res_path+'curves/runtime_'+str(i)+'.txt')
        if i==0:
            curves = curves_i[::2].reshape((1, -1))
            errors = curves_i[1::2].reshape((1, -1))
        else:
            curves = np.concatenate((curves, [curves_i[::2]]), axis=0)
            errors = np.concatenate((errors, [curves_i[1::2]]), axis=0)

    # plot runtime curves
    plt.figure()
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.8,hspace=0.8)
    for i in range(len(val_sizes)):
        plt.subplot(len(val_sizes), 1, i+1)
        plt.errorbar(train_sizes, curves[i], errors[i])
        plt.legend('runtime (sec)')
        plt.xlabel('training set size')
        plt.ylabel('{} val repeats x {} train repeats'.format(val_its, train_its))
        _ = plt.title(title)
    plt.savefig(res_path+'runtime_curves.png')
    plt.close()

def plot_curves_without_runtime(rnd_params, title, res_path):
    # unpack params
    train_sizes = rnd_params['train_sizes']
    val_sizes = rnd_params['val_sizes']
    train_its = rnd_params['train_its']
    val_its = rnd_params['val_its']
    assess_qs = ['min_acc', 'val_acc', 'missed', 'false_alarm']

    # read training curves
    curves = {}
    errors = {}
    for i in range(len(val_sizes)):
        curves_i = np.loadtxt(res_path+'curves/train_'+str(i)+'.txt')
        j = 0
        for quantity in assess_qs:
            if i==0:
                curves[quantity] = curves_i[j::8].reshape((1, -1))
                errors[quantity] = curves_i[j+1::8].reshape((1, -1))
            else:
                curves[quantity] = np.concatenate((curves[quantity], [curves_i[j::8]]), axis=0)
                errors[quantity] = np.concatenate((errors[quantity], [curves_i[j+1::8]]), axis=0)
            j += 2
    
    # plot training curves
    plt.figure()
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.8,hspace=0.8)
    for i in range(len(val_sizes)):
        plt.subplot(len(val_sizes), 1, i+1)
        for quantity in curves:
            plt.errorbar(train_sizes, curves[quantity][i], errors[quantity][i])
        plt.legend(curves.keys())
        plt.xlabel('training set size')
        plt.ylabel('{} val repeats x {} train repeats'.format(val_its, train_its))
        _ = plt.title(title)
    plt.savefig(res_path+'train_curves.png')
    plt.close()

    # read validation curves
    curves_i = np.loadtxt(res_path+'curves/val.txt')
    j = 0
    for quantity in assess_qs:
        curves[quantity] = curves_i[j::8]
        errors[quantity] = curves_i[j+1::8]
        j += 2
    
    # plot validation curves
    plt.figure()
    for quantity in assess_qs:
        plt.errorbar(val_sizes, curves[quantity], errors[quantity])
    plt.legend(curves.keys())
    plt.xlabel('validation set size')
    plt.ylabel('{} val repeats x {} train repeats x {} train set sizes'.format(val_its, train_its, len(train_sizes)))
    _ = plt.title(title)
    plt.savefig(res_path+'val_curves.png')
    plt.close()
