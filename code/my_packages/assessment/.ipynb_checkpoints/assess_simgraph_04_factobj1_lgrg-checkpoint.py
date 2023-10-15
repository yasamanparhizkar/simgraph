"""
* Branched from assess_simgraph_08_cvxlmnn_factobj1.py 
* Use factorization (M=B.T@B) and gradient descent to minimize the objective#1 (GLR+trace)
* Compare the above method with logistic from the scikit-learn package
* Save log and draw comparison plots

Last Modified: April 27, 2023
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import simgraph.my_simgraph_06 as sg
import dataprocess.data_handler_03 as dh

def get_valset(train_data, val_num, ind_min, ind_max, data_params):
    minus_set = train_data['smpls']
    lbl_func = data_params['lbl_func']
    val_num, val_smpls = dh.update_indices_balanced(val_num, ind_min, ind_max, minus_set, lbl_func(data_params), seed=None)
    val_dess, val_lbls = dh.update_set(val_smpls, data_params)
    val_data   = {'des': val_dess, 'lbls': val_lbls, 'smpls': val_smpls}
    
    return val_num, val_data

def visualize_M(M, fig_params, train_comb, train_num, val_num, res_path):
    # unpack params
    rmark_th = fig_params['rmark_th']
    xloc = fig_params['xloc']
    yloc = fig_params['yloc']


    sg.display_matrix(M, None)
    # mark prominent elements          
    lim = (rmark_th/100) * np.max(M) # marker threshold                
    plt.plot(xloc[M > lim],yloc[M > lim], marker='o', markersize=3, color='r', linestyle='')
    plt.title('M - marked above {}%'.format(rmark_th))
    # save figure
    plt.savefig(res_path+'figures/finalM_'+str(val_num)+'_'+str(train_num)+'_'+str(train_comb)+'.png')
    plt.close()
    
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

def take_train_step(lgrg, train_comb, train_num, val_num, data_params, sg_params, sg_opt_params, fig_params, res_path_sg, res_path_lgrg, seed):
    # create training set
    # time0 = time.time()
    train_num, _, train_data, _ = dh.random_train_val_balanced(train_num, 1, data_params, seed)
    # train_data['des'] = dh.normalize(train_data['des'])
    # print('dh.random_train_val_balanced took {} sec'.format(time.time()-time0))

    # update sg_params
    sg_params['train_t'] = train_data['smpls']

    # train the model   
    # time0 = time.time()
    sg_opt_params['Theta0'] = None
    sg_params['edges_tt'] = None
    B, sg_stats = sg.fit_graph(train_data['des'], train_data['lbls'], sg_params, sg_opt_params, seed)
    # print('sg.fit_graph took {} sec'.format(time.time()-time0))

    # train the logistic regression model for comparison
    lgrg = lgrg.fit(train_data['des'], train_data['lbls'])

    # visualize and save learned M
    # time0 = time.time()
    M = B.T @ B
    visualize_M(M, fig_params, train_comb, train_num, val_num, res_path_sg)
    np.save(res_path_sg+'matrices/finalM_'+str(val_num)+'_'+str(train_num)+'_'+str(train_comb), M)
    # print('visualizing and saving M took {} sec'.format(time.time()-time0))
    
    return train_num, train_data, M, lgrg

def take_val_step(lgrg, train_data, val_num, data_params, M, sg_params, seed, show_edges=False):
    # unpack params
    ind_min = data_params['ind_min']
    ind_max = data_params['ind_max']

    # create validation set, NO overlap with the training set
    # time0 = time.time()
    val_num, val_data = get_valset(train_data, val_num, ind_min, ind_max, data_params)
    # val_data['des'] = dh.normalize(val_data['des'])
    # print('get_valset took {} sec'.format(time.time()-time0))

    # update sg_params
    sg_params['val_t'] = val_data['smpls']

    # validate the model
    # time0 = time.time()
    sg_params['edges_vv'] = None
    sg_params['edges_vt'] = None
    val_acc_sg, y_est_sg, t = sg.get_acc(M, train_data['des'], train_data['lbls'], val_data['des'], val_data['lbls'], sg_params, seed, show_edges)
    # print('sg.get_acc took {} sec'.format(time.time()-time0))

    # validate the logistic regression model for comparison
    # validate the model
    y_est_lgrg = lgrg.predict(val_data['des'])
    val_acc_lgrg = lgrg.score(val_data['des'], val_data['lbls'])

    # compute several assessment quantities
    # time0 = time.time()
    assess_qs_sg = assessment_quantities(val_data, val_num, y_est_sg, val_acc_sg)
    assess_qs_lgrg = assessment_quantities(val_data, val_num, y_est_lgrg, val_acc_lgrg)
    # print('assessment_quantities took {} sec'.format(time.time()-time0))
    
    return val_num, val_data, assess_qs_sg, y_est_sg, assess_qs_lgrg, y_est_lgrg

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


def assess_sg_model(data_params, sg_params, sg_opt_params, rnd_params, fig_params, res_path_sg, res_path_lgrg):
    time0 = time.time()

    # unpack params
    train_sizes = rnd_params['train_sizes']
    val_sizes = rnd_params['val_sizes']
    train_its = rnd_params['train_its']
    val_its = rnd_params['val_its']
    seed = rnd_params['seed'] if 'seed' in rnd_params else None

    ################################################################################
    # set up the result directories for sg
    if not os.path.exists(res_path_sg):
        os.mkdir(res_path_sg)
        os.mkdir(res_path_sg+'curves/')
        os.mkdir(res_path_sg+'matrices/')
        os.mkdir(res_path_sg+'figures/')
    else:
        if not os.path.exists(res_path_sg+'curves/'):
            os.mkdir(res_path_sg+'curves/')
        if not os.path.exists(res_path_sg+'matrices/'):
            os.mkdir(res_path_sg+'matrices/')
        if not os.path.exists(res_path_sg+'figures/'):
            os.mkdir(res_path_sg+'figures/')

    # set up the log files for sg
    with open(res_path_sg+'log.txt', 'w') as file:    
        arr = ('{:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^17} | {:^17} \n'\
               .format('i', 'train_num', 'val_num', 'min_acc(%)', 'val_acc(%)',\
                       'missed spks(%)', 'false alarms(%)'),'-'*101+'\n')
        file.writelines(arr)

    for i in range(len(val_sizes)):
        file = open(res_path_sg+'curves/train_'+str(i)+'.txt', 'w')
        file.close()
        file = open(res_path_sg+'curves/runtime_'+str(i)+'.txt', 'w')
        file.close()
    file = open(res_path_sg+'curves/val.txt', 'w')
    file.close()

    # create dictionaries to keep the desired assessment quantities for sg
    assess_qs_sg = {'min_acc': 0, 'val_acc': 0, 'missed': 0, 'false_alarm': 0}
    val_comb_res_sg = {}
    train_comb_res_sg = {}
    train_num_res_sg = {}
    train_num_err_sg = {}
    val_num_res_sg = {}
    val_num_err_sg = {}
    for quantity in assess_qs_sg:
        val_comb_res_sg[quantity] = np.zeros(val_its)
        train_comb_res_sg[quantity] = np.zeros(train_its)
        train_num_res_sg[quantity] = np.zeros(len(train_sizes))
        train_num_err_sg[quantity] = np.zeros(len(train_sizes))
        val_num_res_sg[quantity] = np.zeros(len(val_sizes))
        val_num_err_sg[quantity] = np.zeros(len(val_sizes))

    ##########################################################################
    # set up the result directories for logistic regression
    if not os.path.exists(res_path_lgrg):
        os.mkdir(res_path_lgrg)
        os.mkdir(res_path_lgrg+'curves/')
        os.mkdir(res_path_lgrg+'matrices/')
        os.mkdir(res_path_lgrg+'figures/')
    else:
        if not os.path.exists(res_path_lgrg+'curves/'):
            os.mkdir(res_path_lgrg+'curves/')
        if not os.path.exists(res_path_lgrg+'matrices/'):
            os.mkdir(res_path_lgrg+'matrices/')
        if not os.path.exists(res_path_lgrg+'figures/'):
            os.mkdir(res_path_lgrg+'figures/')

    # set up the log files for logistic regression
    with open(res_path_lgrg+'log.txt', 'w') as file:    
        arr = ('{:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^17} | {:^17} \n'\
               .format('i', 'train_num', 'val_num', 'min_acc(%)', 'val_acc(%)',\
                       'missed spks(%)', 'false alarms(%)'),'-'*101+'\n')
        file.writelines(arr)

    for i in range(len(val_sizes)):
        file = open(res_path_lgrg+'curves/train_'+str(i)+'.txt', 'w')
        file.close()
        file = open(res_path_lgrg+'curves/runtime_'+str(i)+'.txt', 'w')
        file.close()
    file = open(res_path_lgrg+'curves/val.txt', 'w')
    file.close()

    # create dictionaries to keep the desired assessment quantities for logistic regression
    assess_qs_lgrg = {'min_acc': 0, 'val_acc': 0, 'missed': 0, 'false_alarm': 0}
    val_comb_res_lgrg = {}
    train_comb_res_lgrg = {}
    train_num_res_lgrg = {}
    train_num_err_lgrg = {}
    val_num_res_lgrg = {}
    val_num_err_lgrg = {}
    for quantity in assess_qs_lgrg:
        val_comb_res_lgrg[quantity] = np.zeros(val_its)
        train_comb_res_lgrg[quantity] = np.zeros(train_its)
        train_num_res_lgrg[quantity] = np.zeros(len(train_sizes))
        train_num_err_lgrg[quantity] = np.zeros(len(train_sizes))
        val_num_res_lgrg[quantity] = np.zeros(len(val_sizes))
        val_num_err_lgrg[quantity] = np.zeros(len(val_sizes))

    ###########################################################################
    # instantiate the logistic regression model
    lgrg = LogisticRegression(penalty='none', max_iter=1000)

    i = 0
    for val_num in val_sizes:
        j = 0
        for train_num in train_sizes:
            train_comb_time = []
            for train_comb in range(train_its):
                time3 = time.time()
                # train
                train_num, train_data, M, lgrg = \
                take_train_step(lgrg, train_comb, train_num, val_num, data_params, sg_params, sg_opt_params, fig_params, res_path_sg, res_path_lgrg, seed)

                for val_comb in range(val_its):
                    # validate
                    val_num, val_data, assess_qs_sg, y_est_sg, assess_qs_lgrg, y_est_lgrg = \
                    take_val_step(lgrg, train_data, val_num, data_params, M, sg_params, seed)
                    # log resutls
                    val_comb_res_sg = avg_and_log(val_comb_res_sg, assess_qs_sg, val_comb, str(val_comb), train_num, val_num, 'mean', res_path_sg)
                    val_comb_res_lgrg = avg_and_log(val_comb_res_lgrg, assess_qs_lgrg, val_comb, str(val_comb), train_num, val_num, 'mean', res_path_lgrg)

                # average over various validation set combinations and log
                train_comb_res_sg = avg_and_log(train_comb_res_sg, val_comb_res_sg, train_comb, '>'+str(train_comb), train_num, val_num, 'mean', res_path_sg)
                train_comb_res_lgrg = avg_and_log(train_comb_res_lgrg, val_comb_res_lgrg, train_comb, '>'+str(train_comb), train_num, val_num, 'mean', res_path_lgrg)
                # measure runtime
                train_comb_time.append(time.time() - time3)

            # average over various training and validation set combinations and log
            train_num_res_sg = avg_and_log(train_num_res_sg, train_comb_res_sg, j, '*t*', train_num, val_num, 'mean', res_path_sg)
            train_num_err_sg = avg_and_log(train_num_err_sg, train_comb_res_sg, j, '*te*', train_num, val_num, 'std', res_path_sg)
            train_num_res_lgrg = avg_and_log(train_num_res_lgrg, train_comb_res_lgrg, j, '*t*', train_num, val_num, 'mean', res_path_lgrg)
            train_num_err_lgrg = avg_and_log(train_num_err_lgrg, train_comb_res_lgrg, j, '*te*', train_num, val_num, 'std', res_path_lgrg)
            # save the curves in a separate file
            with open(res_path_sg+'curves/train_'+str(i)+'.txt', 'a') as file:
                for quantity in assess_qs_sg:
                    file.write(str(train_num_res_sg[quantity][j])+'\n')
                    file.write(str(train_num_err_sg[quantity][j])+'\n')
                file.write('\n')
            with open(res_path_lgrg+'curves/train_'+str(i)+'.txt', 'a') as file:
                for quantity in assess_qs_lgrg:
                    file.write(str(train_num_res_lgrg[quantity][j])+'\n')
                    file.write(str(train_num_err_lgrg[quantity][j])+'\n')
                file.write('\n')
            # save this iteration's runtime
            with open(res_path_sg+'curves/runtime_'+str(i)+'.txt', 'a') as file:
                file.write(str(np.mean(train_comb_time))+'\n')
                file.write(str(np.std(train_comb_time))+'\n')
                file.write('\n')
            j += 1

        # average over various training set sizes and training and validation set combinations, and log
        val_num_res_sg = avg_and_log(val_num_res_sg, train_num_res_sg, i, '**v**', train_num, val_num, 'mean', res_path_sg)
        val_num_err_sg = avg_and_log(val_num_err_sg, train_num_res_sg, i, '**ve**', train_num, val_num, 'std', res_path_sg)
        val_num_res_lgrg = avg_and_log(val_num_res_lgrg, train_num_res_lgrg, i, '**v**', train_num, val_num, 'mean', res_path_lgrg)
        val_num_err_lgrg = avg_and_log(val_num_err_lgrg, train_num_res_lgrg, i, '**ve**', train_num, val_num, 'std', res_path_lgrg)
        # save train_num_res curves for this specific val_num
        with open(res_path_sg+'curves/val.txt', 'a') as file:
            for quantity in assess_qs_sg:
                file.write(str(val_num_res_sg[quantity][i])+'\n')
                file.write(str(val_num_err_sg[quantity][i])+'\n')
            file.write('\n')
        with open(res_path_lgrg+'curves/val.txt', 'a') as file:
            for quantity in assess_qs_lgrg:
                file.write(str(val_num_res_lgrg[quantity][i])+'\n')
                file.write(str(val_num_err_lgrg[quantity][i])+'\n')
            file.write('\n')
        i += 1

    # eng.quit()
    print('Done. Elapsed time: {:.3f} sec'.format(time.time()-time0))
        
    return val_num_res_sg, val_num_err_sg, val_num_res_lgrg, val_num_err_lgrg

def plot_curves(rnd_params, sg_params, sg_opt_params, feature_id, res_path):
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
        _ = plt.title('{}, Dt = {}, Dvt = {}, Dv = {}, $\mu$ = {}, num_its = {}'.format(feature_id, sg_params['Dt'], sg_params['Dvt'], sg_params['Dv'], sg_params['mu'], sg_opt_params['num_its']))
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
    _ = plt.title('{}, Dt = {}, Dvt = {}, Dv = {}, $\mu$ = {}, num_its = {}'.format(feature_id, sg_params['Dt'], sg_params['Dvt'], sg_params['Dv'], sg_params['mu'], sg_opt_params['num_its']))
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
        _ = plt.title('{}, Dt = {}, Dvt = {}, Dv = {}, $\mu$ = {}, num_its = {}'.format(feature_id, sg_params['Dt'], sg_params['Dvt'], sg_params['Dv'], sg_params['mu'], sg_opt_params['num_its']))
    plt.savefig(res_path+'runtime_curves.png')
    plt.close()

def plot_curves_without_runtime(rnd_params, sg_params, sg_opt_params, feature_id, res_path):
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
        _ = plt.title('{}, Dt = {}, Dvt = {}, Dv = {}, $\mu$ = {}, num_its = {}'.format(feature_id, sg_params['Dt'], sg_params['Dvt'], sg_params['Dv'], sg_params['mu'], sg_opt_params['num_its']))
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
    _ = plt.title('{}, Dt = {}, Dvt = {}, Dv = {}, $\mu$ = {}, num_its = {}'.format(feature_id, sg_params['Dt'], sg_params['Dvt'], sg_params['Dv'], sg_params['mu'], sg_opt_params['num_its']))
    plt.savefig(res_path+'val_curves.png')
    plt.close()