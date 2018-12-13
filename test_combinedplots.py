#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing function

Created on Sun Dec  9 21:30:22 2018

@author: peter
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

doPlot = False
use_all_features = False

def main():
    # prelims
    plt.close("all")
    matplotlib.rcParams.update({'font.size': 18})
    
    models = ['enet','RF','Adaboost']
    use_log_cycle_life = [1,0,0]
    
    f1, ax_1 = plt.subplots(1, 3, sharey=True)
    f2, ax_2 = plt.subplots(1, 3, sharey=True)
    
    for model_idx, MODEL in enumerate(models):
        N_cycles = np.array([20,30,40,50,60,70,80,90,100])
        rmse = np.zeros(N_cycles.shape)
        mpe = np.zeros(N_cycles.shape)
        which_features = [2,3,4,21,22,24,25,39,40,48,49,63,65]
        
        if MODEL == 'enet': # versioning issue with rf & ab
            # load all models
            models = pickle.load(open('results/'+MODEL+"_trained_models.pkl", "rb" ))
            
            # loop through enets and make predictions
            for i in np.arange(len(N_cycles)):
                print('Starting N_cycles = ' + str(int(N_cycles[i])))
                
                file_name = "testing/cycles_2TO" + str(int(N_cycles[i])) + "_log.csv"
                features, cycle_lives, feature_names = load_dataset(file_name, False, use_all_features, which_features)
                
                file_name = "training/cycles_2TO" + str(int(N_cycles[i])) + "_log.csv"
                train_features, train_cycle_lives, feature_names = load_dataset(file_name, False, use_all_features, which_features)
                
                
                # set model
                m = models[i]
                
                # predictions
                if use_log_cycle_life:
                    predicted_cycle_lives = 10**m.predict(features)
                    train_predicted_cycle_lives = 10**m.predict(train_features)
                else:
                    predicted_cycle_lives = m.predict(features)
                    train_predicted_cycle_lives = m.predict(train_features)
                
                print(predicted_cycle_lives)
                print(train_predicted_cycle_lives)
                
                if doPlot:
                    plt.figure()
                    plt.plot(cycle_lives,predicted_cycle_lives,'o')
                    plt.plot([0,1400],[0,1400],'r-')
                    plt.ylabel('Predicted cycle life')
                    plt.xlabel('Observed cycle life')
                    plt.title('Cycle number '+str(N_cycles[i]))
                    #plt.axis('equal')
                    plt.axis([0, 1400, 0, 1400])
                
                residuals = predicted_cycle_lives - cycle_lives
                rmse[i] = np.sqrt(((residuals) ** 2).mean())
                mpe[i] = np.mean(np.abs(residuals)/cycle_lives*100)
                
                print('RMSE (cycles):')
                print(rmse[i])
                print('MPE (%):')
                print(mpe[i])
                print('=======================================')
                
            
            train_mpe = pickle.load(open('results/'+MODEL+"_training_percenterror.pkl", "rb" ))
            if MODEL != 'enet':
                cv_mpe = pickle.load(open('results/'+MODEL+"_crossvalid_percenterror.pkl", "rb" ))
            else:
                cv_mpe = train_mpe # swap
                
        else:
            # random forest only
            data = pickle.load(open('results/'+MODEL+"_data.pkl", "rb" ))
            predicted_cycle_lives = data[0]
            train_predicted_cycle_lives = data[1]
            train_mpe = data[2]
            cv_mpe = data[3]
            mpe = data[4]
            
            i=8
            
            file_name = "testing/cycles_2TO" + str(int(N_cycles[i])) + "_log.csv"
            features, cycle_lives, feature_names = load_dataset(file_name, False, use_all_features, which_features)
            
            file_name = "training/cycles_2TO" + str(int(N_cycles[i])) + "_log.csv"
            train_features, train_cycle_lives, feature_names = load_dataset(file_name, False, use_all_features, which_features)
    
    
        # plot error vs cycle number
        # mpe
        plt.figure(1)
        plt.sca(ax_1[model_idx])
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if MODEL != 'enet':
            plt.plot(N_cycles, train_mpe, '-o',label='Train',color=colors[2])
        plt.plot(N_cycles, cv_mpe, '-o',label='CV',color=colors[0])
        plt.plot(N_cycles, mpe, '-o',label='Test',color=colors[1])
        plt.ylabel('Mean percent error (%)')
        plt.xlabel('Cycle number')
        plt.legend()
        plt.ylim([0,36])

#        # diagonal line plot
#        plt.figure(2)
#        plt.sca(ax_2[model_idx])
#        plt.plot(train_cycle_lives,train_predicted_cycle_lives,'rs',label='Train')
#        plt.plot(cycle_lives,predicted_cycle_lives,'bo',label='Test')
#        plt.plot([0,2400],[0,2400],'k--')
#        plt.ylabel('Predicted cycle life')
#        plt.xlabel('Observed cycle life')
#        #plt.title('Cycle '+str(N_cycles[i]))
#        plt.axes().set_aspect('equal', 'box')
#        plt.xticks(np.arange(0, 2501, step=500))
#        plt.yticks(np.arange(0, 2501, step=500))
#        plt.axis([0, 2501, 0, 2501])
#        plt.legend()
#            
    f1.tight_layout()
    #plt.savefig('./plots/'+MODEL+'_error.png',bbox_inches='tight')
    #plt.savefig('./plots/'+MODEL+'_error.svg',bbox_inches='tight')
    
    plt.tight_layout()
    #plt.savefig('./plots/'+MODEL+'_obs_vs_pred.png',bbox_inches='tight')
    #plt.savefig('./plots/'+MODEL+'_obs_vs_pred.svg',bbox_inches='tight')
    
    

def load_dataset(csv_path, add_intercept=True, use_all_features=True, which_features=[2]):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (features).
        ys: Numpy array of y-values (labels).
        headers: list of headers
    """
    
    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    # x_cols = [i for i in range(len(headers)) if headers[i] == 'cycle_lives']
    # l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    if use_all_features:
        features = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=range(2, len(headers)))
    else:
        features = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=which_features)
    cycle_lives = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=[1])
    feature_names = headers[2:len(headers)]

    m = features.shape[0]
    if add_intercept:
        features = np.concatenate((np.ones([m, 1]), features),axis=1)
        feature_names = ['intercept'] + feature_names

    return features, cycle_lives, feature_names


if __name__ == "__main__":
    main()