#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing function

Created on Sun Dec  9 21:30:22 2018

@author: peter
"""
    
use_log_cycle_life = True    
use_all_features = False
which_features = [2,3,4,21,22,24,25,39,40,48,49,63,65]
N_cycles = np.array([20,30,40,50,60,70,80,90,100])

for i in np.arange(len(N_cycles)):
    print('Starting N_cycles = ' + str(int(N_cycles[i])))
    
    
    file_name = "testing/cycles_2TO" + str(int(N_cycles[i])) + "_log.csv"
    features, cycle_lives, feature_names = load_dataset(file_name, False, use_all_features, which_features)
    
    # make some arrays that depend on number of features
    if i == 0:
        norm_coeffs = np.zeros([features.shape[1],len(N_cycles)])

    # Elastic Net CV
    l1_ratio = [0, .1, .5, .7, .9, .95, .99, 1]
    enet = ElasticNetCV(l1_ratio=l1_ratio, cv=5, fit_intercept=True, normalize=True,verbose=False, max_iter=60000,random_state=0)
    # print('Elastic Net CV parameters:')    
    # print(enet.get_params())
    
    if use_log_cycle_life:
        enet.fit(features,np.log10(cycle_lives))
        predicted_cycle_lives = 10**enet.predict(features)
    else:
        enet.fit(features,cycle_lives)
        predicted_cycle_lives = enet.predict(features)

    plt.plot(cycle_lives,predicted_cycle_lives,'o')
    plt.plot([0,1400],[0,1400],'r-')
    plt.ylabel('Predicted cycle lives')
    plt.xlabel('Actual cycle lives')
    #plt.axis('equal')
    plt.axis([0, 1400, 0, 1400])
    plt.show()
    
    residuals = predicted_cycle_lives - cycle_lives
    min_rmse[i] = np.sqrt(((residuals) ** 2).mean())
    # mean_mse = np.mean(enet.mse_path_, axis=2)
    # mean_rmse = np.sqrt(mean_mse)
    #min_rmse[i] = min(mean_rmse[~np.isnan(mean_rmse)])
    optimal_l1_ratio[i] = enet.l1_ratio_
    optimal_alpha[i] = enet.alpha_
            
    print('Min RMSE:')
    print(min_rmse[i])
    print('Optimal alpha:')
    print(enet.alpha_)
    print('Optimal l1_ratio:')
    print(enet.l1_ratio_)
    print('Normalized coefficients:')
    norm_coeffs[:,i] = enet.coef_ * np.std(features,axis=0)
    print(norm_coeffs[:,i])
    print('N iterations to convergence: ' + str(int(enet.n_iter_)))
    print()
    print('Finished N_cycles = ' + str(int(N_cycles[i])))
    print('=======================================')

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