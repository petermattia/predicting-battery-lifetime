import numpy as np
import matplotlib.pyplot as plt
import math
#import sklearn as sk
from sklearn.linear_model import ElasticNetCV
import pandas as pd
import pickle

def main():
    N_cycles = np.array([20,30,40,50,60,70,80,90,100])
    #N_cycles = np.array([40])

    
    min_rmse = np.zeros(N_cycles.shape)
    min_percent_error = np.zeros(N_cycles.shape)
    optimal_l1_ratio = np.zeros(N_cycles.shape)
    optimal_alpha = np.zeros(N_cycles.shape)
    use_log_cycle_life = True    
    use_all_features = False
    which_features = [2,3,4,21,22,24,25,39,40,48,49,63,65]#list(map(int, n
    
    trained_models = []    
    
    for i in np.arange(len(N_cycles)):
        print('Starting N_cycles = ' + str(int(N_cycles[i])))
        
        
        file_name = "training/cycles_2TO" + str(int(N_cycles[i])) + "_log.csv"
        features, cycle_lives, feature_names = load_dataset(file_name, False, use_all_features, which_features)
        
        # make some arrays that depend on number of features
        if i == 0:
            norm_coeffs = np.zeros([features.shape[1],len(N_cycles)])

    
        # Elastic Net CV
        l1_ratio = [0.01, .1, .5, .7, .9, .95, .99, 1]
        enet = ElasticNetCV(l1_ratio=l1_ratio, cv=5, fit_intercept=True, normalize=True,verbose=False, max_iter=60000,random_state=0)
        # print('Elastic Net CV parameters:')    
        # print(enet.get_params())
        
        if use_log_cycle_life:
            enet.fit(features,np.log10(cycle_lives))
            predicted_cycle_lives = 10**enet.predict(features)
        else:
            enet.fit(features,cycle_lives)
            predicted_cycle_lives = enet.predict(features)

        trained_models.append(enet)        
        
        plt.plot(cycle_lives,predicted_cycle_lives,'o')
        plt.plot([0,1400],[0,1400],'r-')
        plt.ylabel('Predicted cycle lives')
        plt.xlabel('Actual cycle lives')
        #plt.axis('equal')
        plt.axis([0, 1400, 0, 1400])
        plt.show()
        
        residuals = predicted_cycle_lives - cycle_lives
        min_rmse[i] = np.sqrt(((residuals) ** 2).mean())
        min_percent_error[i] = (np.abs(residuals)/cycle_lives).mean()*100
        
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


        
        
    # explainElasticNetCVResults(enet)

    # make nice plots
    plt.plot(N_cycles, min_rmse, '-o')
    plt.ylabel('RMSE error')
    plt.xlabel('N cycles')
    plt.show()

    plt.plot(N_cycles, min_percent_error, '-o')
    plt.ylabel('Percent error')
    plt.xlabel('N cycles')
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(N_cycles, optimal_l1_ratio, '-o')
    plt.ylabel('Optimal L1 ratio')
    plt.xlabel('N cycles')
    
    plt.subplot(2, 1, 2)
    plt.plot(N_cycles, optimal_alpha, '-o')
    plt.ylabel('Optimal alpha')
    plt.xlabel('N cycles')
    plt.show()
    
    # export coeff matrix to csv
    if use_all_features:
        df = pd.DataFrame(norm_coeffs, columns=N_cycles, index=feature_names)
    else:
        which_features[:] = [x - 2 for x in which_features]
        df = pd.DataFrame(norm_coeffs, columns=N_cycles, index=[feature_names[i] for i in which_features])
    df.to_csv("norm_coeffs.csv")
    #export trained models and training error
    pickle.dump(trained_models, open('enet_trained_models.pkl', 'wb'))
    pickle.dump(min_rmse, open('enet_training_rmse.pkl', 'wb'))
    pickle.dump(min_percent_error, open('enet_training_percenterror.pkl', 'wb'))


    
    
    
    
def explainElasticNetCVResults(enet):
    print('Optimal alpha:')
    print(enet.alpha_)
    print('Optimal l1_ratio:')
    print(enet.l1_ratio_)
    print('Mean-squared error array (:')
    print(enet.mse_path_)

    mean_mse = np.mean(enet.mse_path_, axis=2)
    print('Min mean-squared error (avg. over CV):')
    print(min(mean_mse[~np.isnan(mean_mse)]))    
    
    print('Coefficients:')
    print(enet.coef_)
    
    
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

    # def add_intercept_fn(x):
    #     global add_intercept
    #     return add_intercept(x)

    # # Validate label_col argument
    # allowed_label_cols = ('y', 't')
    # if label_col not in allowed_label_cols:
    #     raise ValueError('Invalid label_col: {} (expected {})'
    #                      .format(label_col, allowed_label_cols))

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
    # print(m)
    # print(features.shape)
    # print( np.ones([m, 1]))
    if add_intercept:
        features = np.concatenate((np.ones([m, 1]), features),axis=1)
        feature_names = ['intercept'] + feature_names

    return features, cycle_lives, feature_names






if __name__ == "__main__":
    main()
