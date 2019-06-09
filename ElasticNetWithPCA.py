import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
#import pandas as pd
import pickle
import csv

def main():
    N_cycles = np.array([100]) #np.array([20,30,40,50,60,70,80,90,100])
    #N_cycles = np.array([40])
    N_features_to_use = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30]);

    
    min_rmse = np.zeros([len(N_cycles), len(N_features_to_use)])
    min_percent_error = np.zeros([len(N_cycles), len(N_features_to_use)])
    optimal_l1_ratio = np.zeros([len(N_cycles), len(N_features_to_use)])
    optimal_alpha = np.zeros([len(N_cycles), len(N_features_to_use)])
    dev_error = np.zeros([len(N_cycles), len(N_features_to_use)])
    use_log_cycle_life = True    
    use_all_features = False
        
    trained_models = []    
    
    for i in np.arange(len(N_cycles)):
        print('Starting N_cycles = ' + str(int(N_cycles[i])))
        
        for j in np.arange(len(N_features_to_use)):
    
            N_features = N_features_to_use[j]   
            print('Starting N_features = ' + str(int(N_features)))
            
            feature_path = "pca/train_cycle" + str(int(N_cycles[i])) + ".csv"
            cycle_lives_path = "pca/train_cycle_lives.csv"
            features, cycle_lives = load_dataset(feature_path, cycle_lives_path, False, use_all_features, N_features)
            
            # pre-initiliaze array which depends on N_features_to_use
            if i == 0:
                norm_coeffs = np.zeros([features.shape[1],len(N_cycles)])
    
    
            
                
            # Elastic Net CV
            l1_ratio = [1]#[0.01, .1, .5, .7, .9, .95, .99, 1]
            alphas = [0]
            model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio, cv=5,
                                fit_intercept=True, normalize=True,verbose=False, 
                                max_iter=60000,random_state=0)
            #model = ElasticNetCV(cv=5, fit_intercept=True, normalize=True,verbose=False,max_iter=60000,random_state=0)
            # print('Elastic Net CV parameters:')    
            # print(model.get_params())
            
            if use_log_cycle_life:
                model.fit(features,np.log10(cycle_lives))
                dev_error[i,j] = np.mean(np.sqrt(-cross_val_score(model, features, np.log10(cycle_lives), cv=5, scoring='neg_mean_squared_error')))
                predicted_cycle_lives = 10**model.predict(features)
            else:
                model.fit(features,cycle_lives)
                dev_error[i,j] = np.mean(np.sqrt(-cross_val_score(model, features, cycle_lives, cv=5, scoring='neg_mean_squared_error')))
                predicted_cycle_lives = model.predict(features)
    
            trained_models.append(model)        
            
            plt.plot(cycle_lives,predicted_cycle_lives,'o')
            plt.plot([0,1400],[0,1400],'r-')
            plt.ylabel('Predicted cycle lives')
            plt.xlabel('Actual cycle lives')
            #plt.axis('equal')
            plt.axis([0, 1400, 0, 1400])
            plt.show()
            
            residuals = predicted_cycle_lives - cycle_lives
            min_rmse[i,j] = np.sqrt(((residuals) ** 2).mean())
            min_percent_error[i,j] = (np.abs(residuals)/cycle_lives).mean()*100
            
            # mean_mse = np.mean(model.mse_path_, axis=2)
            # mean_rmse = np.sqrt(mean_mse)
            #min_rmse[i] = min(mean_rmse[~np.isnan(mean_rmse)])
            optimal_l1_ratio[i,j] = model.l1_ratio_
            optimal_alpha[i,j] = model.alpha_
                    
            print('Min RMSE:')
            print(min_rmse[i,j])
            print('Optimal alpha:')
            print(model.alpha_)
            print('Optimal l1_ratio:')
            print(model.l1_ratio_)
            print('Normalized coefficients:')
            norm_coeffs[:,i] = model.coef_ * np.std(features,axis=0)
            print(norm_coeffs[:,i])
            print('N iterations to convergence: ' + str(int(model.n_iter_)))
            print()
            print('Finished N_cycles = ' + str(int(N_cycles[i])))
            print('=======================================')
    
    
            
            
        # explainElasticNetCVResults(model)
            
        plt.plot(N_features_to_use,dev_error[i,:],'-o')
        plt.ylabel('Dev rmse')
        plt.xlabel('N features used')
        plt.show()    
        
        
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
    #    if use_all_features:
    #        df = pd.DataFrame(norm_coeffs, columns=N_cycles, index=feature_names)
    #    else:
    #        which_features[:] = [x - 2 for x in which_features]
    #        df = pd.DataFrame(norm_coeffs, columns=N_cycles, index=[feature_names[i] for i in which_features])
    #    df.to_csv("norm_coeffs.csv")
    #    
    #    #export trained models and training error
    #    pickle.dump(trained_models, open('model_trained_models.pkl', 'wb'))
    #    pickle.dump(min_rmse, open('model_training_rmse.pkl', 'wb'))
    #    pickle.dump(min_percent_error, open('model_training_percenterror.pkl', 'wb'))
    

    
    
    
    
def explainElasticNetCVResults(model):
    print('Optimal alpha:')
    print(model.alpha_)
    print('Optimal l1_ratio:')
    print(model.l1_ratio_)
    print('Mean-squared error array (:')
    print(model.mse_path_)

    mean_mse = np.mean(model.mse_path_, axis=2)
    print('Min mean-squared error (avg. over CV):')
    print(min(mean_mse[~np.isnan(mean_mse)]))    
    
    print('Coefficients:')
    print(model.coef_)
    
    
def load_dataset(feature_path, cycle_lives_path, add_intercept=True, use_all_features=True, N_features_to_use=20):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (features).
        ys: Numpy array of y-values (labels).
        headers: list of headers
    """

    # first load features 
    with open(feature_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        # get all the rows as a list
        features = list(reader)
        # transform data into numpy array
        features = np.array(features).astype(float)
        m = features.shape[0]
        # print(m)
        # print(features.shape)
        # print( np.ones([m, 1]))
                
    features = features[:, 0:N_features_to_use]
    
    if add_intercept:
        features = np.concatenate((np.ones([m, 1]), features),axis=1)


    with open(cycle_lives_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        # get all the rows as a list
        cycle_lives = list(reader)
        # transform data into numpy array
        cycle_lives = np.array(cycle_lives).astype(float).flatten()
        m = cycle_lives.shape[0]
        # print(m)
        # print(features.shape)
        # print( np.ones([m, 1]))

    return features, cycle_lives






if __name__ == "__main__":
    main()
