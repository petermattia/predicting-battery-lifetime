"""
Implement linear regression and elastic net feature selection on PCA features
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer
import csv
import os

def main():
    plt.close('all')
    # All cycles to consider
    N_cycles = np.array([20,30,40,50,60,70,80,90,100])
    
    # Number of principal components to use as features
    N_features_to_use = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30]);
    
    # Settings
    use_elastic_net = False
    use_log_cycle_life = False
    use_all_features = False
    create_plots = True
    print_output = False
    
    if use_elastic_net and use_log_cycle_life:
        substr = 'enet_log_life'
    elif use_elastic_net and not use_log_cycle_life:
        substr = 'enet_lin_life'
    elif not use_elastic_net and use_log_cycle_life:
        substr = 'lr_log_life'
    elif not use_elastic_net and not use_log_cycle_life:
        substr = 'lr_lin_life'
    
    # Preinitialization
    min_rmse = np.zeros([len(N_cycles), len(N_features_to_use)])
    min_percent_error = np.zeros([len(N_cycles), len(N_features_to_use)])
    dev_error = np.zeros([len(N_cycles), len(N_features_to_use)])
    if use_elastic_net:
        optimal_l1_ratio = np.zeros([len(N_cycles), len(N_features_to_use)])
        optimal_alpha = np.zeros([len(N_cycles), len(N_features_to_use)])
    
    trained_models = []
    
    # loop through all cycles to consider for prediction
    for i, n_cyc in enumerate(N_cycles):
        print('Starting N_cycles = ' + str(int(n_cyc)))
        
        for j, n_features in enumerate(N_features_to_use):
            print('Starting N_features = ' + str(int(n_features)))
            
            # Load data
            feature_path = "pca/train_cycle" + str(int(n_cyc)) + ".csv"
            cycle_lives_path = "pca/train_cycle_lives.csv"
            features, cycle_lives = load_dataset(feature_path, 
                                                 cycle_lives_path, 
                                                 False, 
                                                 use_all_features, 
                                                 n_features)
            
            # pre-initiliaze array which depends on N_features_to_use
            if i == 0:
                norm_coeffs = np.zeros([features.shape[1],len(N_cycles)])
                
            # Build model
            if use_elastic_net: # elastic net
                model = ElasticNetCV(cv=5, fit_intercept=True, normalize=True,
                                     verbose=False,max_iter=60000,random_state=0)

            else: # linear regression
                model = LinearRegression(fit_intercept=True,normalize=True)
                
            # Fit and predict
            if use_log_cycle_life:
                def mean_squared_pow_error(y_true, y_pred,
                           sample_weight=None,
                           multioutput='uniform_average'):
                    # adapted from:
                    # https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/metrics/regression.py#L255                    
                    return mean_squared_error(np.power(10,y_true), np.power(10,y_pred),
                                              sample_weight, multioutput)
                
                
                log_cycle_life = np.log10(cycle_lives)
                model.fit(features,log_cycle_life)
                
                # make custom scorer
                pow_scorer = make_scorer(mean_squared_pow_error, greater_is_better=True)
                dev_error[i,j] = np.mean(np.sqrt(cross_val_score(model, 
                         features, log_cycle_life, cv=5, scoring=pow_scorer)))
                predicted_cycle_lives = 10**model.predict(features)
                
            else: # linear cycle life
                model.fit(features,cycle_lives)
                dev_error[i,j] = np.mean(np.sqrt(-cross_val_score(model, 
                         features, cycle_lives, cv=5, scoring='neg_mean_squared_error')))
                predicted_cycle_lives = model.predict(features)
            
            trained_models.append(model)
            
            if create_plots:
                plt.figure()
                plt.plot(cycle_lives,predicted_cycle_lives,'o')
                plt.plot([0,2400],[0,2400],'r-')
                plt.ylabel('Predicted cycle life')
                plt.xlabel('Actual cycle life')
                plt.title('n_cyc = ' + str(int(n_cyc)) + ', p = ' + str(int(n_features)))
                #plt.axis('equal')
                plt.axis([0, 2400, 0, 2400])
                plt.show()
                
                # save
                directory = 'plt/' + str(int(n_cyc)) + 'cyc'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig('plt/' + str(int(n_cyc)) + 'cyc/' + str(int(n_features))
                             + 'feat_' + substr, bbox_inches='tight')
            
            residuals = predicted_cycle_lives - cycle_lives
            min_rmse[i,j] = np.sqrt(((residuals) ** 2).mean())
            min_percent_error[i,j] = (np.abs(residuals)/cycle_lives).mean()*100
            
            # mean_mse = np.mean(model.mse_path_, axis=2)
            # mean_rmse = np.sqrt(mean_mse)
            #min_rmse[i] = min(mean_rmse[~np.isnan(mean_rmse)])
            if use_elastic_net:
                optimal_l1_ratio[i,j] = model.l1_ratio_
                optimal_alpha[i,j] = model.alpha_
                    
            if print_output:
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
                print('Finished N_cycles = ' + str(int(n_cyc)))
                print('=======================================')
    
            
        # explainElasticNetCVResults(model)
        
        ## Plots
        # Number of features used vs dev error
        plt.figure()
        plt.plot(N_features_to_use,dev_error[i,:],'-o')
        plt.xlabel('N features used')
        plt.ylabel('Dev RMSE')
        plt.title('N cycles = ' + str(int(n_cyc)))
        plt.ylim([200, 500])
        plt.show()
        plt.savefig('plt/' + str(int(n_cyc)) + 'cyc_' + substr, bbox_inches='tight')
        
        if create_plots:
            # RMSE vs number of cycles used
            plt.figure()
            plt.plot(N_cycles, min_rmse, '-o')
            plt.xlabel('N cycles')
            plt.ylabel('RMSE error')
            plt.show()
            
            # MAPE vs number of cycles used
            plt.figure()
            plt.plot(N_cycles, min_percent_error, '-o')
            plt.ylabel('Percent error')
            plt.xlabel('N cycles')
            plt.show()
            
            if use_elastic_net:
                # elastic net parameters
                plt.figure()
                plt.subplot(2, 1, 1)
                plt.plot(N_cycles, optimal_l1_ratio, '-o')
                plt.ylabel('Optimal L1 ratio')
                plt.xlabel('N cycles')
                
                plt.subplot(2, 1, 2)
                plt.plot(N_cycles, optimal_alpha, '-o')
                plt.ylabel('Optimal alpha')
                plt.xlabel('N cycles')
                plt.show()
                
        plt.close('all')
        
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
    

"""
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
"""
    
    
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
