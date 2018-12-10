import numpy as np
import matplotlib.pyplot as plt
import math
#import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import pandas as pd
import seaborn as sns

def main():
    #N_cycles = np.array([20,30,40,50,60,70,80,90,100])
    N_cycles = np.array([100])

    
    min_rmse = np.zeros(N_cycles.shape)
    use_log_cycle_life = False    
    use_log_features = False
    
    for i in np.arange(len(N_cycles)):
        print('Starting N_cycles = ' + str(int(N_cycles[i])))
        
        if use_log_features:
            file_name = "training/Cycles_2TO" + str(int(N_cycles[i])) + "_log.csv"
        else:
            file_name = "training/Cycles_2TO" + str(int(N_cycles[i])) + ".csv"

        features, cycle_lives, feature_names = load_dataset(file_name, False)
        
        # make some arrays that depend on number of features
        if i == 0:
            norm_coeffs = np.zeros([features.shape[1],len(N_cycles)])

#        from sklearn import svm
#        X = [[0, 0], [2, 2]]
#        y = [0.5, 2.5] 
#        clf = svm.SVR()
#        clf.fit(X, y) 
#        clf.predict([[1, 1]])
#        

#        C =  np.array([1000]) #np.linspace(1,1000,50)
        n_trees =  np.array([5, 10, 20, 50, 100,200,1000])
        max_depth = [1,2,3,5,10,20,50,200, None] 
#        eps = np.logspace(-4,4,20)
        rmse = np.zeros([len(n_trees),len(max_depth)])        
        train_rmse = np.zeros([len(n_trees),len(max_depth)])        

        
        for j in np.arange(len(n_trees)):
            for k in np.arange(len(max_depth)):
                # SVR
        #        my_SVR = sk.svm.SVR(kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=1000)
                RF = RandomForestRegressor(n_estimators=n_trees[j], max_depth=max_depth[k], max_features='sqrt')
                print('n_trees = ' + str(n_trees[j]))
                print('max_depth = ' + str(max_depth[k]))
                
                if use_log_cycle_life:
                    RF.fit(features,np.log10(cycle_lives))
                    predicted_cycle_lives = 10**RF.predict(features)
                    residuals = predicted_cycle_lives - cycle_lives
                    train_rmse[j,k] = np.sqrt(((residuals) ** 2).mean())
#                    R_squared = my_SVR.score(features,np.log10(cycle_lives))
                    mse = cross_val_score(RF, features, np.log10(cycle_lives), cv=5, scoring='mean_squared_error')
                    rmse[j,k] = np.sqrt(np.mean(mse))
                else:
                    RF.fit(features,cycle_lives)
                    predicted_cycle_lives = RF.predict(features)
                    residuals = predicted_cycle_lives - cycle_lives
                    train_rmse[j,k] = np.sqrt(((residuals) ** 2).mean())
#                    R_squared = my_SVR.score(features,cycle_lives)
                    mse = cross_val_score(RF, features, cycle_lives, cv=5, scoring='mean_squared_error')
                    rmse[j,k] = np.sqrt(abs(np.mean(mse)))
        
#                train_rmse[j,k] = 
                plt.plot(cycle_lives,predicted_cycle_lives,'o')
                plt.plot([0,2400],[0,2400],'r-')
                plt.ylabel('Predicted cycle lives')
                plt.xlabel('Actual cycle lives')
                #plt.axis('equal')
                plt.axis([0, 1400, 0, 1400])
                plt.show()
                
                residuals = predicted_cycle_lives - cycle_lives
                min_rmse[i] = np.sqrt(((residuals) ** 2).mean())
                
                print('Training error:')
                print(train_rmse[j,k])
                        
                print('RMSE with cross validation:')
                print(rmse[j,k])
        #        print('N iterations to convergence: ' + str(int(enet.n_iter_)))
#                print('R_square = ' + str(R_squared))
                print('Finished N_cycles = ' + str(int(N_cycles[i])))
                print('=======================================')

        
        print('Training MSE:')
        ax = sns.heatmap(train_rmse)
        plt.show()
        print('Cross-validation MSE:')
        ax = sns.heatmap(rmse)
        plt.show()
        
#    # make nice plots
#    plt.plot(N_cycles, min_rmse, '-o')
#    plt.ylabel('RMSE error')
#    plt.xlabel('N cycles')
#    plt.show()

#    plt.subplot(2, 1, 1)
#    plt.plot(N_cycles, optimal_l1_ratio, '-o')
#    plt.ylabel('Optimal L1 ratio')
#    plt.xlabel('N cycles')
#    
#    plt.subplot(2, 1, 2)
#    plt.plot(N_cycles, optimal_alpha, '-o')
#    plt.ylabel('Optimal alpha')
#    plt.xlabel('N cycles')
#    plt.show()
    
    # export coeff matrix to csv
#    df = pd.DataFrame(norm_coeffs, columns=N_cycles, index=feature_names)
#    df.to_csv("norm_coeffs.csv")

    
    
def load_dataset(csv_path, add_intercept=True):
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
    features = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=range(2, len(headers)))
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
