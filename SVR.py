import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import pandas as pd
import seaborn as sns

def main():
    N_cycles = np.array([20,30,40,50,60,70,80,90,100])
#    N_cycles = np.array([ 40])

    
    min_rmse = np.zeros(N_cycles.shape)
    min_mpe = np.zeros(N_cycles.shape)
    training_mpe = np.zeros(N_cycles.shape)
    min_percent_error = np.zeros(N_cycles.shape)
#    use_log_cycle_life = False    
    use_log_features = True
    use_all_features = False
    which_features = [2,3,4,21,22,24,25,39,40,48,49,63,65]#list(map(int, np.linspace(2,12,11) ))
    
    best_C = np.zeros(N_cycles.shape)
    best_eps = np.zeros(N_cycles.shape)    
    trained_models = []
    
    for i in np.arange(len(N_cycles)):
        print('Starting N_cycles = ' + str(int(N_cycles[i])))
        
        if use_log_features:
            file_name = "training/Cycles_2TO" + str(int(N_cycles[i])) + "_log.csv"
        else:
            file_name = "training/Cycles_2TO" + str(int(N_cycles[i])) + ".csv"

        features, cycle_lives, feature_names = load_dataset(file_name, False, use_all_features, which_features)
        
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
#        C =  np.linspace(1,100000,50)
        C = np.logspace(0,6,8)
        eps = np.logspace(-2.5,-0.5,6)

        rmse = np.zeros([len(C),len(eps)])        
        train_rmse = np.zeros([len(C),len(eps)])
        mpe = np.zeros([len(C),len(eps)])        


        
        for j in np.arange(len(C)):
            for k in np.arange(len(eps)):
                # SVR
        #        my_SVR = sk.svm.SVR(kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=1000)
                my_SVR = SVR(kernel='rbf',C=C[j], epsilon=eps[k])
                print('C = ' + str(C[j]))
                print('eps = ' + str(eps[k]))
                
#                if use_log_cycle_life:
#                    my_SVR.fit(features,np.log10(cycle_lives))
#                    predicted_cycle_lives = 10**my_SVR.predict(features)
#                    residuals = predicted_cycle_lives - cycle_lives
#                    train_rmse[j,k] = np.sqrt(((residuals) ** 2).mean())
##                    R_squared = my_SVR.score(features,np.log10(cycle_lives))
#                    mse = cross_val_score(my_SVR, features, np.log10(cycle_lives), cv=5, scoring='mean_squared_error')
#                    rmse[j,k] = np.sqrt(np.mean(mse))
#                    next_mpe = cross_val_score(my_SVR, features, np.log10(cycle_lives), cv=5, scoring=mean_percent_error)
#                    mpe[j,k] = np.mean(next_mpe)
#                else:
                my_SVR.fit(features,cycle_lives)
                predicted_cycle_lives = my_SVR.predict(features)
                residuals = predicted_cycle_lives - cycle_lives
                train_rmse[j,k] = np.sqrt(((residuals) ** 2).mean())
#                    R_squared = my_SVR.score(features,cycle_lives)
                mse = -cross_val_score(my_SVR, features, cycle_lives, cv=5, scoring='mean_squared_error')
                rmse[j,k] = np.sqrt(abs(np.mean(mse)))
                next_mpe = cross_val_score(my_SVR, features, np.log10(cycle_lives), cv=5, scoring=mean_percent_error)
                mpe[j,k] = np.mean(next_mpe)
                
                plt.plot(cycle_lives,predicted_cycle_lives,'o')
                plt.plot([0,2400],[0,2400],'r-')
                plt.ylabel('Predicted cycle lives')
                plt.xlabel('Actual cycle lives')
                #plt.axis('equal')
                plt.axis([0, 1400, 0, 1400])
                plt.show()
                
        
                print('Training error:')
                print(train_rmse[j,k])
                
                print('RMSE with cross validation:')
                print(rmse[j,k])

                print('MPE with cross validation:')
                print(mpe[j,k])
    
                
        #        print('N iterations to convergence: ' + str(int(enet.n_iter_)))
#                print('R_square = ' + str(R_squared))
                print('Finished N_cycles = ' + str(int(N_cycles[i])))
                print('=======================================')

        
        print('Min RMSE with cross validation:')
        print(np.min(rmse))        
        
        print('Min MPE with cross validation:')
        print(np.min(mpe))          
        
        min_rmse[i] = np.min(rmse)
        min_mpe[i] = np.min(mpe)
#        best_index = np.where(mpe==np.min(mpe))
#        best_C[i] = C[best_index[0]]
#        best_eps[i] = eps[best_index[1]]
        index_best = np.argmin(rmse)
        best_C_index = index_best // eps.shape[0]
        print('Best C index ' + str(int(best_C_index)))
        best_C[i] = C[best_C_index]
        
        best_eps_index = np.mod(index_best, eps.shape[0])
        print('Best eps index ' + str(int(best_eps_index)))
        best_eps[i] = eps[best_eps_index]

        print('Best eps:')
        print(best_eps[i])
        print('Best C:')
        print(best_C[i])
        
        print('Training MSE:')
        ax = sns.heatmap(train_rmse)
        plt.show()
        print('Cross-validation MSE:')
        ax = sns.heatmap(rmse)
        plt.show()
        print('Cross-validation MPE:')
        ax = sns.heatmap(mpe)
        plt.show()
        
        # Train new model using best hyperparameters:
        best_SVR = SVR(kernel='rbf',C=best_C[i], epsilon=best_eps[i])
        best_SVR.fit(features,cycle_lives)
        trained_models.append(best_SVR)
        
        predicted_cycle_lives = best_SVR.predict(features)
        residuals = predicted_cycle_lives - cycle_lives
        training_mpe[i] = (np.abs(residuals)/cycle_lives).mean()*100
        
        plt.plot(cycle_lives,predicted_cycle_lives,'o')
        plt.plot([0,2400],[0,2400],'r-')
        plt.ylabel('Predicted cycle lives')
        plt.xlabel('Actual cycle lives')
        #plt.axis('equal')
        plt.axis([0, 1400, 0, 1400])
        plt.show()
        
 # make nice plots
    plt.plot(N_cycles, min_rmse, '-o')
    plt.ylabel('RMSE error')
    plt.xlabel('N cycles')
    plt.show()
    
    plt.plot(N_cycles, min_mpe, '-o')
    plt.ylabel('MPE error')
    plt.xlabel('N cycles')
    plt.show()

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
    
    
    pickle.dump(trained_models, open('SVR_trained_models.pkl', 'wb'))
    pickle.dump(min_mpe, open('SVR_crossvalid_percenterror.pkl', 'wb'))  
    pickle.dump(training_mpe, open('SVR_training_percenterror.pkl', 'wb'))    

    

    
    
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


def mean_percent_error(model, X, y):
    predicted_y = model.predict(X)
    residuals = predicted_y - y
    return (np.abs(residuals)/y).mean()*100

    




if __name__ == "__main__":
    main()
