# predicting battery lifetime

This repository contains code for our work on early prediction of battery lifetime. Features are generated in MATLAB, while the machine learning is performed in python using numpy, scikit-learn, and matplotlib.

Our key scripts and functions are summarized here:

MATLAB code:
- featuregeneration.m: Generates large set of features extracted from battery dataset and exports them to csvs. This function loops through cycles 20 through 100 in increments of 10.

Python code:
- ElasticNet.py:
- 
- test.py: Runs models on test data and generates plots of the results
