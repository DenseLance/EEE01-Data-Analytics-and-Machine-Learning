# Set Seed (Replication of Results)
seed = 40

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

# Import Dataset
import numpy as np
import pandas as pd

dataset = pd.read_csv("filtered dataset/user based classification.csv")
dataset.shape
dataset.head()
dataset.describe()

# IDs
y = dataset["bot"]
X = dataset.drop(["id", "language", "bot"], axis = 1)

# 5-Fold Cross Validation
from sklearn.model_selection import KFold

cross_validator = KFold(n_splits = 5, random_state = seed, shuffle = True)

# Hyperparameter Grid
param_grid = {'bootstrap': [True, False],
              'max_depth': [i * 10 for i in range(1, 21)], # 10 to 200
              'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_leaf': [i for i in range(1, 11)], # 1 to 10
              'min_samples_split': [i * 2 for i in range(1, 11)], # 2 to 20
              'n_estimators': [i * 100 for i in range(1, 21)]} # 100 to 2000

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(random_state = seed)

# Change 1 hyperparameter while the rest kept constant at default values
from sklearn.model_selection import GridSearchCV, cross_validate

for param in param_grid:
    param_dict = {param: param_grid[param]}
    print(f"Time-scoring analysis of hyperparameter: {param}")
    print(param_grid[param])
    print("\n\n\n\n")
    
    grid_search = GridSearchCV(estimator = random_forest, param_grid = param_dict, cv = cross_validator, scoring = ["accuracy", "precision", "recall", "f1"], refit = "f1").fit(X, y)
    
    for i in range(len(param_grid[param])):
        print(f"*{param}: {param_grid[param][i]}")
        
        print(f"time taken: {grid_search.cv_results_['mean_fit_time'][i]}")
        print(f"std: {grid_search.cv_results_['std_fit_time'][i]}")
        print()
        
        print(f"accuracy: {grid_search.cv_results_['mean_test_accuracy'][i]}")
        print(f"std: {grid_search.cv_results_['std_test_accuracy'][i]}")
        print()
        
        print(f"precision: {grid_search.cv_results_['mean_test_precision'][i]}")
        print(f"std: {grid_search.cv_results_['std_test_precision'][i]}")
        print()

        print(f"recall: {grid_search.cv_results_['mean_test_recall'][i]}")
        print(f"std: {grid_search.cv_results_['std_test_recall'][i]}")
        print()
        
        print(f"f1: {grid_search.cv_results_['mean_test_f1'][i]}")
        print(f"std: {grid_search.cv_results_['std_test_f1'][i]}")
        print()

        print("\n\n\n")
