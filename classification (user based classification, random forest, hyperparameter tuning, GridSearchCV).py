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

# Hyperparameter grid (108 settings)
from pprint import pprint

param_grid = {'bootstrap': [False],
              'max_depth': [20, 30, 40, 50],
              'max_features': ['sqrt'],
              'min_samples_leaf': [1, 2, 3],
              'min_samples_split': [2, 3, 4],
              'n_estimators': [1500, 1600, 1700]}

print("Hyperparameters considered for exhaustive search:")
pprint(param_grid)
print("\n\n\n\n")

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(random_state = seed)

# Exhaustive search for best hyperparameters
from numpy import mean, std
from sklearn.model_selection import GridSearchCV, cross_validate
grid_search = GridSearchCV(estimator = random_forest, param_grid = param_grid, cv = cross_validator, scoring = ["accuracy", "precision", "recall", "f1"], refit = "f1").fit(X, y)

print("Best hyperparameters using exhaustive search:")
pprint(grid_search.best_params_)
print("\n\n\n\n")

# Results using best hyperparameters (from exhaustive search)
best_index = grid_search.best_index_
print("Results when using best hyperparamaters from exhaustive search:")

print(f"accuracy: {grid_search.cv_results_['mean_test_accuracy'][best_index]}")
print(f"std: {grid_search.cv_results_['std_test_accuracy'][best_index]}")
print()

print(f"precision: {grid_search.cv_results_['mean_test_precision'][best_index]}")
print(f"std: {grid_search.cv_results_['std_test_precision'][best_index]}")
print()

print(f"recall: {grid_search.cv_results_['mean_test_recall'][best_index]}")
print(f"std: {grid_search.cv_results_['std_test_recall'][best_index]}")
print()

print(f"f1: {grid_search.cv_results_['mean_test_f1'][best_index]}")
print(f"std: {grid_search.cv_results_['std_test_f1'][best_index]}")
print()
print("\n\n\n")
