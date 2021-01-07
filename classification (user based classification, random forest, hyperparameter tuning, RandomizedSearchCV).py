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

# Random Hyperparameter Grid (6480 settings chosen at random)
from pprint import pprint

n_estimators = [i * 200 for i in range(1, 11)] # number of trees in random forest
max_features = ["auto", "sqrt", "log2"] # maximum number of features to consider at every split
max_depth = [i * 10 for i in range(1, 11)] + [None] # maximum number of levels in tree
min_samples_split = [2, 5, 10] # minimum number of samples required to split
min_samples_leaf = [1, 2, 4] # minimum number of samples required at each leaf node
bootstrap = [True, False] # method of selecting samples for training each tree

random_grid = {"n_estimators": n_estimators,
               "max_features": max_features,
               "max_depth": max_depth,
               "min_samples_split": min_samples_split,
               "min_samples_leaf": min_samples_leaf,
               "bootstrap": bootstrap}

print("Hyperparameters considered for randomized search:")
pprint(random_grid)
print("\n\n\n\n")

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(random_state = seed)

# Randomized search for best hyperparameters
from numpy import mean, std
from sklearn.model_selection import RandomizedSearchCV, cross_validate
randomized_search = RandomizedSearchCV(estimator = random_forest, param_distributions = random_grid, n_iter = 100, cv = cross_validator, random_state = seed, scoring = ["accuracy", "precision", "recall", "f1"], refit = "f1").fit(X, y)

print("Best hyperparameters using randomized search:")
pprint(randomized_search.best_params_)
print("\n\n\n\n")

measures = {}
scores = cross_validate(random_forest, X, y, cv = cross_validator, scoring = ["accuracy", "precision", "recall", "f1"])
measures["accuracy"] = (mean(scores["test_accuracy"]), std(scores["test_accuracy"]))
measures["precision"] = (mean(scores["test_precision"]), std(scores["test_precision"]))
measures["recall"] = (mean(scores["test_recall"]), std(scores["test_recall"]))
measures["f1"] = (mean(scores["test_f1"]), std(scores["test_f1"]))

# Default hyperparameters
default_params = {"n_estimators": random_forest.get_params()["n_estimators"],
                  "max_features": random_forest.get_params()["max_features"],
                  "max_depth": random_forest.get_params()["max_depth"],
                  "min_samples_split": random_forest.get_params()["min_samples_split"],
                  "min_samples_leaf": random_forest.get_params()["min_samples_leaf"],
                  "bootstrap": random_forest.get_params()["bootstrap"]}

print("Default hyperparameters:")
pprint(default_params)
print("\n\n\n\n")

# Results using best hyperparameters (from randomized search)
best_index = randomized_search.best_index_
print("Results when using best hyperparamaters from randomized search:")

print(f"accuracy: {randomized_search.cv_results_['mean_test_accuracy'][best_index]}")
print(f"std: {randomized_search.cv_results_['std_test_accuracy'][best_index]}")
print()

print(f"precision: {randomized_search.cv_results_['mean_test_precision'][best_index]}")
print(f"std: {randomized_search.cv_results_['std_test_precision'][best_index]}")
print()

print(f"recall: {randomized_search.cv_results_['mean_test_recall'][best_index]}")
print(f"std: {randomized_search.cv_results_['std_test_recall'][best_index]}")
print()

print(f"f1: {randomized_search.cv_results_['mean_test_f1'][best_index]}")
print(f"std: {randomized_search.cv_results_['std_test_f1'][best_index]}")
print()
print("\n\n\n")

# Results using default hyperparameters
print("Results when using default hyperparamaters:")
for measure in measures:
    print(measure + ":" , measures[measure][0])
    print("std:" , measures[measure][1])
    print()
print("\n\n\n")
