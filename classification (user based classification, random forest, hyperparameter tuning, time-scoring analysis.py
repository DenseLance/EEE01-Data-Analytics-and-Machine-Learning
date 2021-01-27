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

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(random_state = seed, bootstrap = False, max_depth = 20, max_features = "sqrt", min_samples_leaf = 3, min_samples_split = 4, n_estimators = 1000)

from numpy import mean, std
from sklearn.model_selection import cross_validate
scores = cross_validate(random_forest, X, y, cv = cross_validator, scoring = ["accuracy", "precision", "recall", "f1"])

print("Results when using best hyperparamaters from time-scoring analysis:")
print("accuracy:", mean(scores["test_accuracy"]))
print("std:", std(scores["test_accuracy"]))
print()

print("precision:", mean(scores["test_precision"]))
print("std:", std(scores["test_precision"]))
print()

print("recall:", mean(scores["test_recall"]))
print("std:", std(scores["test_recall"]))
print()

print("f1:", mean(scores["test_f1"]))
print("std:", std(scores["test_f1"]))
print()

print("\n\n\n")
