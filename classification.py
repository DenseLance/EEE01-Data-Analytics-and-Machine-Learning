# Set Seed
seed = 40

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

# Import Dataset
import numpy as np
import pandas as pd

dataset = pd.read_csv("filtered dataset/tweet based classification (by tweet).csv")
dataset.shape
dataset.head()
dataset.describe()

# IDs
dataset.drop("id", axis = 1)

X = dataset.drop("bot", axis = 1)
y = dataset["bot"]

# 5-Fold Cross Validation
from sklearn.model_selection import KFold

cross_validator = KFold(n_splits = 5, random_state = seed, shuffle = True)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

classifiers = {"Logistic Regression": LogisticRegression(random_state = seed),
               "SGD": SGDClassifier(random_state = seed),
               "Regular Gradient Boosting": GradientBoostingClassifier(random_state = seed),
               "XGBoost": XGBClassifier(random_state = seed),
               "Decision Tree": DecisionTreeClassifier(random_state = seed),
               "Random Forest": RandomForestClassifier(n_estimators = 20, random_state = seed),
               "KNN": KNeighborsClassifier(n_neighbors = 5),
               "Neural Network": MLPClassifier(hidden_layer_sizes = (10, 10, 10), max_iter = 10000, random_state = seed),
               "SVM": SVC(random_state = seed)}

from numpy import mean, std
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import matplotlib as mpl

measures = {}
i = 0
no_of_classifiers = len(classifiers)
fig = plt.figure(figsize = (15, 10))
mpl.style.use('seaborn')

for classifier in classifiers:
    print(f"[{classifier}]")
    print("__" * len(classifier), "\n")

    # Accuracy, Precision, Recall, F1, ROC-AUC score
    scores = cross_validate(classifiers[classifier], X, y, cv = cross_validator, scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"])
    measures["accuracy"] = (mean(scores["test_accuracy"]), std(scores["test_accuracy"]))
    measures["precision"] = (mean(scores["test_precision"]), std(scores["test_precision"]))
    measures["recall"] = (mean(scores["test_recall"]), std(scores["test_recall"]))
    measures["f1"] = (mean(scores["test_f1"]), std(scores["test_f1"]))
    measures["roc-auc"] = (mean(scores["test_roc_auc"]), std(scores["test_roc_auc"]))
    for measure in measures:
        print(measure, ": ", measures[measure][0])
        print("std: ", measures[measure][1])
        print()
    print("\n\n\n")
    
    # Graph
    i += 1
    ax = fig.add_subplot(3, 3, i)
    plt.bar(list(measures.keys()), [measure[0] for measure in list(measures.values())], color = "grey")
    ax.set_title(classifier)
    plt.yticks([0.2 * label for label in range(0, 6)])
    plt.ylim(bottom = 0, top = 1.0)
    fig.tight_layout(pad = 4)
    
plt.bar(list(measures.keys()), [measure[0] for measure in list(measures.values())], color = "grey")

plt.suptitle("Tweet Based Classification (By Tweet)", fontweight = "bold", fontsize = "x-large", x = 0.51, y = 0.99)

wm = plt.get_current_fig_manager()
wm.window.state("zoomed")
plt.show()
