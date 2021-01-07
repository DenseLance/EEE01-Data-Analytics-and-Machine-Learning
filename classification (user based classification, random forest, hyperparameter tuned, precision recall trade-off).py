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

# Random forest classifier (after exhaustive search for hyperparameters)
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(bootstrap = False,
                                       max_depth = 30,
                                       max_features = "sqrt",
                                       min_samples_leaf = 1,
                                       min_samples_split = 3,
                                       n_estimators = 1600,
                                       random_state = seed)

# Graph showing variation of precision and recall with thresholds
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib as mpl

predictions = cross_val_predict(random_forest, X, y, cv = cross_validator, method = "predict_proba")
precisions, recalls, thresholds = precision_recall_curve(y, predictions[:,1])

fig = plt.figure(figsize = (15, 10))
mpl.style.use('seaborn')
plt.plot(thresholds, precisions[:-1], label = "Precision", color = "green")
plt.plot(thresholds, recalls[:-1], label = "Recall", color = "red")
plt.xlabel("Threshold")
plt.ylabel("Precision/Recall")
plt.yticks([0.2 * label for label in range(0, 6)])
plt.ylim(bottom = 0, top = 1.0)
plt.xticks([0.0, 1.0])
plt.xlim(left = 0, right = 1.0)
fig.legend(labels = ("Precision", "Recall"), loc = "best")
plt.title("User Based Classification (Analysis of Hyperparameter Optimised Random Forest Using Precision-Recall Curve With Thresholds)")

wm = plt.get_current_fig_manager()
wm.window.state("zoomed")

# Graph showing variation of precision and recall with iso-f1 curves
from scikitplot.metrics import plot_precision_recall

pr_curve = plot_precision_recall(y, predictions, plot_micro = False, classes_to_plot = [1], title = "User Based Classification (Analysis of Hyperparameter Optimised Random Forest Using Precision-Recall Curve With Iso-F1 Curves)", cmap = "Blues", figsize = (15, 10))
mpl.style.use('seaborn')
pr_curve.get_legend().remove()

f1_curve_values = [0.2 * i for i in range(1, 5)] # 0.2 to 0.8

for value in f1_curve_values:
    x_coor = np.linspace(0.01, 1)
    y_coor = value * x_coor / (2 * x_coor - value)
    plt.plot(x_coor[y_coor >= 0], y_coor[y_coor >= 0], label = "iso-f1 curves", color = 'gray', alpha = 0.2)
    plt.annotate(f"f1 = {round(value, 1)}", xy = (0.9, y_coor[45] + 0.02))

pr_curve.legend(labels = ("precision-recall curve", "iso-f1 curves"), loc = "best")

wm = plt.get_current_fig_manager()
wm.window.state("zoomed")
plt.show()
