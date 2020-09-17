import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


"""import warnings filter"""
from warnings import simplefilter
"""ignore all future warnings"""
simplefilter(action='ignore', category=FutureWarning)


"""Read csv file"""
Leave_Dataset = pd.read_csv('Combined-Employees-Dataset.csv', nrows=1).columns
Leave_Dataset = pd.read_csv('Combined-Employees-Dataset.csv', usecols=Leave_Dataset[1:])

print(Leave_Dataset.tail(5))

"""Target data Isolated"""
YtargetData = Leave_Dataset['left']


"""these columns not needed now"""
to_drop = ['Emp ID', 'salary', 'left']
leave_feat_space = Leave_Dataset.drop(to_drop, axis=1)


"""Pull out features for future use"""
features = leave_feat_space.columns


"""Convert label features to integers"""
le_sales = preprocessing.LabelEncoder()
le_sales.fit(leave_feat_space["dept"])
leave_feat_space["dept"] = le_sales.transform(leave_feat_space.loc[:,('dept')])

"""Transform the whole feature space into a matrix"""
X = leave_feat_space.values


"""Standardize all features"""
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

print("Feature space holds %d observations and %d features" % X.shape)
print("Unique target labels:", np.unique(YtargetData))


"""Prediction Function"""
def prediction_func(X, y, clf_class, method, **kwargs):
    from sklearn.model_selection import cross_val_predict

    """Initialize a classifier with key word arguments"""
    clf = clf_class(**kwargs)

    predicted = cross_val_predict(clf, X, y, cv=3, method=method)

    return predicted


def accuracy_measure(y, predicted):
    """NumPy interprets True and False as 1. and 0."""
    return metrics.accuracy_score(y, predicted)


print("Support Vector Machines:")
print("%.3f" % accuracy_measure(YtargetData, prediction_func(X, YtargetData, SVC, method='predict')))
print("Random Forest:")
print("%.3f" % accuracy_measure(YtargetData, prediction_func(X, YtargetData, RF, method='predict')))
print("K-Nearest-Neighbors:")
print("%.3f" % accuracy_measure(YtargetData, prediction_func(X, YtargetData, KNN, method='predict')))


"""Target classes"""
y = np.array(YtargetData)
class_names = np.unique(y)

"""Confusion matrices calculation"""
confusion_matrices = [
    ("Support Vector Machines", confusion_matrix(y, prediction_func(X, y, SVC, method='predict'))),
    ("Random Forest", confusion_matrix(y, prediction_func(X, y, RF, method='predict'))),
    ("K-Nearest-Neighbors", confusion_matrix(y, prediction_func(X, y, KNN, method='predict'))),
]
#
"""Print confusion matrix values"""
print confusion_matrices



"""Draw confusion matrices for prediction algorithms evaluation"""
for cfm in confusion_matrices:
    ax = plt.axes()
    ax.set_title(cfm[0])

    df_cm = pd.DataFrame(cfm[1], index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
    plt.figure(figsize=(6, 5))
    sb.heatmap(df_cm, annot=True, ax=ax, square=True, fmt="d", linewidths=.5)
plt.show()



"""Use 10 estimators so predictions are all multiples of 0.1"""
pred_prob = prediction_func(X, y, RF, n_estimators=10,  method='predict_proba',)

pred_leave = pred_prob[:, 1]
is_leave = y == 1

"""Number of times a predicted probability is assigned to an observation"""
counts = pd.value_counts(pred_leave)

"""probabilities of truth calculation"""
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_leave[pred_leave == prob])
    true_prob = pd.Series(true_prob)

"""Pandas concatenate function"""
counts = pd.concat([counts, true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
print(counts)



"""create a dataframe containing prob values"""
pred_prob_df = pd.DataFrame(pred_prob)
pred_prob_df.columns = ['prob_not_leaving', 'prob_leaving']

"""merge dataframes to get the name of employees"""
all_employees_pred_prob_df = pd.concat([Leave_Dataset, pred_prob_df], axis=1)

"""filter out employees still in the company and having a good evaluation"""
good_employees_still_working_df = all_employees_pred_prob_df[(all_employees_pred_prob_df["left"] == 0) & (all_employees_pred_prob_df["last_evaluation"] >= 0.7)]

good_employees_still_working_df.sort_values(by='prob_leaving', ascending=False, inplace=True)

"""Save into csv file"""
good_employees_still_working_df.to_csv("employees_leaving_prob.csv")
