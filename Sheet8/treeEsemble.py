# Author: Tim Head <betatim@gmail.com>
#
# License: BSD 3 clause

import numpy as np
np.random.seed(10)

import matplotlib.pyplot as plt

import sys
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
import pandas as pd

filenameX = 'only_trainX.dat'
filenamet = 'only_traint.dat'
# filenameX_test = 'only_testX.dat'
# filenameX = 'trainX.dat'
# filenamet = 'traint.dat'


X = pd.read_csv(filenameX, header = None, sep = '\t').values
y = pd.read_csv(filenamet, header = None, sep = '\t').values.ravel()
# X_test = pd.read_csv(filenameX_test, header = None, sep = '\t').values
X_test = pd.read_csv(sys.stdin, sep = '\t', header = None).values

n_estimator = 10
# X, y = make_classification(n_samples=80000)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0)
X_train = X
y_train = y

# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid
# overfitting, in particular if the total number of leaves is
# similar to the number of training samples
X_train, X_train_lr, y_train, y_train_lr = train_test_split(
    X_train, y_train, test_size=0.5)

# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
                          random_state=0)

rt_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
# fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder(categories='auto')
rf_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
# fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

# Supervised transformation based on gradient boosted trees
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder(categories='auto')
grd_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
# fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)

# The gradient boosted model by itself
y_pred_grd = grd.predict_proba(X_test)[:, 1]
# fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
# fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)


def prob_to_class(y_pred_prob, threshold):
    if y_pred_prob > threshold:
        return 1
    else:
        return 0
prob_to_class_vectorized = np.vectorize(prob_to_class)

threshold = 0.4 # empirical value because of looking at data
y_pred = prob_to_class_vectorized(y_pred_grd_lm, threshold)

# output
for i in y_pred:
    print(i)

# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
# plt.plot(fpr_rf, tpr_rf, label='RF')
# plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
# plt.plot(fpr_grd, tpr_grd, label='GBT')
# plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.show()

# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
# plt.plot(fpr_rf, tpr_rf, label='RF')
# plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
# plt.plot(fpr_grd, tpr_grd, label='GBT')
# plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve (zoomed in at top left)')
# plt.legend(loc='best')
# plt.show()
