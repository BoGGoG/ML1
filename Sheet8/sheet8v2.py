import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

train_percentage = 0.8
C = 1.
penalty = 'l2'
max_iter = 500

X_file = 'trainX.dat'
y_file = 'traint.dat'

df = pd.read_csv(X_file, sep = '\t', header = None)
y = pd.read_csv(y_file, sep = '\t', header = None)
df['y'] = y[0]


## TRAIN TEST SPLIT
train = df.sample(frac = train_percentage, random_state = 1337)
test = df.drop(train.index)

trainy = train['y']
trainX = train.drop('y', axis = 1)

testy = test['y']
testX = test.drop('y', axis = 1)

## One-Hot-Encoding
ohe_cols = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]
non_ohe_cols = [1,4,12, 19] # leaving out 19

categories = [[1,2,3,4], [0,1,2,3,4], [0,1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5],
       [1,2,3,4,5], [1,2,3,4], [1,2,3,4], [1,2,3], [1,2,3,4], [1,2,3,4], [1,2,3],
       [1,2,3], [1,2,3,4], [1,2,3,4], [1,2], [1,2]]

trainX_ohe = trainX[ohe_cols].values
ohe_encoder = OneHotEncoder(categories = categories)
ohe_encoder.fit(trainX_ohe)
trainX_ohe = pd.DataFrame(ohe_encoder.transform(trainX_ohe).toarray())
trainX_ohe['a'] = trainX[non_ohe_cols[0]].values
trainX_ohe['b'] = trainX[non_ohe_cols[1]].values
trainX_ohe['c'] = trainX[non_ohe_cols[2]].values
trainX_ohe['d'] = trainX[non_ohe_cols[3]].values
print(trainX_ohe.head())


testX_ohe = testX[ohe_cols].values
ohe_encoder = OneHotEncoder(categories = categories)
ohe_encoder.fit(testX_ohe)
testX_ohe = pd.DataFrame(ohe_encoder.transform(testX_ohe).toarray())

testX_ohe['a'] = testX[non_ohe_cols[0]].values
testX_ohe['b'] = testX[non_ohe_cols[1]].values
testX_ohe['c'] = testX[non_ohe_cols[2]].values
testX_ohe['d'] = testX[non_ohe_cols[3]].values

# LOGISTIC MODEL
# logreg = LogisticRegression(C = C, penalty = penalty, max_iter = max_iter, solver = 'lbfgs')
# logreg.fit(trainX_ohe, trainy.ravel())


# # # PREDICTION
# predy = logreg.predict(testX_ohe)
# predy_prob = logreg.predict_proba(testX_ohe)
# score = logreg.score(testX_ohe, testy)
# print(score)

pipe_lr = Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(penalty = 'l2', C = C, max_iter = max_iter, random_state=42, solver = 'lbfgs'))])

pipe_lr.fit(trainX_ohe, trainy.ravel())

score = pipe_lr.score(testX_ohe, testy)
print(score)
