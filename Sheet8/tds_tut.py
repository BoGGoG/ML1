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
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures

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

testX_ohe = testX[ohe_cols].values
ohe_encoder = OneHotEncoder(categories = categories)
ohe_encoder.fit(testX_ohe)
testX_ohe = pd.DataFrame(ohe_encoder.transform(testX_ohe).toarray())

# Alpha (regularization strength) of LASSO regression
lasso_eps = 0.0001
lasso_nalpha=20
lasso_iter=5000

# Min and max degree of polynomials features to consider
degree_min = 2
degree_max = 8


# Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)

for degree in range(degree_min,degree_max+1):
    model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,
normalize=True,cv=5))
    model.fit(trainX_ohe,trainy)
    test_pred = np.array(model.predict(testX_ohe))
    RMSE=np.sqrt(np.sum(np.square(test_pred-testy)))
    test_score = model.score(testX_ohe,testy)
