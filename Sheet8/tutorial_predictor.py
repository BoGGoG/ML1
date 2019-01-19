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

models_file = 'tutorial_fitted_model.joblib'
ohe_encoder_file = 'tutorial_ohe_encoder.joblib'

ohe_encoder = load(ohe_encoder_file)
fitted_logreg_models = load(models_file)

X_test_file = 'only_testX.dat'
y_test_file = 'only_testt.dat'

X_test = pd.read_csv(X_test_file, sep = '\t', header = None)
y_test = pd.read_csv(y_test_file, sep = '\t', header = None)

ohe_cols = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]
non_ohe_cols = [1,4,12] # leaving out 19

categories = [[1,2,3,4], [0,1,2,3,4], [0,1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5],
       [1,2,3,4,5], [1,2,3,4], [1,2,3,4], [1,2,3], [1,2,3,4], [1,2,3,4], [1,2,3],
       [1,2,3], [1,2,3,4], [1,2,3,4], [1,2], [1,2]]
ohe_X = X_test[ohe_cols].values
ohe_encoder = OneHotEncoder(categories = categories)
ohe_encoder.fit(ohe_X)
ohe_X = pd.DataFrame(ohe_encoder.transform(ohe_X).toarray())

non_ohe_X = X_test[non_ohe_cols]
ohe_X['a'] = non_ohe_X[1]
ohe_X['b'] = non_ohe_X[4]
ohe_X['c'] = non_ohe_X[12]

X_test = ohe_X

# print(classification_report(y_test, fitted_logreg_models['l2'].predict(X_test)))

def prob_to_class(y_pred_prob, threshold):
    if y_pred_prob > threshold:
        return 1
    else:
        return 0
prob_to_class_vectorized = np.vectorize(prob_to_class)

y_pred_prob = fitted_logreg_models['l2'].predict_proba(X_test)[:,1]

threshold = 0.05
y_pred = prob_to_class_vectorized(y_pred_prob, threshold)

for line in y_pred:
    print(line)
