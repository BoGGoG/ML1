# run with cat trainX.dat | python3 sheet8.py
import pandas as pd
import numpy as np
import sys
from joblib import load, dump
from numpy.lib import recfunctions as rfn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

logreg_model_file = 'logreg.joblib'
usecols_file = 'usecols.joblib'
ohe_cols_file  = 'ohe_cols.joblib'
scaler_file = 'scaler.joblib'

use_cols = load(usecols_file)
ohe_cols = load(ohe_cols_file)
ohe_cols = np.intersect1d(use_cols, ohe_cols) # don't ohe unused features
scaler = load(scaler_file)

do_standard_scaler = False
do_transform = False


# data = sys.stdin.readlines()
dfX = pd.read_csv(sys.stdin, sep = '\t', header = None)
logreg = load(logreg_model_file)

dfX = dfX[use_cols]
dfX_ohe = pd.get_dummies(dfX, columns=ohe_cols) # y, X one hot encoded

if do_standard_scaler:
    dfX_ohe = scaler.transform(dfX_ohe)

if do_transform:
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution='uniform', random_state=0)
    dfX_ohe = pd.DataFrame(quantile_transformer.fit_transform(dfX_ohe))

# easy prediction
# predy = logreg.predict(dfX_ohe)

# prediction with threshold
threshold = 0.25
pred_proba_df = pd.DataFrame(logreg.predict_proba(dfX_ohe))
predy = pred_proba_df.applymap(lambda x: 1 if x>threshold else 0)
predy = predy[1].values

# output
for line in predy:
    print(line)


