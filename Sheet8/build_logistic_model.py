import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


C = 1. # 1/regularizer for logistic regression
penalty = "l2"
max_iter = 500

do_transform = False
do_standard_scaler = False
oversampling = True
# oversampling = False

# for final model take full data
# filenameX = 'trainX.dat'
# filenamet = 'traint.dat'
filenameX = 'only_trainX.dat'
filenamet = 'only_traint.dat'

# not using column 19
use_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# use_cols = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15, 17, 18]

# read train data
dfx = pd.read_csv(filenameX, sep = '\t', header = None)
dft = pd.read_csv(filenamet, sep = '\t', header = None)



df = dfx[use_cols]
df['y'] = dft[0]


# one hot encoding columns (they are categorical)
ohe_cols = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]
ohe_cols = np.intersect1d(ohe_cols, use_cols)
df_ohe = pd.get_dummies(df, columns=ohe_cols) # y, X one hot encoded

# oh_encoder = OneHotEncoter()
# print(df_ohe.head())

# standard scaler
if do_standard_scaler:
    dfX = df_ohe.drop('y', axis = 1).values
    scaler = StandardScaler()
    scaler.fit(dfX)
    dfX = scaler.transform(dfX)
    dfX = pd.DataFrame(dfX)
    df_ohe = dfX
    df_ohe['y'] = dft[0]
    dump(scaler, 'scaler.joblib')
    print('Standard scaler saved in scaler.joblib.')

if do_transform:
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution='uniform', random_state=0)
    y = df_ohe['y']
    df_ohe = pd.DataFrame(quantile_transformer.fit_transform(df_ohe.drop('y', axis = 1)))
    df_ohe['y'] = dft[0]

# oversampling
if oversampling:
    X = df_ohe.loc[:, df_ohe.columns != 'y']
    y = df_ohe.loc[:, df_ohe.columns == 'y']


    os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    columns = X_train.columns

    os_data_X,os_data_y=os.fit_sample(X_train, y_train.values.ravel())
    os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
    os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
    trainX = os_data_X.values
    trainy = os_data_y.values
else:
    trainX = df_ohe.drop('y', axis = 1).values
    trainy = df_ohe['y'].values

# add non ohe cols

# Logistic Model
logreg = LogisticRegression(C = C, penalty = penalty, max_iter = max_iter)
logreg.fit(trainX, trainy.ravel())

# print("columns: ", df_ohe.drop('y', axis = 1).columns)


# export model
dump(logreg, 'logreg.joblib')
print('saved logred.joblib')
dump(use_cols, 'usecols.joblib')
print('saved usecols.joblib')
dump(ohe_cols, 'ohe_cols.joblib')
print('saved ohe_cols.joblib')
