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

train_percentage = 0.8

filenameX = 'trainX.dat'
filenamet = 'traint.dat'

df = pd.read_csv(filenameX, sep = '\t', header = None)
dft = pd.read_csv(filenamet, sep = '\t', header = None)

df['y'] = dft[0]

train = df.sample(frac = train_percentage, random_state = 1337)
test = df.drop(train.index)

traint = train['y']
trainX = train.drop('y', axis = 1)

testt = test['y']
testX = test.drop('y', axis = 1)

trainX.to_csv('only_trainX.dat', index = False, sep = '\t', header = False)
traint.to_csv('only_traint.dat', index = False, sep = '\t', header = False)
testX.to_csv('only_testX.dat', index = False, sep = '\t', header = False)
testt.to_csv('only_testt.dat', index = False, sep = '\t', header = False)
