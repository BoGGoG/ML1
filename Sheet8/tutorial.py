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
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, confusion_matrix, classification_report

# filenameX = 'trainX.dat'
# filenamet = 'traint.dat'
filenameX = 'only_trainX.dat'
filenamet = 'only_traint.dat'
train_test_size = 0.0 # already only using train data

X = pd.read_csv(filenameX, sep = '\t', header = None)
y = pd.read_csv(filenamet, sep = '\t', header = None)

ohe_cols = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]
non_ohe_cols = [1,4,12] # leaving out 19

categories = [[1,2,3,4], [0,1,2,3,4], [0,1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5],
       [1,2,3,4,5], [1,2,3,4], [1,2,3,4], [1,2,3], [1,2,3,4], [1,2,3,4], [1,2,3],
       [1,2,3], [1,2,3,4], [1,2,3,4], [1,2], [1,2]]
ohe_X = X[ohe_cols].values
ohe_encoder = OneHotEncoder(categories = categories)
ohe_encoder.fit(ohe_X)
ohe_X = pd.DataFrame(ohe_encoder.transform(ohe_X).toarray())

non_ohe_X = X[non_ohe_cols]
ohe_X['a'] = non_ohe_X[1]
ohe_X['b'] = non_ohe_X[4]
ohe_X['c'] = non_ohe_X[12]

#Splitting the variables into predictor and target variables
X = ohe_X
y = y

#Setting up pipelines with a StandardScaler function to normalize the variables
pipelines = {
    'l1' : make_pipeline(StandardScaler(), 
                         LogisticRegression(penalty='l1' , random_state=42, class_weight='balanced')),
    'l2' : make_pipeline(StandardScaler(), 
                         LogisticRegression(penalty='l2' , random_state=42, class_weight='balanced')),
    #Setting the penalty for simple Logistic Regression as L2 to minimize the fitting time
    'logreg' : make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', random_state=42, class_weight='balanced'))
}

#Setting up a very large hyperparameter C for the non-penalized Logistic Regression (to cancel the regularization)
logreg_hyperparameters = {
    'logisticregression__C' : np.linspace(100000, 100001, 1),
    'logisticregression__fit_intercept' : [True, False]
}

#Setting up hyperparameters for the Logistic Regression with L1 penalty
l1_hyperparameters = {
    'logisticregression__C' : np.linspace(1e-3, 1e3, 10),
    'logisticregression__fit_intercept' : [True, False]
}

#Setting up hyperparameters for the Logistic Regression with L2 penalty
l2_hyperparameters = {
    'logisticregression__C' : np.linspace(1e-3, 1e3, 10),
    'logisticregression__fit_intercept' : [True, False]
}

#Creating the dictionary of hyperparameters
hyperparameters = {
    'logreg' : logreg_hyperparameters,
    'l1' : l1_hyperparameters,
    'l2' : l2_hyperparameters
}

#Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_size, random_state=42)

#Creating an empty dictionary for fitted models
fitted_logreg_models = {}

# Looping through model pipelines, tuning each with GridSearchCV and saving it to fitted_logreg_models
for name, pipeline in pipelines.items():
    #Creating cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    
    #Fitting the model on X_train, y_train
    model.fit(X_train, y_train)
    
    #Storing the model in fitted_logreg_models[name] 
    fitted_logreg_models[name] = model
    
    #Printing the status of the fitting
    print(name, 'has been fitted.')

#Creating an empty dictionary for predicted models
# predicted_logreg_models = {}

#Predicting the response variables and displaying the prediction score
# for name, model in fitted_logreg_models.items():
    # y_pred = model.predict(X_test)
    # predicted_logreg_models[name] = accuracy_score(y_test, y_pred)

# print(predicted_logreg_models)

#Creating the classification report
# print(classification_report(y_test, fitted_logreg_models['l2'].predict(X_test)))
# print(classification_report(y_test, fitted_logreg_models['l1'].predict(X_test)))
# print(classification_report(y_test, fitted_logreg_models['logreg'].predict(X_test)))

dump(ohe_encoder, 'tutorial_ohe_encoder.joblib')
dump(fitted_logreg_models, 'tutorial_fitted_model.joblib')


