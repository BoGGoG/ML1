import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
df = pd.DataFrame(iris.data[:, :4])
# df.columns = iris.feature_names
df['y'] = iris.target
# print(df.head())
X = df[[0,1,2,3]]
y = df['y']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# a) one-verus-the-rest classification
# build a classifier for every class

classes = y.unique()
models = []
# class 0
for val in classes:
    y_train0 = y_train.map(lambda y: 1 if y == val else 0)
    logreg = LogisticRegression(solver = 'lbfgs')
    logreg.fit(X_train, y_train0)
    models.append(logreg)


# predict
predictions = pd.DataFrame(columns = classes, index = range(0, X_test.shape[0]))
for model_nr, model in enumerate(models):
    print(model_nr, model)
    predictions[model_nr] = model.predict_proba(X_test)[:,1]

y_pred = predictions.apply(lambda row: np.argmax(row.values), axis = 1)
print(confusion_matrix(y_test, y_pred))

# b) multinomial logistic model
logreg = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
# logreg = LogisticRegression(solver = 'lbfgs', multi_class = 'ovr') # ovr: one versus the rest, gives same as what I did manually above
logreg.fit(X_train, y_train)
y_pred_multinomial = logreg.predict(X_test)
print(confusion_matrix(y_test, y_pred_multinomial))

# 2: petal length
# 3: petal heigth

x_plot_0 = X_train[y_train == 0][2].values
y_plot_0 = X_train[y_train == 0][3].values
x_plot_1 = X_train[y_train == 1][2].values
y_plot_1 = X_train[y_train == 1][3].values
x_plot_2 = X_train[y_train == 2][2].values
y_plot_2 = X_train[y_train == 2][3].values
x_plot_0_test_mtn= X_test[y_pred_multinomial == 0][2].values
y_plot_0_test_mtn= X_test[y_pred_multinomial == 0][3].values
x_plot_1_test_mtn= X_test[y_pred_multinomial == 1][2].values
y_plot_1_test_mtn= X_test[y_pred_multinomial == 1][3].values
x_plot_2_test_mtn= X_test[y_pred_multinomial == 2][2].values
y_plot_2_test_mtn= X_test[y_pred_multinomial == 2][3].values

x_plot_0_test_ovr= X_test[y_pred.values == 0][2].values
y_plot_0_test_ovr= X_test[y_pred.values == 0][3].values
x_plot_1_test_ovr= X_test[y_pred.values == 1][2].values
y_plot_1_test_ovr= X_test[y_pred.values == 1][3].values
x_plot_2_test_ovr= X_test[y_pred.values == 2][2].values
y_plot_2_test_ovr= X_test[y_pred.values == 2][3].values

plt.scatter(x_plot_0, y_plot_0, color = 'red', marker = ".", label = 'training')
plt.scatter(x_plot_1, y_plot_1, color = 'blue', marker = ".")
plt.scatter(x_plot_2, y_plot_2, color = 'green', marker = ".")
plt.scatter(x_plot_0_test_mtn, y_plot_0_test_mtn - 0.02, color = 'red', marker = 'v', label = "multinomial")
plt.scatter(x_plot_1_test_mtn, y_plot_1_test_mtn - 0.02, color = 'blue', marker = 'v')
plt.scatter(x_plot_2_test_mtn, y_plot_2_test_mtn - 0.02, color = 'green', marker = 'v')
plt.scatter(x_plot_0_test_ovr, y_plot_0_test_ovr + 0.02, color = 'red', marker = '^', label = "one-versus-the-rest")
plt.scatter(x_plot_1_test_ovr, y_plot_1_test_ovr + 0.02, color = 'blue', marker = '^')
plt.scatter(x_plot_2_test_ovr, y_plot_2_test_ovr + 0.02, color = 'green', marker = '^')
plt.legend()
plt.xlabel("petal length")
plt.ylabel("petal heigth")
plt.show()
plt.savefig("petal_classified.png")
