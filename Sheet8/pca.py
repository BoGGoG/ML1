import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

filenameX = 'trainX.dat'
filenamet = 'traint.dat'

dfx = pd.read_csv(filenameX, sep = '\t', header = None)
dft = pd.read_csv(filenamet, sep = '\t', header = None, names = 'y')
x = StandardScaler().fit_transform(dfx.values)

pca = PCA(.95)

pca.fit(x)
print('Principal components for 95% of variance: ', pca.n_components_)
