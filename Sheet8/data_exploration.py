import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

filenameX = 'trainX.dat'
filenamet = 'traint.dat'

dfx = pd.read_csv(filenameX, sep = '\t', header = None)
dft = pd.read_csv(filenamet, sep = '\t', header = None, names = 'y')
# x = StandardScaler().fit_transform(dfx.values)
# df = pd.DataFrame(x)
df = dfx
df['y'] = dft['y']

# histogram
df.hist(figsize = (15,15))
plt.show()

figsize = (10, 8)
cols = 3
gs = gridspec.GridSpec(3 // cols + 1, cols)
gs.update(hspace=0.4)

# ax = []
# for feature in [0,1,2,3,4,5]:
    # pd.crosstab(df[feature],df.y).plot(kind='bar')

# plt.show()

#Calculate correlations between numeric features
correlations = df.corr()

#Make the figsize 7 x 6
plt.figure(figsize=(7,6))

#Plot heatmap of correlations
# _ = sns.heatmap(correlations, cmap="Greens")
plt.show()
