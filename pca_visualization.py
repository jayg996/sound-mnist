from sklearn.decomposition import PCA
from dataloader import data_load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

z_normalization = True
x_train, y_train = data_load(split='train')
x_train = np.reshape(x_train, (x_train.shape[0], -1))

if z_normalization:
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train = (x_train - mean) / std

# PCA to dataframe
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_train)
pc_y = np.c_[principalComponents, y_train]
df = pd.DataFrame(pc_y, columns=['PC1','PC2','label'])

# scatter plot from dataframe
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:olive', 'tab:red', 'tab:purple']
for target, color in zip(targets,colors):
    indicesToKeep = df['label'] == target
    ax.scatter(df.loc[indicesToKeep, 'PC1'], df.loc[indicesToKeep, 'PC2'], c = color, s = 4)
ax.legend(targets)
ax.grid()
plt.show()