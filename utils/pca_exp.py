print(__doc__)


# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random
from sklearn import decomposition
r1 = 3
r2 = 10
n = 40 #points to generate
rot_angle = 10
ellipsePoints = list()
for i in range(1,800):
    angle = random.uniform(0,2*math.pi)
    x = math.cos(angle) * random.uniform(0, r2)
    y = math.sin(angle) * random.uniform(0, r1)
    x_ = math.cos(rot_angle)*x - math.sin(rot_angle)*y
    y_ = math.sin(rot_angle)*x + math.cos(rot_angle)*y
    ellipsePoints.append((x_,y_))


ep = np.array(ellipsePoints)
fig = plt.figure(1, figsize=(4, 4))
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.scatter(ep[:,0], ep[:,1], s=2)

pca = decomposition.PCA(n_components=1)
ep_pca = pca.fit_transform(ep)

plt.plot([0,10*pca.components_[0][0]], [0 ,10*pca.components_[0][1]], 'k')

# code to plot the reprojected points
proj_ = ep_pca[0:20].squeeze()
proj = list()
for i in range(0, 20):
    proj.append((pca.components_[0][0] * proj_[i], pca.components_[0][1] * proj_[i]))

proj = np.array(proj)
plt.scatter(proj[:, 0], proj[:, 1], s=8, color='red')
plt.show()
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

X = pca.transform(X)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()