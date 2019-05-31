# %% [markdown]
# # 1. Clustering
# ## 1.1 Visualizing the data

# %%
import random

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from scipy import linalg

iris = np.genfromtxt("data/iris.txt", delimiter=None)
X, Y = iris[:, :2], iris[:, -1]
plt.scatter(iris[:, 0], iris[:, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title("Iris Dataset")
plt.show()

# %% [markdown]
# Possibly 4 clusters here.
# ## 1.2

# %%
random.seed(0)
initializations = [
    X[[0, 1], :],
    X[[37, 111], :],
    'random',
    'farthest',
    'k++'
]
k = 2

Z, C, S = 0, 0, 0
max_score = 0
for index, init in enumerate(initializations):
    (z, c, sumd) = ml.cluster.kmeans(X, k, init=init, max_iter=100)
    if sumd > max_score:
        max_score = sumd
        (Z, C, S) = (z, c, sumd)

    print("Score for assignment: #{}: {}".format(index+1, sumd))

# %% [markdown]
# They all seem to have the similar squared sum on distances. However, #4 and #5 seems to be the largest.

# %%

ml.plotClassify2D(None, X, Z)
plt.scatter(C[:, 0], C[:, 1], color='red', label='Cluster centers')
plt.title('k=2 means clustering with the best assignment as demonstrated above.')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# %%
initializations = [
    X[[0, 1, 2, 3, 4], :],
    X[[24, 48, 72, 96, 120], :],
    'random',
    'farthest',
    'k++'
]
k = 5

Z, C, S = 0, 0, 0
max_score = 0
for index, init in enumerate(initializations):
    (z, c, sumd) = ml.cluster.kmeans(X, k, init=init, max_iter=100)
    if sumd > max_score:
        max_score = sumd
        (Z, C, S) = (z, c, sumd)

    print("Score for assignment: #{}: {}".format(index+1, sumd))

# %% [markdown]
# Assignment #5 seems to be best assignment.

# %%

ml.plotClassify2D(None, X, Z)
plt.scatter(C[:, 0], C[:, 1], color='red', label='Cluster centers')
plt.title('k=5 means clustering with the best assignment as demonstrated above.')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# %%
initializations = [
    X[list(range(20)), :],
    X[list(range(0, 140, 7)), :],
    'random',
    'farthest',
    'k++'
]
k = 20

Z, C, S = 0, 0, 0
max_score = 0
for index, init in enumerate(initializations):
    (z, c, sumd) = ml.cluster.kmeans(X, k, init=init, max_iter=100)
    if sumd > max_score:
        max_score = sumd
        (Z, C, S) = (z, c, sumd)

    print("Score for assignment: #{}: {}".format(index+1, sumd))

# %% [markdown]
# Assignment #1 seems to be the best assignment.

# %%
ml.plotClassify2D(None, X, Z)
plt.scatter(C[:, 0], C[:, 1], color='red', label='Cluster centers')
plt.title('k=20 means clustering with the best assignment as demonstrated above.')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# %% [markdown]
# ## 1.3 Run agglomerative clustering on the data, using single linkage and then again using complete linkage.

# %%
for k in [2, 5, 20]:
    (z, _) = ml.cluster.agglomerative(X, k, method='min')
    ml.plotClassify2D(None, X, z)
    plt.title('Agglomerative Clustering For K = {} and Single Linkage'.format(k))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# %%
for k in [2, 5, 20]:
    (z, _) = ml.cluster.agglomerative(X, k, method='max')
    ml.plotClassify2D(None, X, z)
    plt.title('Agglomerative Clustering For K = {} and Complete Linkage'.format(k))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# %% [markdown]
# For k = 2, K means has more meaningful clusters. Complete linkage comes close to this but single linkage has very few data points in one cluster.
#
# For k = 5, Both single linkage agglomerative and complete linkage agglomerative have good clustering with complete linkage slightly better, while K means shows some extent of overfitting.
#
# For k = 20, K means shows clear overfitting while agglomerative shows better clustering.
#
# K means clusters outliers well. But agglomerative classifies these outliers as a new cluster.

# %% [markdown]
# # 2. EigenFaces

# %%
X = np.genfromtxt("data/faces.txt", delimiter=None)  # load face dataset
plt.figure()
img = np.reshape(X[6, :], (24, 24))
plt.imshow(img.T, cmap="gray")  # display image patch; you may have to squint


# %% [markdown]
# ## 2.1 Plot the mean

# %%
mean = np.mean(X, axis=0)
X0 = X - mean
img = np.reshape(mean, (24, 24))
plt.imshow(img.T, cmap="gray")
plt.title("Mean of the Face")
plt.show()

# %% [markdown]
# ## 2.2  SVD of the data

# %%
U, s, Vh = linalg.svd(X0, full_matrices=False)
W = np.dot(U, np.diag(s))
print("Shape of W: {}".format(W.shape))
print("Shape of Vh: {}".format(Vh.shape))

# %% [markdown]
# ## 2.3 For K = 1 . . . 10, compute the approximation to X0 given by the first K eigendirections.

# %%

K = range(0, 10, 1)
mse = [
    np.mean(
        (
            X0 - np.dot(W[:, :k], Vh[:k, :])
        )**2
    )
    for k in K
]

plt.plot(K, mse)
plt.xlabel('K')
plt.ylabel('MSE')
plt.title('MSE vs K')
plt.show()

# %% [markdown]
# ## 2.4 Display the first three principal directions of the data

# %%
for j in range(3):
    alpha = 2*np.median(np.abs(W[:, j]))
    img = np.reshape(mean+alpha*Vh[j, :], (24, 24))
    plt.subplot(1, 2, 1)
    plt.imshow(img.T, cmap='gray')
    plt.title('Principal Direction: {} (+alpha)'.format(j+1))

    img = np.reshape(mean-alpha*Vh[j, :], (24, 24))
    plt.subplot(1, 2, 2)
    plt.imshow(img.T, cmap='gray')
    plt.title('Principal Direction: {} (-alpha)'.format(j+1))
    plt.show()

# %% [markdown]
# ## 2.5 Reconstruct two images
# %%
for i in (10, 40):
    for k in (5, 10, 50, 100):
        plt.subplot(1, 2, 1)
        image = np.reshape(X[i, :], (24, 24))
        plt.title('Original Image #{}'.format(i))
        plt.imshow(image.T, cmap='gray')

        image = np.reshape(np.dot(W[i, :k], Vh[:k]) + mean, (24, 24))
        plt.subplot(1, 2, 2)
        plt.title('Reconstructed Image #{} (K = {})'.format(i, k))
        plt.imshow(image.T, cmap='gray')
        plt.show()

# %% [markdown]
# ## 2.6 Latent representation

# %%
idx = random.sample(range(0, X.shape[0]), 25)
coord, params = ml.transforms.rescale(W[:, 0:2])
for i in idx:
    loc = (
        coord[i, 0], coord[i, 0]+0.5, coord[i, 1],
        coord[i, 1]+0.5
    ) 
    img = np.reshape(X[i, :], (24, 24))
    plt.imshow(img.T, cmap="gray", extent=loc)  # draw each image
    plt.title('Latent Space Plot')
    plt.axis((-2, 2, -2, 2))
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

#%% [markdown]
# # Statement of collaboration
# I have not collaborated with anyone for this homework.