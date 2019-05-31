# %% [markdown]
# # 2 Setting up the data

# %%
import numpy as np
import matplotlib.pyplot as pt
import mltools as ml
import random

random.seed(0)
X = np.genfromtxt("data/X_train.txt")
Y = np.genfromtxt("data/Y_train.txt")

X, Y = ml.shuffleData(X, Y)

# %% [markdown]
# ## 2.1 . Print the minimum, maximum, mean, and the variance of all of the features.

# %%

minimum = [
    min(X[:, feature])
    for feature in range(X.shape[1])
]

maximum = [
    max(X[:, feature])
    for feature in range(X.shape[1])
]

mean = [
    np.mean(X[:, feature])
    for feature in range(X.shape[1])
]

variance = [
    np.var(X[:, feature])
    for feature in range(X.shape[1])
]

print("Minimum of the featurs: \n{}\n".format(minimum))
print("Maximum of the featurs: \n{}\n".format(maximum))
print("Mean of the featurs: \n{}\n".format(mean))
print("Variance of the featurs: \n{}\n".format(variance))

# %% [markdown]
# # 2.2 Split the dataset, and rescale each into training and validation.

# %%
Xtr, Xva, Ytr, Yva = ml.splitData(X, Y)
Xt, Yt = Xtr[:5000], Ytr[:5000]  # subsample for efficiency (you can go higher)
XtS, params = ml.rescale(Xt)  # Normalize the features
XvS, _ = ml.rescale(Xva, params)  # Normalize the features

minimum = [
    min(XtS[:, feature])
    for feature in range(XtS.shape[1])
]

maximum = [
    max(XtS[:, feature])
    for feature in range(XtS.shape[1])
]

mean = [
    np.mean(XtS[:, feature])
    for feature in range(XtS.shape[1])
]

variance = [
    np.var(XtS[:, feature])
    for feature in range(XtS.shape[1])
]

print("Minimum of the featurs: \n{}\n".format(minimum))
print("Maximum of the featurs: \n{}\n".format(maximum))
print("Mean of the featurs: \n{}\n".format(mean))
print("Variance of the featurs: \n{}\n".format(variance))

# %% [markdown]
# # 3. Linear Classifiers
# %%
learner = ml.linearC.linearClassify()
learner.train(XtS, Yt, reg=0.0, initStep=0.5, stopTol=1e-6, stopIter=100)
learner.auc(XtS, Yt)  # train AUC

# %% [markdown]
# ## 3.1 Regularization
# %%

reg = np.arange(0.0, 10.0, 0.5)
auc_val, auc_tr = [], []

for r in reg:
    learner.train(XtS, Yt, reg=r, initStep=0.5, stopTol=1e-6, stopIter=100)
    auc_tr.append(learner.auc(XtS, Yt))
    auc_val.append(learner.auc(XvS, Yva))

pt.plot(reg, auc_tr, "r", label="Training UAC")
pt.plot(reg, auc_val, "b", label="Validation AUC")
pt.xlabel("Regularization")
pt.ylabel("AUC")
pt.legend()
pt.title("Plot of AUC vs Regularization parameter")
pt.show()

# %% [markdown]
# ## 3.2 Adding a 2nd degree polynomial
XtsP = ml.transforms.fpoly(XtS, 2, bias=False)
print("Number of features before: {}".format(XtS.shape[1]))
print("Number of features after adding a 2nd degree polynomialization: {}".format(
    XtsP.shape[1]))

# %% [markdown]
# Total number of features before = 14.
#
# The new features will consist of all these $14$ old features.
#
# Additionally, we will now have all combinations of these features in our second degree polynomial as well.
#
# Hence, $+^{14}C_2=91$ features.
#
# Finally, we also have $+14$ new features where each feature is a square of the corresponding old feature.
#
# Total new features = $14+91+14$
# $=119 features$

# %% [markdown]
# ## 3.3 AUC performance with second degree polynomial.

# %%
XtP = ml.transforms.fpoly(Xt, 2, bias=False)
XvP = ml.transforms.fpoly(Xva, 2, bias=False)
XtPS, params = ml.rescale(XtP)
XvPS, _ = ml.rescale(XvP, params)

auc_val, auc_tr = [], []

for r in reg:
    learner = ml.linearC.linearClassify()
    learner.train(XtPS, Yt, reg=r, initStep=0.5, stopTol=1e-6, stopIter=100)
    auc_tr.append(learner.auc(XtPS, Yt))
    auc_val.append(learner.auc(XvPS, Yva))

pt.plot(reg, auc_tr, "r", label="Training UAC")
pt.plot(reg, auc_val, "b", label="Validation AUC")
pt.xlabel("Regularization")
pt.ylabel("AUC")
pt.legend()
pt.title("Plot of AUC vs Regularization parameter")
pt.show()

# %% [markdown]
# # 4 Nearest Neighbors

# %%
learner = ml.knn.knnClassify()
learner.train(XtS, Yt, K=1, alpha=0.0)
learner.auc(XtS, Yt)  # train AUC

# %% [markdown]
# ## 4.1 Plot of the training and validation performance for an appropriately wide range of K, with α = 0.

# %%
K = list(range(1, 10, 2)) + list(range(10, 100, 10))
auc_val, auc_tr = [], []

for k in K:
    learner = ml.knn.knnClassify()
    learner.train(XtS, Yt, K=int(k), alpha=0.0)
    auc_tr.append(learner.auc(XtS, Yt))
    auc_val.append(learner.auc(XvS, Yva))

pt.plot(K, auc_tr, "r", label="Training UAC")
pt.plot(K, auc_val, "b", label="Validation AUC")
pt.xlabel("K")
pt.ylabel("AUC")
pt.legend()
pt.title("Plot of AUC vs the size of the neighborhood(K)")
pt.show()

# %% [markdown]
# ## 4.2 Unscaled/original data

# %%
K = list(range(1, 10, 2)) + list(range(10, 100, 10))
auc_val, auc_tr = [], []

for k in K:
    learner = ml.knn.knnClassify()
    learner.train(Xt, Yt, K=k, alpha=0.0)
    auc_tr.append(learner.auc(Xt, Yt))
    auc_val.append(learner.auc(Xva, Yva))

pt.plot(K, auc_tr, "r", label="Training UAC")
pt.plot(K, auc_val, "b", label="Validation AUC")
pt.xlabel("K")
pt.ylabel("AUC")
pt.legend()
pt.title("Plot of AUC vs the size of the neighborhood(K)")
pt.show()

# %% [markdown]
# ## 4.3 Select both the value of K and α

# %%
K = range(1, 50, 5)
A = range(0, 5, 1)
tr_auc = np.zeros((len(K), len(A)))
va_auc = np.zeros((len(K), len(A)))

for i, k in enumerate(K):
    for j, a in enumerate(A):
        learner = ml.knn.knnClassify()
        learner.train(Xt, Yt, K=k, alpha=a)
        tr_auc[i][j] = learner.auc(Xt, Yt)
        va_auc[i][j] = learner.auc(Xva, Yva)

f, ax = pt.subplots(1, 1, figsize=(8, 5))
cax = ax.matshow(tr_auc, interpolation='nearest')
f.colorbar(cax)
ax.set_xticklabels(['']+list(A))
ax.set_yticklabels(['']+list(K))
pt.ylabel("K")
pt.xlabel("A")
pt.title("Training data AUC")
pt.show()

f, ax = pt.subplots(1, 1, figsize=(8, 5))
cax = ax.matshow(va_auc, interpolation='nearest')
f.colorbar(cax)
ax.set_xticklabels(['']+list(A))
ax.set_yticklabels(['']+list(K))
pt.ylabel("K")
pt.xlabel("A")
pt.title("Validation data AUC")
pt.show()

# %% [markdown]
# # 5. Decision Trees
# ## 5.1 Vary maxDepth to a range of your choosing, and plot the training and validation

# %%
max_depths = list(range(1, 50, 3))
tr_auc, va_auc = [], []

for max_depth in max_depths:
    learner = ml.dtree.treeClassify(
        XtS, Yt, minParent=2, minLeaf=1, maxDepth=max_depth
    )
    tr_auc.append(learner.auc(XtS, Yt))
    va_auc.append(learner.auc(XvS, Yva))

pt.plot(max_depths, tr_auc, "r", label="Training AUC")
pt.plot(max_depths, va_auc, label="Validation AUC")
pt.xlabel("Max depth")
pt.ylabel("AUC")
pt.legend()
pt.show()

# %% [markdown]
# ## 5.2 Plot the number of nodes in the tree as maxDepth is varied.

# %%
l1, l2 = [], []

for max_depth in max_depths:
    learner1 = ml.dtree.treeClassify(
        XtS, Yt, minParent=2, minLeaf=1, maxDepth=max_depth
    )
    learner2 = ml.dtree.treeClassify(
        XtS, Yt, minParent=2, minLeaf=4, maxDepth=max_depth
    )
    l1.append(learner1.sz)
    l2.append(learner2.sz)

pt.plot(max_depths, l1, "r", label="Min leaf = 1")
pt.plot(max_depths, l2, label="Min leaf = 4")
pt.xlabel("Max depth")
pt.ylabel("Number of nodes")
pt.legend()
pt.show()

# %% [markdown]
# ## 5.3 Recommend a choice for minParent and minLeaf

# %%
min_parents = range(1, 20, 4)
min_leaves = range(1, 10, 2)
tr_auc = np.zeros((len(min_parents), len(min_leaves)))
va_auc = np.zeros((len(min_parents), len(min_leaves)))

for i,  min_parent in enumerate(min_parents):
    for j, min_leaf in enumerate(min_leaves):
        learner = ml.dtree.treeClassify(
            XtS, Yt,
            minParent=min_parent, minLeaf=min_leaf, maxDepth=30
        )
        tr_auc[i][j] = learner.auc(XtS, Yt)
        va_auc[i][j] = learner.auc(XvS, Yva)

f, ax = pt.subplots(1, 1, figsize=(8, 5))
cax = ax.matshow(tr_auc, interpolation='nearest')
f.colorbar(cax)
ax.set_xticklabels(['']+list(min_leaves))
ax.set_yticklabels(['']+list(min_parents))
pt.ylabel("Minimum parent")
pt.xlabel("Minimum leaf")
pt.title("Training data AUC")
pt.show()

f, ax = pt.subplots(1, 1, figsize=(8, 5))
cax = ax.matshow(va_auc, interpolation='nearest')
f.colorbar(cax)
ax.set_xticklabels(['']+list(min_leaves))
ax.set_yticklabels(['']+list(min_parents))
pt.ylabel("Minimum parent")
pt.xlabel("Minimum leaf")
pt.title("Validation data AUC")
pt.show()

# %% [markdown]
# I would recommend Minimum parent as 1 and Minimum leaf as 7.

# %% [markdown]
# # 6. Neural Networks
# ## 6.1 Vary the number of hidden layers and the nodes in each layer
# %%
layers, nodes = range(1, 6), range(2, 8)
tr_auc = np.zeros((len(layers), len(nodes)))
va_auc = np.zeros((len(layers), len(nodes)))
for i, layer in enumerate(layers):
    for j, node in enumerate(nodes):
        nn = ml.nnet.nnetClassify()
        nn.init_weights([XtS.shape[1], 5, 2], 'random', XtS, Yt)
        nn.train(XtS, Yt, stopTol=1e-8, stepsize=.25, stopIter=300)
        tr_auc[i][j] = nn.auc(XtS, Yt)
        va_auc[i][j] = nn.auc(XvS, Yva)


f, ax = pt.subplots(1, 1, figsize=(8, 5))
cax = ax.matshow(tr_auc, interpolation='nearest')
f.colorbar(cax)
ax.set_xticklabels(['']+list(nodes))
ax.set_yticklabels(['']+list(layers))
pt.ylabel("Layer")
pt.xlabel("Node")
pt.title("Training data AUC")
pt.show()

f, ax = pt.subplots(1, 1, figsize=(8, 5))
cax = ax.matshow(va_auc, interpolation='nearest')
f.colorbar(cax)
ax.set_xticklabels(['']+list(nodes))
ax.set_yticklabels(['']+list(layers))
pt.ylabel("Layer")
pt.xlabel("Node")
pt.title("Validation data AUC")
pt.show()

#%% [markdown]
# ## 6.2 Implement a custom activation function
def sig(x):
    return np.atleast_2d(np.exp(x**2/2))

def dsig(x):
    return np.atleast_2d(-x*np.exp(x**2/2))

nn = ml.nnet.nnetClassify()
nn.init_weights([XtS.shape[1], 5, 2], 'random', XtS, Yt)
nn.setActivation('custom', sig, dsig)
nn.train(XtS, Yt, stopTol=1e-8, stepsize=.25, stopIter=300)
print("Gaussian Training AUC: {}".format(nn.auc(XtS, Yt)))
print("Gaussian Validation AUC: {}".format(nn.auc(XvS, Yva)))


nn = ml.nnet.nnetClassify()
nn.init_weights([XtS.shape[1], 5, 2], 'random', XtS, Yt)
nn.setActivation('logistic')
nn.train(XtS, Yt, stopTol=1e-8, stepsize=.25, stopIter=300)
print("Logistic Training AUC: {}".format(nn.auc(XtS, Yt)))
print("Logistic Validation AUC: {}".format(nn.auc(XvS, Yva)))

nn = ml.nnet.nnetClassify()
nn.init_weights([XtS.shape[1], 5, 2], 'random', XtS, Yt)
nn.setActivation('htangent')
nn.train(XtS, Yt, stopTol=1e-8, stepsize=.25, stopIter=300)
print("HTangent Training AUC: {}".format(nn.auc(XtS, Yt)))
print("HTangent Validation AUC: {}".format(nn.auc(XvS, Yva)))

#%% [markdown]
# Gaussian Training AUC: 0.49458635778635773
# 
# Gaussian Validation AUC: 0.496823807293988
# 
# 
# 
# Logistic Training AUC: 0.649433462033462
# 
# Logistic Validation AUC: 0.6483055317573599
# 
# 
# 
# HTangent Training AUC: 0.6832237666237666
# 
# HTangent Validation AUC: 0.6660746786034655
# 
# For both Training and Validation AUC, htangent seems to be the best activation function.

#%% [markdown]
# ## Statement of Collaboration
# I have not collaborated with anyone for this homework.