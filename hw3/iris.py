# %% [markdown]
# # Problem 1: Logistic Regression
# %%
import numpy as np
import matplotlib.pyplot as pt
import mltools as ml
import matplotlib.patches as mpatches

iris = np.genfromtxt("data/iris.txt", delimiter=None)
X, Y = iris[:, 0:2], iris[:, -1]
X, Y = ml.shuffleData(X, Y)
X, _ = ml.rescale(X)
XA, YA = X[Y < 2, :], Y[Y < 2]
XB, YB = X[Y > 0, :], Y[Y > 0]

# %% [markdown]
# ## 1. Scatter plots
colors = ['r' if y == 0 else 'g' for y in YA]
pt.scatter(XA[:, 0], XA[:, 1], color=colors)
pt.title("Class 0 vs Class 1")
pt.xlabel("Feature 1")
pt.ylabel("Feature 2")
recs = [
    mpatches.Rectangle((0, 0), 1, 1, fc=c)
    for c in ['r', 'g']
]
pt.legend(recs, ["Class 0", "Class 1"])
pt.show()

colors = ['red' if y == 1 else 'green' for y in YB]
pt.scatter(XB[:, 0], XB[:, 1], color=colors)
pt.xlabel("Feature 1")
pt.ylabel("Feature 2")
pt.title("Class 1 vs Class 2")
pt.legend(recs, ["Class 1", "Class 2"])
pt.show()

# %% [markdown]
# ## 2. Plot Boundary

# %%
from logisticClassify2 import *

learnerA = logisticClassify2()
learnerA.classes = np.unique(YA)
wts = np.array([.5, -.25, 1])
learnerA.theta = wts

pt.title("Class 0 vs Class 1")
learnerA.plotBoundary(XA, YA)
pt.legend(recs, ["Class 0", "Class 1"])
pt.show()

learnerB = logisticClassify2()
learnerB.classes = np.unique(YB)
learnerB.theta = wts

pt.title("Class 1 vs Class 2")
learnerB.plotBoundary(XB, YB)
pt.legend(recs, ["Class 1", "Class 2"])
pt.show()

# %% [markdown]
# ### Plot Boundary Code
# ```python
# def plotBoundary(self,X,Y):
#     if len(self.theta) != 3: raise ValueError('Data & model must be 2D');
#     ax = X.min(0),X.max(0); ax = (ax[0][0],ax[1][0],ax[0][1],ax[1][1]);
#     ## TODO: find points on decision boundary defined by theta0 + theta1 X1 + theta2 X2 == 0
#     x1b = np.array([ax[0],ax[1]]);  # at X1 = points in x1b
#     (t0, t1, t2) = self.theta
#     x2b = ( -np.array([t0, t0]) - t1 * x1b) / t2
#     ## Now plot the data and the resulting boundary:
#     A = Y==self.classes[0]; # and plot it:
#     plt.plot(X[A,0],X[A,1],'r.',X[~A,0],X[~A,1],'g.',x1b,x2b,'k-'); plt.axis(ax); plt.draw();
# ```

# %% [markdown]
# ## 3. Predict
# ```python
# def predict(self, X):
#     (t0, t1, t2) = self.theta
#     g = lambda x: t0 + t1 * x[0] + t2 * x[1]
#     return [
#         self.classes[1] if g(x) > 0 else self.classes[0]
#         for x in X
#     ]
# ```
# %%
errTrA = learnerA.err(XA, YA)
errTrB = learnerB.err(XB, YB)
print("Learner A Training error: {}".format(errTrA))
print("Learner B Training error: {}".format(errTrB))

# %% [markdown]
# From above, it is clear that,
# #### Learner A Training error: ~0.05
# #### Learner B Training error: ~0.46

# %% [markdown]
# ## 4. Plot Classify 2D

# %%
pt.title("Learner A")
ml.plotClassify2D(learnerA, XA, YA)
pt.show()

pt.title("Learner B")
ml.plotClassify2D(learnerB, XB, YB)
pt.show()

# %% [markdown]
# It is clear that this decision boundary matches our analytical decision boundary.

# %% [markdown]
# ## 5. Gradient Equation
# <img src="q5.jpg"/>

# %% [markdown]
# ## 6. Stochastic Gradient Descent train implementation
# Train Code
# ```python
# def train(self, X, Y, initStep=1.0, stopTol=1e-4, stopEpochs=5000, plot=None):
#     M,N = X.shape;                     # initialize the model if necessary:
#     self.classes = np.unique(Y);       # Y may have two classes, any values
#     XX = np.hstack((np.ones((M,1)),X)) # XX is X, but with an extra column of ones
#     YY = ml.toIndex(Y,self.classes);   # YY is Y, but with canonical values 0 or 1
#     if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
#     sigma = lambda r: 1 / (1+np.exp(-r))
#     # init loop variables:
#     epoch=0; done=False; Jnll=[float('inf')]; J01=[float('inf')];
#     while not done:
#         stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
#         Jsurr_i = 0
#         # Do an SGD pass through the entire data set:
#         for i in np.random.permutation(M):
#             ri = np.dot(self.theta, XX[i, :])
#             gradi = (-YY[i] + sigma(ri)) * XX[i, :]
#             self.theta -= stepsize * gradi;  # take a gradient step
#             Jsurr_i += (
#                 -YY[i] * np.log(sigma(np.dot(self.theta, XX[i, :]))) -
#                 ((1-YY[i])*np.log(1-sigma(np.dot(self.theta, XX[i, :]))))
#             )

#         J01.append( self.err(X,Y) )  # evaluate the current error rate

#         Jsur = Jsurr_i / M
#         Jnll.append(Jsur)
#         plt.figure(1); plt.plot(Jnll,'b-',J01,'r-'); plt.draw();    # plot losses
#         if N==2: plt.figure(2); self.plotBoundary(X,Y); plt.draw(); # & predictor if 2D
#         plt.pause(.01);                    # let OS draw the plot

#         ## For debugging: you may want to print current parameters & losses
#         # print self.theta, ' => ', Jnll[-1], ' / ', J01[-1]
#         # raw_input()   # pause for keystroke

#         done = (
#             epoch > stopEpochs or
#             abs(Jnll[-2] - Jnll[-1]) < stopTol
#         )
# ```

# %% [markdown]
# ## 7. Run train

# %%
from logisticClassify2 import *

learnerA = logisticClassify2()
learnerA.theta = np.array([.5, -.25, 1])

learnerA.train(XA, YA, initStep=.5)

pt.close()
learnerB = logisticClassify2()
learnerB.theta = np.array([.5, -.25, 1])

learnerB.train(XB, YB, initStep=.5)