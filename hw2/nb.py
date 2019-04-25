# %% [markdown]
# # Problem 1: Linear regression
# ## 1. Print the shapes of these four objects.

# %%
import numpy as np
import matplotlib.pyplot as pt
import mltools as ml


data = np.genfromtxt("data/curve80.txt")
X = data[:, 0]
X = np.atleast_2d(X).T
Y = data[:, 1]
Xtr, Xte, Ytr, Yte = ml.splitData(X, Y, 0.75)

print("Xtr Shape: ", Xtr.shape)
print("Xte Shape: ", Xte.shape)
print("Ytr Shape: ", Ytr.shape)
print("Yte Shape: ", Yte.shape)

# %% [markdown]
# ## 2. Creating a linear regression predictor
# a) Creating a linear regression predictor
# %%
lr = ml.linear.linearRegress(Xtr, Ytr)
xs = np.linspace(0, 10, 200)
xs = xs[:, np.newaxis]
ys = lr.predict(xs)

pt.scatter(Xtr, Ytr, color="red", label="Training")
pt.plot(xs, ys, color="blue", label="Prediction function")
pt.xlabel("X values")
pt.ylabel("Y label")
pt.show()

print("Linear regression co-efficients:")
print(lr.theta)

print("Co-efficients obtained from the predicted values:")
print(ys[0], (ys[199]-ys[0])/(xs[199]-xs[0]))

# %% [markdown]
# b) From above it is clear that the regression co-efficients match our plot.

# %%


def mse(X, Y, lr):
    Yh = lr.predict(X)
    Ys = np.atleast_2d(Y).T
    e = Ys - Yh
    mse = e.T.dot(e) / X.shape[0]
    return np.squeeze(mse)


print('MSE on Training Data')
print(mse(Xtr, Ytr, lr))
print('MSE on Test Data')
print(mse(Xte, Yte, lr))

# %% [markdown]
# ## 3. Fitting y = f(x) using a polynomial function f(x) of increasing order.
# ###  a) Plot their learned prediction functions f(x)
# %%

Xtr2 = np.zeros((Xtr.shape[0], 2))
Xtr2[:, 0] = Xtr[:, 0]
Xtr2[:, 1] = Xtr[:, 0]**2

lr = ml.linear.linearRegress(Xtr2, Ytr)
xs = np.linspace(0, 10, 200)
xs = xs[:, np.newaxis]

# %%


def Phi(X, degree, params):
    return ml.transforms.rescale(ml.transforms.fpoly(X, degree, False), params)[0]


degrees = [1, 3, 5, 7, 10, 18]
errTrain = [0] * len(degrees)
errTest = [0] * len(degrees)

for index, degree in enumerate(degrees):
    XtrP = ml.transforms.fpoly(Xtr, degree, False)
    XtrP, params = ml.transforms.rescale(XtrP)
    lr = ml.linear.linearRegress(XtrP, Ytr)
    XteP = Phi(Xte, degree, params)
    xsP = Phi(xs, degree, params)
    YsHat = lr.predict(xsP)
    _fig, plot = pt.subplots(1, 1, figsize=(10, 8))
    plot.set_xlabel("X")
    plot.set_ylabel("Y")
    plot.scatter(Xtr, Ytr, color="red", label="Training")
    plot.scatter(Xte, Yte, color="green", label="Testing")
    plot.plot(
        xs, YsHat, color="blue",
        label="Prediction function for degree {}".format(degree),
        marker="."
    )
    plot.set_xlim(0, 10)
    plot.set_ylim(-5, 10)
    plot.set_title("Prediction function for degree {}".format(degree))
    plot.legend()
    errTrain[index] = mse(XtrP, Ytr, lr)
    errTest[index] = mse(XteP, Yte, lr)

pt.show()

# %% [markdown]
# ###  b) Plot their training and test errors on a log scale

# %%
pt.ylabel("Mean Squared Error")
pt.xlabel("Degree")
pt.semilogy(
    degrees, errTrain, 'red', marker='x',
    label='Training set MSE'
)
pt.semilogy(
    degrees, errTest, 'green', marker='o',
    label='Validation set MSE'
)
pt.legend()
pt.show()

# %% [markdown]
# ### c) A ploynomial of degree 10 is ideal looking at the graph above. MSE for test data is minimal when degree = 10.

# # Problem 2: Cross-validation
# ### a) Plot the five-fold cross-validation error and test error.
# %%
errCross = [0] * len(degrees)
errTest = [0] * len(degrees)
nFolds = 5
for index, degree in enumerate(degrees):
    errorCrossValidation = [0] * nFolds
    errCrossValidationTesting = [0] * nFolds
    for iFold in range(nFolds):
        Xti, Xvi, Yti, Yvi = ml.crossValidate(Xtr, Ytr, nFolds, iFold)
        XtiP = ml.transforms.fpoly(Xti, degree, bias=False)
        XtiP, params = ml.transforms.rescale(XtiP)
        XviP = Phi(Xvi, degree, params)
        XteP = Phi(Xte, degree, params)
        lr = ml.linear.linearRegress(XtiP, Yti)
        errorCrossValidation[iFold] = mse(XviP, Yvi, lr)
        errCrossValidationTesting[iFold] = mse(XteP, Yte, lr)
    errCross[index] = np.mean(errorCrossValidation)
    errTest[index] = np.mean(errCrossValidationTesting)

pt.ylabel("Mean Squared Error")
pt.xlabel("Degree")
pt.semilogy(
    degrees, errCross, 'red', marker='x',
    label='Cross-validation MSE'
)
pt.semilogy(
    degrees, errTest, 'green', marker='o',
    label='Testing set MSE'
)
pt.legend()
pt.show()
#%% [markdown]
# ### b) How do the MSE estimates from five-fold cross-validation compare to the MSEs evaluated on the actual test data
# For smaller degrees, MSE from both tends to remain approximately the same. But as the degree increases, MSE from the test data 
# %% [markdown]
# # Statement of collaboration
# I have not collaborated with anyone for this homework and have maintained the UCI code of honesty.
