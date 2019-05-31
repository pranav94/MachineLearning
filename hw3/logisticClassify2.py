import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Fix the required "not implemented" functions for the homework ("TODO")

################################################################################
## LOGISTIC REGRESSION BINARY CLASSIFIER #######################################
################################################################################


class logisticClassify2(ml.classifier):
    """A binary (2-class) logistic regression classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for logisticClassify2 object.

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array
        """
        self.classes = [0,1]              # (default to 0/1; replace during training)
        self.theta = np.array([])         # placeholder value before training

        if len(args) or len(kwargs):      # if we were given optional arguments,
            self.train(*args,**kwargs)    #  just pass them through to "train"


## METHODS ################################################################

    def plotBoundary(self,X,Y,axis=None):
        """ Plot the (linear) decision boundary of the classifier, along with data """
        if len(self.theta) != 3: raise ValueError('Data & model must be 2D');
        ax = X.min(0),X.max(0); ax = (ax[0][0],ax[1][0],ax[0][1],ax[1][1]);
        ## TODO: find points on decision boundary defined by theta0 + theta1 X1 + theta2 X2 == 0
        x1b = np.array([ax[0],ax[1]]);  # at X1 = points in x1b
        (t0, t1, t2) = self.theta
        x2b = ( -np.array([t0, t0]) - t1 * x1b) / t2
        ## Now plot the data and the resulting boundary:
        A = Y==self.classes[0]; # and plot it:
        recs = [
            mpatches.Rectangle((0, 0), 1, 1, fc=c)
            for c in ['r', 'g']
        ]
        if not axis:
            plt.plot(X[A,0],X[A,1],'r.',X[~A,0],X[~A,1],'g.',x1b,x2b,'k-'); plt.axis(ax); plt.draw(); plt.legend(recs, self.classes)
        else:
            axis.plot(X[A,0],X[A,1],'r.',X[~A,0],X[~A,1],'g.',x1b,x2b,'k-'); axis.axis(ax); axis.legend(recs, self.classes)

    def predictSoft(self, X):
        """ Return the probability of each class under logistic regression """
        raise NotImplementedError
        ## You do not need to implement this function.
        ## If you *want* to, it should return an Mx2 numpy array "P", with
        ## P[:,1] = probability of class 1 = sigma( theta*X )
        ## P[:,0] = 1 - P[:,1] = probability of class 0
        return P

    def predict(self, X):
        """ Return the predictied class of each data point in X"""
        (t0, t1, t2) = self.theta
        g = lambda x: t0 + t1 * x[0] + t2 * x[1]
        return np.array([
            self.classes[1] if g(x) > 0 else self.classes[0]
            for x in X
        ])


    def train(self, X, Y, initStep=1.0, stopTol=1e-4, stopEpochs=5000, plot=None, alpha=0):
        """ Train the logistic regression using stochastic gradient descent """
        M,N = X.shape;                     # initialize the model if necessary:
        self.classes = np.unique(Y);       # Y may have two classes, any values
        XX = np.hstack((np.ones((M,1)),X)) # XX is X, but with an extra column of ones
        YY = ml.toIndex(Y,self.classes);   # YY is Y, but with canonical values 0 or 1
        if len(self.theta)!=N+1: self.theta=np.random.rand(N+1);
        sigma = lambda r: 1 / (1+np.exp(-r))
        # init loop variables:
        epoch=0; done=False; Jnll=[float('inf')]; J01=[float('inf')];
        fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
        recs = [
            mpatches.Rectangle((0, 0), 1, 1, fc=c)
            for c in ['b', 'r']
        ]
        plt.subplots_adjust(hspace=.7)
        while not done:
            stepsize, epoch = initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            Jsurr_i = 0
            # Do an SGD pass through the entire data set:
            for i in np.random.permutation(M):
                ri = np.dot(self.theta, XX[i, :])
                gradi = (-YY[i] + sigma(ri)) * XX[i, :] + alpha * self.theta
                self.theta -= stepsize * gradi;  # take a gradient step
                Jsurr_i += (
                    -YY[i] * np.log(sigma(np.dot(self.theta, XX[i, :]))) -
                    ((1-YY[i])*np.log(1-sigma(np.dot(self.theta, XX[i, :]))))
                )

            J01.append( self.err(X,Y) )  # evaluate the current error rate

            Jsur = Jsurr_i / M
            L2 = 0
            if alpha:
                L2 = alpha * sum(list(map(lambda x: x*x, gradi)))
            
            Jsur += L2
            Jnll.append(Jsur)
            ax1.plot(Jnll,'b-',J01,'r-')
            ax1.set_xlabel("Epoch")
            ax1.set_title("Convergence of Surrogate loss and Error rate")
            ax1.legend(recs, ["Surrogate loss", "Error rate"])
            if N==2:
                ax2.set_title("Convergence of classifier")
                self.plotBoundary(X,Y,ax2)
            plt.pause(.01);                    # let OS draw the plot

            ## For debugging: you may want to print current parameters & losses
            # print self.theta, ' => ', Jnll[-1], ' / ', J01[-1]
            # raw_input()   # pause for keystroke

            done = (
                epoch > stopEpochs or
                epoch > 1 and abs(Jnll[-2] - Jnll[-1]) < stopTol
            )

        ax3.set_title("Final classifier")
        self.plotBoundary(X, Y, ax3)

################################################################################
################################################################################
################################################################################
