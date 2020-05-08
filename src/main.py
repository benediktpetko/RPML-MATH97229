# main.py
#%%
import numpy as np
import scipy as sp
import pandas as pd
import iisignature as iisig
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error





#%%
class sde(object):
    """
    Class for numerical computation of an Ito diffusion by the Milstein scheme and linear regression on increments and signature.
    Parameters:
    --------
    drift:    drift coefficient
        
    diffusion: diffusion coefficient

    time:    terminal time of the simulation

    steps:   number of discretization steps
    """

    def __init__(self, drift = lambda y:  1 - y, diffusion = lambda y: y**2, ydim = 1, y_0 = 0):
        self.drift = drift
        self.diffusion = diffusion
        self.ydim = ydim
        self.y_0 = y_0
    
    def compute_milstein_scheme(self, time = 0.25, steps = 250, rep = 1600, d_diffusion = lambda y: 2*y):
        """
        Simulate according to the Milstein scheme. Creates attribute sde.milstein
        Parameters:
        ----------
        time :  float
            terminal time of simulation
        steps :  int
            number of discretization steps
        rep :   int
            number of replications
        d_diffusion : function 
            derivative of diffusion coefficient
        """
        self.steps = steps
        self.d_diffusion = d_diffusion
        self.rep =rep

        dt = time / steps   # time step
        self.dt = dt
        
        W = np.sqrt(dt)*np.random.randn(steps, rep)    # generate Wiener noise

        y = np.zeros((steps + 1, rep))
        y[0,:] = self.y_0

        for i in range(steps):      #iterations of the Milstein scheme
            y[i+1,:] = y[i,:] + self.drift(y[i,:]) * self.dt + self.diffusion(y[i,:]) * W[i,:] + 1.0/2 * self.diffusion(y[i,:]) * self.d_diffusion(y[i,:]) * (W[i,:] ** 2) #- dt)
        
        self.milstein = y
        self.noise = W

    def plot_paths(self):
        plt.plot(np.arange(self.steps+1)*self.dt, self.milstein)
        plt.show()


    def regression_ols(self, method="increments", level=2):
        """
        Compute least squares errors of linear regression with the increments or signature as features.
        Parameters:
        -----------
        method:  string
            {"increments","signature","logsignature"}
        level: int
            truncate the signature at given level
        """

        if method == "increments":
            model = LinearRegression()

            features = np.concatenate((np.transpose([np.ones(self.rep)*self.dt]), np.transpose(self.noise)), axis = 1)
            target = self.milstein[self.steps,:]

            X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.5)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            err = mean_squared_error(y_pred, y_test)
            print("Mean squared error: ", err)

        if method == "signature":
            number_of_features = 2 ** (level +1 ) - 2 

            signature_features = np.zeros((self.rep, number_of_features))

            self.time_col = np.transpose([np.cumsum(np.ones(self.steps)*self.dt)])

            for i in range(self.rep):
                path = np.concatenate((self.time_col, np.transpose([np.cumsum(self.noise[:,i])])), axis = 1)

                signature_features[i,:] = iisig.sig(path, level)
            
            self.signature_features = signature_features
        
        
        if method == "logsignature":
            pass

#%%
if __name__=="__main__":
    sim = sde()
    sim.compute_milstein_scheme()
    #sim.plot_paths()

    sim.regression_ols(method="signature")
# %%
