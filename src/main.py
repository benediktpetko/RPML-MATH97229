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
        time :  terminal time of simulation

        steps :  number of discretization steps

        rep :   number of replications

        d_diffusion : derivative of diffusion coefficient
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


    def regression_ols(self, method="increments"):
        """
        Compute coefficients of linear regression with the increments or signature as features.
        Parameters:
        -----------
        method:  {"increments","signature","logsignature"}
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
            pass
        
        
        if method == "logsignature":
            pass

#%%
if __name__=="__main__":
    sim = sde()
    sim.compute_milstein_scheme()
    #sim.plot_paths()

    sim.regression_ols()
# %%
