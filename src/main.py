# main.py

import numpy as np
import scipy as sp
import pandas as pd
import iisignature as iisig


class milstein_scheme(object):
    """
    Class for numerical computation of an Ito diffusion by the Milstein scheme.
    Parameters:
    func    
    """

    def __init__(self, func=lambda x,y: (1-x, y**2) , time=1, steps=100):
        pass
        
    pass


class regression_ols(object):
    pass

if __name__=="__main__":
    W = np.random.randn(250,1600)
    T = 0.25
    K= 250.0
    dt = T/K


    pass