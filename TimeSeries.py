# To Do Here:
#  - Add a "streaming" version of the time-series that simply returns the
#    next step (via a generator) rather than the whole time series

import numpy as np
import matplotlib.pyplot as plt

# Add option to start the series with random noise
class TimeSeries_AR():

    def __init__(self, ar_coeffs=[0.3,0.7], phi0=0.3):
        self.noise = np.random.normal

        self.ar_coeffs = np.array(ar_coeffs)
        self.ar_ord = self.ar_coeffs.size
        self.phi0 = phi0

        self.is_stationary = np.sum(ar_coeffs) < 1. # Check this

    # Add an option for how to start the samples here
    # Add an option for burn-in here
    def sample(self,num_samples, num_steps, *, mean_start=0., burn_in=0):
        p_ = self.ar_ord
        
        noise = self.noise( size=(num_samples, num_steps+burn_in) )
        series = np.zeros( (num_samples, num_steps+p_+burn_in) )
        series[:,:p_] = mean_start
        
        for i in range(num_steps+burn_in): 
            series[:,i+p_] = np.sum(series[:,i:i+p_]*self.ar_coeffs) + noise[:,i]
        return series[:,p_+burn_in:] 

    def mean(self):
        """
        If the series is stationary, return the (analytically known) mean of the AR(p) series.

        Otherwise return NaN
        """
        if (self.is_stationary):
            return self.phi0 / (1.-np.sum(ar_coeffs))
        return np.nan

class TimeSeries_MA():

    def __init__(self, ma_ord, ma_coeffs, ):
    
if __name__=="__main__":
    pass
