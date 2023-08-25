# To Do Here:
#  - Add a "streaming" version of the time-series that simply returns the
#    next step (via a generator) rather than the whole time series

import numpy as np
import matplotlib.pyplot as plt

class TimeSeries:

    def __init__(self, cache_size=10000):
        self.cache = None
        self.max_cache = cache_size
        self.noise = np.random.normal

    def sample(self, num_samples, num_steps):
        self.cache = self.noise( size=(num_samples, num_steps) )
        return self.cache

    # Improve this to only do the extra number
    def sample_mean(self, *, min_samples=10000):
        steps = 64
        if self.cache is None or self.cache.shape[0] < min_samples:
            self.cache = self.sample( min_samples, steps ) 
        
        return np.mean(self.cache, axis=0)

    def sample_variance(self, *, min_samples=10000):
        steps = 64
        if self.cache is None or self.cache.shape[0] < min_samples:
            self.cache = self.sample( min_samples, steps )
            
        return np.var(self.cache, axis=0)

    def clear_cache(self):
        self.cache = None

    # Need to do a time average over sample_mean
    def mean(self):
        return
        
# To Do:
#   - Add option to start time-streams from some mean value, or sampled from some i.i.d. distribution
#   - Add some more convenience routines for computing lags, etc.
#   - Add some caching of previously generated time series
#   - Add option to specify the i.i.d. distribution samples are drawn from
class TimeSeries_AR(TimeSeries):
    """
    A class to generate time-stream from AR(p) models.
    """
    
    def __init__(self, ar_coeffs, phi0 : float):
        """
        Initialise an AR(p) time-series model 
          y_t = phi_0 + \sum_{l=1}^q w_l y_{t-q} + eta_t

        Inputs
        ------
          ar_coeffs : The coefficients w_l above
          phi0      : The coefficeint phi0 above
        """
        super().__init__()
        
        # Add some data checks here
        self.ar_coeffs = np.array(ar_coeffs)
        self.ar_ord = self.ar_coeffs.size
        self.phi0 = phi0

        self.is_stationary = np.sum(ar_coeffs) < 1. # Check this

    # Add an option to either start the series with some mean value
    # or else drawing from a distribution
    def sample(self, num_samples : int, num_steps : int, *,
               burn_in : int = 0,
               cache = False):
        p_ = self.ar_ord
        
        noise = self.noise( size=(num_samples, num_steps+burn_in) )
        series = np.zeros( (num_samples, num_steps+p_+burn_in) )

        if self.is_stationary:
            series[:,:p_] = self.mean()
        else:
            series[:,:p_] = 0.
            
            
        for i in range(num_steps+burn_in): 
            series[:,i+p_] = ( self.phi0 +
                               np.sum(series[:,i:i+p_]*self.ar_coeffs,axis=-1)
                               + noise[:,i]
                               )

        if cache:
            self.cache = series[:self.max_cache,p_+burn_in:]
            
        return series[:,p_+burn_in:] 

    def mean(self):
        """
        If the series is stationary, return the (analytically known) mean of the AR(p) series.

        Otherwise return NaN
        """
        if (self.is_stationary):
            return self.phi0 / (1.-np.sum(self.ar_coeffs))
        return np.nan
    
    
class TimeSeries_MA(TimeSeries):

    def __init__(self, ma_coeffs, mu):
        super().__init__()
        
        self.ma_coeffs = np.array(ma_coeffs)
        self.mu = mu
        self.ma_ord = ma_coeffs.size
        
        return

    def sample(self, num_samples: int, num_steps: int, *, mean_start=0.):
        q_ = self.ma_ord

        noise = self.noise( size=(num_samples, num_steps+q_) )
        series = np.zeros( (num_samples, num_steps+q_) )
        for i in range(num_steps):
            series[:,i+q_] = mu + np.sum(series[:,i:i+q_]*self.ma_coeffs) + noise[:,i]

        return series

    def mean(self):
        return self.mu

class TimeSeries_ARMA(TimeSeries):

    def __init__(self, ma_coeffs, ar_coeffs, phi0: float):
        super().__init__()
        
        self.ma_coeffs = np.array(ma_coeffs)
        self.ar_coeffs = np.array(ar_coeffs)
        
        self.ma_ord = self.ma_coeffs.size
        self.ar_ord = self.ar_coeffs.size
        self.phi0 = phi0

        return

    def sample(self, num_samples: int,
               num_steps: int, *,
               burn_in : int = 0,
               cache = False):
        p_, q_ = self.ar_ord, self.ma_ord
        
        noise = self.noise( size=(num_samples, num_steps+q_+burn_in) )
        series = np.zeros( (num_samples, num_steps+p_+burn_in) )

        series[:,:p_] = self.mean()  # Adjust this
        
        for i in range(num_steps+burn_in):
            series[:,i+p_] = ( self.phi0 +
                                np.sum(self.ar_coeffs*series[:,i:i+p_],axis=-1) +
                                np.sum(self.ma_coeffs*noise[:,i:i+q_],axis=-1) +
                                noise[:,i]
                                )

        if cache:
            self.cache = series[:self.max_cache,p_+burn_in:]
            
        return series[:,p_+burn_in:]

    def mean(self):
        return self.phi0 / (1.-np.sum(self.ar_coeffs))

    
if __name__=="__main__":
    ARMA = TimeSeries_ARMA([0.3,0.4, 0.5], [0.1,0.3], 1.7)
    s = ARMA.sample(10000,256)
