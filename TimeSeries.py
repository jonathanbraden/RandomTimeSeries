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
               burn_in  = 0,
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

# An improvement to this is to put the noise generation separately
class TimeSeries_GARCH(TimeSeries):

    def __init__(self, ch_coeffs, a0: float):
        assert a0>0, "a0 must be positive"
        assert min(ch_coeffs)>=0, "All coefficients must be non-negative"

        super().__init__()

        self.ch_coeffs = np.array(ch_coeffs)
        self.a0 = a0
        self.ch_ord = self.ch_coeffs.size
        return 

    def sample_noise(self, num_samples: int, num_steps: int, *,
                     burn_in: int = 0):
        ord = self.ch_ord
        
        noise = self.noise( size=(num_samples, num_steps+ord) )
        var = np.zeros( (num_samples, num_steps+ord+burn_in) )
        var[:,:ord] = self.a0

        for i in range(num_steps+burn_in):
            var[:,i] = (self.a0 +
                        np.sum(self.ch_coeffs*noise[:,i:i+ord]**2,axis=-1)
                        )
            noise[:,i] = np.sqrt(var[:,i])*noise[:,i]

        return noise[:,ord+burn_in:]

class TimeSeries_Brownian(TimeSeries):

    def __init__(self, mu: float, sigma: float):
        super().__init__()

        self.noise = np.random.normal
        self.sigma = sigma
        self.mu = mu
        return

    def __str__(self):
        return "Generalised Brownian Motion"
    
    def sample(self, num_samples: int, num_steps: int, dt: float):
        series = np.sqrt(dt)*self.sigma*self.noise( size=(num_samples, num_steps+1) )
        series[:,0] = 0.
        series = np.cumsum(series, axis=-1) + self.mu*dt*np.arange(num_steps+1)
        return series

    def mean(self, time: float):
        return self.mu*time

    def std(self, time: float):
        return self.sigma*np.sqrt(time)

    def distribution(self, time:float):
        s = np.sqrt(time)*self.sigma
        m = time*self.mu
        
        xv = np.linspace(m-5*s, m+5*s, 101, endpoint=True)
        pdf = np.exp(-0.5*(xv-m)**2/s**2) / np.sqrt(2.*np.pi*s**2)
        return (xv, pdf)
    
class TimeSeries_GeometricBrownian(TimeSeries):
    def __init__(self, mu:float, sigma: float):
        super().__init__()

        self.noise = np.random.normal
        self.mu = mu
        self.sigma = sigma
        return

    def __init__(self):
        return "Geometric Brownian Motion"
    
    def sample(self, num_samples: int, num_steps: int, dt: float, *,
               method='exact'):
        
        series = np.sqrt(dt)*self.sigma*self.noise( size=(num_samples, num_steps+1) )
        series[:,0] = 0.
        series = ( dt*(self.mu-0.5*self.sigma**2)*np.arange(num_steps+1)
                   + np.cumsum(series, axis=-1)
                  )
        return np.exp(series)

    # Debug this
    def mean(self, time: float):
        print("Warning, not implemented yet")
        return np.exp(self.mu*time)

    # Debug this expression
    def std(self, time: float):
        print("Warning, not implemented yet")
        return self.mean(time)

    def distribution(self, method='exact'):
        # Add some assertions on method

        if method=='exact':
            pass
        elif method=='sample':
            pass
        
        return

class TimeSeries_FractionalBrownian(TimeSeries):

    def __init__(self):
        return
    
class TimeSeries_BrownianBridge(TimeSeries):
    def __init__(self, mu: float, sigma: float):
        super().__init__()

        self.noise = np.random.normal
        self.mu = mu
        self.sigma = sigma
        return

    def __str__(self):
        return "Brownian Bridge"

    # Add options for starting and ending times
    # Debut and fix this
    def sample(self, num_samples: int, num_steps: int, dt: float):
        series = np.sqrt(dt)*self.sigma*self.noise( size=(num_samples, num_steps+1) )
        series[:,0] = 0.
        series = np.cumsum(series,axis=-1) + dt*self.mu*np.arange(num_steps+1)
        series = series - np.expand_dims(series[:,-1],-1)*np.arange(num_steps+1)/num_steps
        return series
    
class TimeSeries_RW(TimeSeries):

    def __init__(self):
        super().__init__()
        return

    def sample(self, num_samples: int, num_steps: int):
        return

class TimeSeries_OE(TimeSeries):

    def __init__(self, mu, theta, sigma):
        super().__init__()

        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        
        return

    def __str__(self):
        return "Orenstein-Uhlenbeck Process"

    # Allow option to set initial value
    def sample(self, num_samples: int, num_steps: int, dt: float):
        noise = np.sqrt(dt)*self.sigma*np.random.normal( size=(num_samples, num_steps) )
        series = np.zeros( (num_samples, num_steps+1) )
        series[:,0] = self.mu
        
        # Stupid Euler method
        for i in range(num_steps):
            series[:,i+1] = (series[:,i] 
                           + self.theta*(self.mu - series[:,i])*dt
                           + noise[:,i]
                             )
        return series
    
class TimeSeries_CIR(TimeSeries):

    def __init__(self, mu, theta, sigma):
        super().__init__()

        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        return

    def __str__(self):
        return "Cox-Ingersoll-Ross Model"
    
    def sample(self, num_samples: int, num_steps: int, dt: float, *,
               mean: float = None):
        noise = np.sqrt(dt)*self.sigma*np.random.normal(size=(num_samples, num_steps))
        series = np.zeros( (num_samples, num_steps+1) )
        series[:,0] = self.mu  # Adjust this

        # Different options for keeping sqrt defined in here
        for i in range(num_steps):
            series[:,i+1] = (series[:,i] 
                           + dt*self.theta*(self.mu - series[:,i])
                           + np.sqrt(np.abs(series[:,i]))*noise[:,i]
                             )
                             
        return series
    
if __name__=="__main__":
    #ARMA = TimeSeries_ARMA([0.3,0.4, 0.5], [0.1,0.3], 1.7)
    #s = ARMA.sample(10000,256)

    #CIR = TimeSeries_CIR(0.05, 2., 0.2)
    #s = CIR.sample(10000, 500, 0.01)

    GBM = TimeSeries_GeometricBrownian(0.05, 0.2)
    s = GBM.sample(10000, 500, 0.1)
