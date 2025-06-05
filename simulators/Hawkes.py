# File: stdgp/generators/multi_mark_hawkes_dgp.py
import numpy as np
import pandas as pd
from .base_dgp import AbstractSTDataGenerator
from utils.thinning import thinning

# Import the registry functions from the intensities folder
from intensities.registry import get_background, get_kernel
#from intensities.intensity import CompositeIntensity

class MultiMarkHawkesDGP(AbstractSTDataGenerator):
    """
    Multi-mark Hawkes simulator that builds its background and kernel
    models based on configuration dictionaries (e.g., loaded from a YAML file).
    
    The overall intensity is computed as the sum of a background intensity and 
    a kernel (triggered) intensity. The background and kernel are created 
    using centralized registry functions.
    """
    def __init__(self,
                 T: float,
                 domain: list,
                 A: np.array,
                 Lambda: float,
                 background_config: dict,
                 kernel_config: dict,
                 mean: np.ndarray,
                 cov: np.ndarray,
                 rng=None):
        """
        Parameters:
            T : float
                Time horizon.
            domain : list
                Spatial domain as [x_min, x_max, y_min, y_max].
            m : int
                Number of marks/types.
            Lambda : float
                Global upper bound on intensity for thinning.
            background_config : dict
                Configuration dictionary for background intensity.
            kernel_config : dict
                Configuration dictionary for the triggering kernel.
            mean : np.ndarray
                Mean vector for normalization.
            cov : np.ndarray
                Covariance matrix for normalization.
            rng : np.random.Generator or None
                Random number generator (defaults to np.random.default_rng()).
        """
        super().__init__(mean, cov)
        self.T = T
        self.domain = domain
        self.A = A
        self.m = self.A.shape[0]  # Number of marks/types
        self.Lambda = Lambda
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Use the registry functions to build background and kernel from config.
        self.background = get_background(background_config)
        self.kernel = get_kernel(kernel_config)
        
        # Create the composite intensity that sums background and kernel contributions.
        #self.composite_intensity = CompositeIntensity(background, kernel)
    
    def generate_data(self, n_samples: int = None) -> pd.DataFrame:
        """
        Generate spatiotemporal events using the thinning algorithm.
        
        Returns:
            pd.DataFrame: A DataFrame with columns ['t', 'x', 'y', 'type', 'lambda'],
                          sorted by time.
        """
        events = thinning(self.background, self.kernel, self.T, self.domain, self.A, self.Lambda, rng=self.rng)
        events.sort(key=lambda e: e['t'])
        #df_events = pd.DataFrame(events, columns=['t', 'x', 'y', 'type', 'lambda'])
        return events
