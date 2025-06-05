from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class AbstractSTDataGenerator(ABC):
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        Initialize the base generator with mean and covariance matrix.
        These parameters can be used for normalization and unnormalization.
        
        Parameters:
            mean (np.ndarray): Mean vector for the spatial dimensions.
            cov (np.ndarray): Covariance matrix for the spatial dimensions.
        """
        self.mean = mean
        self.cov = cov

    def normalize(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Normalize the given columns of the DataFrame based on the mean and covariance.
        A simple normalization subtracting mean and dividing by standard deviation is shown.
        
        Parameters:
            data (pd.DataFrame): Input data containing the columns to normalize.
            columns (list): List of column names to normalize.
        
        Returns:
            pd.DataFrame: DataFrame with normalized values.
        """
        norm_data = data.copy()
        std = np.sqrt(np.diag(self.cov))
        for i, col in enumerate(columns):
            norm_data[col] = (norm_data[col] - self.mean[i]) / std[i]
        return norm_data

    def unnormalize(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Unnormalize the given columns of the DataFrame.
        
        Parameters:
            data (pd.DataFrame): DataFrame with normalized values.
            columns (list): List of column names to unnormalize.
        
        Returns:
            pd.DataFrame: DataFrame with values scaled back to original.
        """
        unnorm_data = data.copy()
        std = np.sqrt(np.diag(self.cov))
        for i, col in enumerate(columns):
            unnorm_data[col] = unnorm_data[col] * std[i] + self.mean[i]
        return unnorm_data

    @abstractmethod
    def generate_data(self, n_samples: int) -> pd.DataFrame:
        """
        Abstract method to generate spatiotemporal data.
        
        Parameters:
            n_samples (int): Number of data points to generate.
        
        Returns:
            pd.DataFrame: DataFrame with at least 'time', 'x', and 'y' columns.
        """
        pass