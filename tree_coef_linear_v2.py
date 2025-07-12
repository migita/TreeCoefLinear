import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
import xgboost as xgb
import lightgbm as lgb


class TreeCoefficientLinearV2(BaseEstimator, RegressorMixin):
    """
    An improved boosting method that learns functions of the form:
    y = T₁(X)·z₁ + T₂(X)·z₂ + ... + Tₖ(X)·zₖ
    
    This version uses a better optimization strategy:
    1. Initialize all T_i to constants
    2. Use coordinate descent with proper residual updates
    3. Include regularization to prevent overfitting
    
    Parameters
    ----------
    n_components : int, default=2
        Number of tree functions to learn (k in the formula above)
    
    base_estimator : str, default='xgboost'
        Which gradient boosting library to use. Options: 'xgboost', 'lightgbm'
    
    n_estimators : int, default=100
        Number of boosting rounds for each component
    
    learning_rate : float, default=0.1
        Learning rate for the boosting algorithm
    
    max_depth : int, default=3
        Maximum depth of trees
    
    n_iterations : int, default=5
        Number of coordinate descent iterations
    
    regularization : float, default=0.01
        L2 regularization parameter
    
    random_state : int or None, default=None
        Random seed for reproducibility
    
    **kwargs : dict
        Additional parameters passed to the base estimator
    """
    
    def __init__(
        self,
        n_components: int = 2,
        base_estimator: str = 'xgboost',
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        n_iterations: int = 5,
        regularization: float = 0.01,
        random_state: Optional[int] = None,
        **kwargs
    ):
        self.n_components = n_components
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.models_ = []
        self.intercepts_ = []
        self.is_fitted_ = False
        
    def _create_base_model(self, seed: Optional[int] = None):
        """Create a base gradient boosting model."""
        if self.base_estimator == 'xgboost':
            params = {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'random_state': seed,
                'objective': 'reg:squarederror',
                'reg_lambda': self.regularization,
                **self.kwargs
            }
            return xgb.XGBRegressor(**params)
        elif self.base_estimator == 'lightgbm':
            params = {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'random_state': seed,
                'objective': 'regression',
                'verbose': -1,
                'lambda_l2': self.regularization,
                **self.kwargs
            }
            return lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"Unknown base_estimator: {self.base_estimator}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, Z: np.ndarray):
        """
        Fit the model to learn y = T₁(X)·z₁ + T₂(X)·z₂ + ... + Tₖ(X)·zₖ
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature variables
        
        y : array-like of shape (n_samples,)
            Target variable
        
        Z : array-like of shape (n_samples, n_components)
            Coefficient variables. Each column represents one z variable.
        
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate inputs
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        Z = check_array(Z)
        
        if Z.shape[1] != self.n_components:
            raise ValueError(f"Z must have {self.n_components} columns, got {Z.shape[1]}")
        
        if X.shape[0] != y.shape[0] or X.shape[0] != Z.shape[0]:
            raise ValueError("X, y, and Z must have the same number of samples")
        
        n_samples = X.shape[0]
        
        # Initialize T_i(X) as constants
        self.models_ = []
        self.intercepts_ = []
        T_values = np.zeros((n_samples, self.n_components))
        
        # Initialize each T_i as the mean of y/z_i for non-zero z_i
        for i in range(self.n_components):
            z_i = Z[:, i]
            mask = np.abs(z_i) > 1e-10
            if np.sum(mask) > 0:
                initial_value = np.mean(y[mask] / z_i[mask])
            else:
                initial_value = 0.0
            T_values[:, i] = initial_value
            self.intercepts_.append(initial_value)
        
        # Coordinate descent optimization
        for iteration in range(self.n_iterations):
            for i in range(self.n_components):
                # Compute residuals excluding component i
                residuals = y.copy()
                for j in range(self.n_components):
                    if j != i:
                        residuals -= T_values[:, j] * Z[:, j]
                
                # Current coefficient variable
                z_i = Z[:, i]
                
                # Create weighted target for T_i(X)
                # We want to minimize: ||y - sum_j T_j(X)*z_j||^2
                # For component i: ||residuals - T_i(X)*z_i||^2
                # This is a weighted regression problem
                
                # Use sample weights proportional to z_i^2
                weights = z_i ** 2
                if np.mean(weights) > 0:
                    weights = weights / np.mean(weights)  # Normalize
                else:
                    weights = np.ones_like(weights)
                
                # Target is residuals / z_i where z_i != 0
                mask = np.abs(z_i) > 1e-10
                
                # Create or get model
                if iteration == 0:
                    seed = self.random_state + i if self.random_state else None
                    model_i = self._create_base_model(seed)
                    self.models_.append(model_i)
                else:
                    model_i = self.models_[i]
                
                if np.sum(mask) < 10:
                    # If too few non-zero values, fit a constant model
                    # We'll use a dummy fit to initialize the model
                    dummy_X = X[:10] if len(X) >= 10 else X
                    dummy_y = np.zeros(len(dummy_X))
                    model_i.fit(dummy_X, dummy_y)
                    T_values[:, i] = 0.0
                    continue
                
                target_i = residuals[mask] / z_i[mask]
                
                # Fit the model with sample weights
                model_i.fit(X[mask], target_i, sample_weight=weights[mask])
                
                # Update T_values
                T_values[:, i] = model_i.predict(X)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray, Z: np.ndarray):
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature variables
        
        Z : array-like of shape (n_samples, n_components)
            Coefficient variables
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted values
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X)
        Z = check_array(Z)
        
        if Z.shape[1] != self.n_components:
            raise ValueError(f"Z must have {self.n_components} columns, got {Z.shape[1]}")
        
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(self.n_components):
            T_i = self.models_[i].predict(X)
            z_i = Z[:, i]
            predictions += T_i * z_i
        
        return predictions
    
    def get_tree_functions(self, X: np.ndarray):
        """
        Get the values of each tree function T_i(X) for given inputs.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature variables
        
        Returns
        -------
        T_values : array of shape (n_samples, n_components)
            Values of each tree function
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting tree functions")
        
        X = check_array(X)
        n_samples = X.shape[0]
        T_values = np.zeros((n_samples, self.n_components))
        
        for i in range(self.n_components):
            T_values[:, i] = self.models_[i].predict(X)
        
        return T_values
    
    def feature_importances_(self):
        """
        Get feature importances for each component.
        
        Returns
        -------
        importances : dict
            Dictionary mapping component index to feature importances
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importances")
        
        importances = {}
        for i in range(self.n_components):
            if hasattr(self.models_[i], 'feature_importances_'):
                importances[f'component_{i}'] = self.models_[i].feature_importances_
        
        return importances