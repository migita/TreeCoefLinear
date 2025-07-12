import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
import xgboost as xgb
import lightgbm as lgb


class TreeCoefficientLinear(BaseEstimator, RegressorMixin):
    """
    A boosting method that learns functions of the form:
    y = T₁(X)·z₁ + T₂(X)·z₂ + ... + Tₖ(X)·zₖ
    
    Where:
    - X are the feature variables
    - z₁, z₂, ..., zₖ are coefficient variables
    - T₁, T₂, ..., Tₖ are tree-based functions learned by the model
    
    This extends standard boosting which typically learns y = T(X) or y = T(X)·z
    
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
        random_state: Optional[int] = None,
        **kwargs
    ):
        self.n_components = n_components
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.models_ = []
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
        
        # Initialize residuals
        residuals = y.copy()
        self.models_ = []
        
        # Alternating optimization: fit each T_i while keeping others fixed
        for iteration in range(3):  # Multiple passes through all components
            for i in range(self.n_components):
                # Current coefficient variable
                z_i = Z[:, i]
                
                # Create target for this component
                # We want to fit T_i(X) such that T_i(X) * z_i approximates the residuals
                # This is equivalent to fitting T_i(X) to residuals / z_i (with care for z_i = 0)
                
                # Avoid division by zero
                mask = np.abs(z_i) > 1e-10
                if np.sum(mask) < 10:  # Too few non-zero values
                    continue
                
                # Target for T_i(X) is residuals / z_i for samples where z_i != 0
                target_i = np.zeros_like(residuals)
                target_i[mask] = residuals[mask] / z_i[mask]
                
                # Create or update model for component i
                if iteration == 0:
                    seed = self.random_state + i if self.random_state else None
                    model_i = self._create_base_model(seed)
                    self.models_.append(model_i)
                else:
                    model_i = self.models_[i]
                
                # Fit the model
                if self.base_estimator == 'xgboost' and iteration > 0:
                    # For XGBoost, we can continue training from previous iteration
                    model_i.fit(
                        X[mask], 
                        target_i[mask],
                        xgb_model=model_i.get_booster()
                    )
                else:
                    model_i.fit(X[mask], target_i[mask])
                
                # Update residuals
                predictions_i = model_i.predict(X)
                residuals = y - self._predict_components(X, Z, exclude_component=i)
        
        self.is_fitted_ = True
        return self
    
    def _predict_components(self, X: np.ndarray, Z: np.ndarray, exclude_component: Optional[int] = None):
        """Predict using all components except possibly one."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(len(self.models_)):
            if i == exclude_component:
                continue
            T_i = self.models_[i].predict(X)
            z_i = Z[:, i]
            predictions += T_i * z_i
        
        return predictions
    
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
        
        return self._predict_components(X, Z)
    
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