"""
vcboost.py
==========

Varying‑Coefficient Gradient Boosting with **vector‑leaf** trees in XGBoost ≥ 2.0.

Model
-----

    y_i = Σ_{j=1..K}  v_{ij} · F_j(x_i)     (least‑squares loss)

where

* x_i : ordinary feature vector  (shape  (n_samples, n_features))
* v_i : known multipliers        (shape  (n_samples, K))
* F_j : unknown functions to be learned, one per column of *V*

All K functions are fitted **jointly** in one vector‑leaf tree per boosting
iteration – the greedy, stage‑wise procedure used by standard GBMs, but
generalised to multi‑output leaves.
"""
from __future__ import annotations

from typing import Optional, Tuple, Sequence, Any

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (
    check_array,
    check_X_y,
    check_is_fitted,
)


class VCBoostRegressor(BaseEstimator, RegressorMixin):
    """
    Varying‑Coefficient GBM with vector‑leaf trees.

    Parameters
    ----------
    n_estimators : int, default=500
        Number of boosting iterations.
    learning_rate : float, default=0.05
        Shrinkage applied after each tree.
    max_depth : int, default=6
        Maximum depth of a tree.
    subsample : float, default=0.8
        Row subsampling ratio per tree.
    colsample_bytree : float, default=0.8
        Column subsampling ratio per tree.
    reg_lambda : float, default=0.0
        L2 regularisation on leaf values.
    n_jobs : int, default=-1
        Threads to use.
    random_state : int or None, default=None
        Seed for reproducibility.
    """

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 0.0,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    #                           FIT
    # ------------------------------------------------------------------ #
    def fit(self, X, y, V) -> "VCBoostRegressor":
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        V : array-like of shape (n_samples, K)   -- multipliers
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        V = np.asarray(V, dtype=np.float32)
        if V.ndim == 1:
            V = V[:, None]
        n_samples, K = V.shape
        if n_samples != X.shape[0]:
            raise ValueError("X and V must have the same number of rows.")

        self.K_ = K

        # ------------------------------------------------------------------ #
        # Prepare DMatrix with dummy labels (custom objective overrides them)
        dtrain = xgb.DMatrix(X, label=np.zeros((n_samples, K), dtype=np.float32))

        # Keep copies for predict()
        self._V_train_ = V.copy()
        self._X_train_ = X.copy()

        # ------------------------------------------------------------------ #
        # Build custom objective (gradient & Hessian)
        def obj(
            preds: np.ndarray,  # raw scores (n_samples * K,)
            dmat: xgb.DMatrix,
        ) -> Tuple[np.ndarray, np.ndarray]:
            preds = preds.reshape(n_samples, K)
            resid = (y - (preds * V).sum(axis=1)).astype(np.float32)
            grad = (-resid[:, None] * V).ravel()
            hess = (V ** 2).ravel() + 1e-12  # add ε for numerical safety
            return grad, hess

        # ------------------------------------------------------------------ #
        params = {
            "tree_method": "hist",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "lambda": self.reg_lambda,
            "objective": "reg:squarederror",  # placeholder
            "num_target": K,
            "multi_strategy": "multi_output_tree",
            "n_jobs": self.n_jobs,
            "seed": self.random_state,
        }

        self._booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            obj=obj,
        )
        return self

    # ------------------------------------------------------------------ #
    #                          PREDICT
    # ------------------------------------------------------------------ #
    def predict(self, X, V) -> np.ndarray:
        """
        Predict using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        V : array-like of shape (n_samples, K)   -- multipliers

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "_booster")
        X = check_array(X, accept_sparse=False)
        V = np.asarray(V, dtype=np.float32)
        if V.ndim == 1:
            V = V[:, None]
        if V.shape[1] != self.K_:
            raise ValueError(
                f"V must have {self.K_} columns (got {V.shape[1]})."
            )

        dpred = xgb.DMatrix(X)
        preds = self._booster.predict(dpred)  # shape (n_samples, K)
        return (preds * V).sum(axis=1)

    # ------------------------------------------------------------------ #
    #                       GET LEAF CONTRIBUTIONS
    # ------------------------------------------------------------------ #
    def predict_components(self, X) -> np.ndarray:
        """
        Return the K learned functions F_j(X).

        Useful for inspecting how each multiplier column contributes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        F : ndarray of shape (n_samples, K)
        """
        check_is_fitted(self, "_booster")
        X = check_array(X, accept_sparse=False)
        dpred = xgb.DMatrix(X)
        return self._booster.predict(dpred)  # raw vector leaf outputs