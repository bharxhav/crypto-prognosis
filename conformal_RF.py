import warnings

import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from mapie.metrics import (coverage_width_based, regression_coverage_score,
                           regression_mean_width_score)
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap

warnings.simplefilter("ignore")

class RFRegressor:
    """
    TokenPredictor class for predicting token prices using Random Forest regressor.
    """

    def __init__(self, xscaler, yscaler, x_last):
        """
        Initialize TokenPredictor with attributes.
        """
        self.model = None
        self.mapie = None
        self.scaler = xscaler
        self.yscaler = yscaler
        self.x_last = x_last

    def _params_fit(self,x_train, y_train):
        """
        Random search to find best parameters for the regressor

        Args:
        - x_train (array): Training input features.
        - y_train (array): Training target values.
        
        Returns:
        None
        """
        x_train = x_train.reshape(-1, 3 * 12)
        # CV parameter search
        n_iter = 100
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        random_state = 59
        rf_model = RandomForestRegressor(random_state=random_state)
        rf_params = {"max_depth": randint(2, 15), "n_estimators": randint(10, 200)}
        cv_obj = RandomizedSearchCV(
            rf_model,
            param_distributions=rf_params,
            n_iter=n_iter,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            random_state=random_state,
            verbose=0,
            n_jobs=-1,
        )
        cv_obj.fit(x_train, y_train)
        self.model = cv_obj.best_estimator_
        print(cv_obj.best_estimator_.get_params)

    def _train(self, x_train, y_train):
        """
        Train the RF Regressor.
    
        Args:
        - x_train (array): Training input features.
        - y_train (array): Training target values.
        
        Returns:
        None
        """
        x_train = x_train.reshape(-1, 3 * 12)
        cv_mapiets = BlockBootstrap(
            n_resamplings=100, n_blocks=1000, overlapping=False, random_state=59
        )
        self.mapie = MapieTimeSeriesRegressor(
            self.model, method="enbpi", cv=cv_mapiets, agg_function="median", n_jobs=-1
        )
        self.mapie.fit(x_train, y_train)

    def _predict(self, x_test, alpha=0.8):
        """
        Predict the next day's high.

        Returns:
        - y_pred (array): Predicted next day's high.
        - y_pis (3D array): Conformal prediction interval at 1 - alpha confidence level
        """
        # Predicting the next day's high
        y_pred, y_pis = self.mapie.predict(self.x_last.reshape(-1, 3 * 12), alpha=alpha)
        y_pred_test, y_pis_test = self.mapie.predict(x_test.reshape(-1, 3 * 12), alpha=alpha)

        return self.yscaler.inverse_transform(y_pred.reshape(-1, 1))[0][0], self.yscaler.inverse_transform(y_pis.reshape(-1,1))[0][0], self.yscaler.inverse_transform(y_pred_test.reshape(-1,1)), y_pis_test

    def _partial_fit(self, x_test, y_test, gap=1, alpha=0.8):
        x_test = x_test.reshape(-1, 3 * 12)
        y_pred, y_pis = self.mapie.predict(x_test, alpha=alpha)
        y_pred_enbpi_pfit = np.zeros(y_pred.shape)
        y_pis_enbpi_pfit = np.zeros(y_pis.shape)
        y_pred_enbpi_pfit[:gap], y_pis_enbpi_pfit[:gap, :, :] = self.mapie.predict(
            x_test[:gap, :], alpha=alpha, ensemble=True, optimize_beta=True,
            allow_infinite_bounds=True
        )

        for step in range(gap, len(x_test), gap):
            self.mapie.partial_fit(
                x_test[(step - gap):step, :],
                y_test[(step - gap):step],
            )
            (
                y_pred_enbpi_pfit[step:step + gap],
                y_pis_enbpi_pfit[step:step + gap, :, :],
            ) = self.mapie.predict(
                x_test[step:(step + gap), :],
                alpha=alpha,
                ensemble=True,
                optimize_beta=True,
                allow_infinite_bounds=True
            )
            y_pis_enbpi_pfit[step:step + gap, :, :] = np.clip(
                y_pis_enbpi_pfit[step:step + gap, :, :], 0, 1
            )
        return self.yscaler.inverse_transform(y_pred_enbpi_pfit.reshape(-1,1)), y_pis_enbpi_pfit