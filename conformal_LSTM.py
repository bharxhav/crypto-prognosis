"""
Predicts today's trade strategy.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from scikeras.wrappers import KerasRegressor


class LSTM_Predictor:
    """
    TokenPredictor class for predicting token prices using LSTM neural network.
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

    def _create(self, input_shape):
        """
        Create an LSTM neural network model and wraps it as sklearn-compatible regressors using the KerasRegressor wrapper

        Args:
        - input_shape (tuple): Input shape for the model.

        Returns:
        None
        """
        features = input_shape[-1]
        
        model = Sequential()

        model.add(LSTM(units=features, return_sequences=True, input_shape=input_shape[1:]))
        model.add(LSTM(units=features, return_sequences=True))
        model.add(LSTM(units=features//2, return_sequences=True))
        model.add(LSTM(units=features//4, return_sequences=False))
        model.add(Dense(units=1))

        model.compile(loss='mse', optimizer='adam')
        print(model.summary())
        
        self.model = KerasRegressor(
            model=model, 
            epochs=100, 
            verbose=1
        )

    def _train(self, x_train, y_train,batch_size=128):
        """
        Train the LSTM model.
    
        Args:
        - x_train (array): Training input features.
        - y_train (array): Training target values.
        
        Returns:
        None
        """
        cv_mapiets = BlockBootstrap(
            n_resamplings=10, n_blocks=10, overlapping=False, random_state=59
        )
        self.mapie = MapieTimeSeriesRegressor(
            self.model, method="enbpi", cv=cv_mapiets, agg_function="median", n_jobs=-1
        )
        self.mapie.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size)

    def _predict(self, x_test, alpha=0.8):
        """
        Predict the next day's high.

        Returns:
        - y_pred (array): Predicted next day's high.
        - y_pis (3D array): Conformal prediction interval at 1 - alpha confidence level
        """
        # Predicting the next day's high
        y_pred, y_pis = self.mapie.predict(self.x_last.reshape(1,-1,12), alpha=alpha)
        y_pred_test, y_pis_test = self.mapie.predict(x_test, alpha=alpha)

        return self.yscaler.inverse_transform(y_pred.reshape(-1, 1))[0][0], self.yscaler.inverse_transform(y_pis.reshape(-1,1))[0][0], self.yscaler.inverse_transform(y_pred_test.reshape(-1,1)), y_pis_test
