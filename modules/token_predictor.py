import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


class TokenPredictor:
    """
    TokenPredictor class for predicting token prices using LSTM neural network.
    """

    def __init__(self, timesteps=150):
        """
        Initialize TokenPredictor with attributes.

        Args:
            timesteps (int): Number of time steps for LSTM input (default: 150)
        """
        self.model = None
        self.scaler = StandardScaler()
        self.timesteps = timesteps
        self.yscaler = StandardScaler()
        self.x_last = None
        self.feature_count = None

    def _load_and_preprocess_data(self, main, assets, directory="./data/"):
        """
        Load and preprocess data from CSV files and JSON files.

        Args:
            main (str): The main asset for which data is loaded
            assets (list): List of additional assets
            directory (str): Directory where data files are located

        Returns:
            tuple: (x, y) preprocessed input features and target values

        Raises:
            FileNotFoundError: If required files are not found
            ValueError: If data format is incorrect
        """
        try:
            assets_dir = os.path.join(directory, "assets", "{}.csv")
            todays_dir = os.path.join(directory, "todays", "{}.json")
            cols = ['Date', 'Open', 'High', 'Low',
                    'Close', 'Adj Close', 'Volume']

            # Load main asset data
            main_file = assets_dir.format(main)
            if not os.path.exists(main_file):
                raise FileNotFoundError(
                    f"Main asset file not found: {main_file}")

            x = pd.read_csv(main_file)
            if not all(col in x.columns for col in cols):
                raise ValueError(f"Missing required columns in {main_file}")

            # Load and merge additional assets
            for token in assets:
                token_file = assets_dir.format(token)
                if not os.path.exists(token_file):
                    print(f"Warning: Asset file not found: {token_file}")
                    continue

                df2 = pd.read_csv(token_file)
                if not all(col in df2.columns for col in cols):
                    print(
                        f"Warning: Missing columns in {token_file}, skipping")
                    continue

                # Rename columns for merging
                rename_cols = {col: f"{col}_{token}" for col in cols[1:]}
                df2 = df2.rename(columns=rename_cols)

                x = pd.merge(x, df2, on="Date", how="left")

            # Load today's estimates
            todays_file = todays_dir.format(main)
            if not os.path.exists(todays_file):
                raise FileNotFoundError(
                    f"Today's estimates file not found: {todays_file}")

            with open(todays_file, 'r') as f:
                todays_data = json.load(f)
                todays_high_estimate = todays_data['high']

            # Prepare target variable
            y = x['Open'].shift(-1).tolist()
            y.pop(0)
            y.append(todays_high_estimate)
            y = np.array(y).reshape((-1, 1))

            # Scale the data
            x = x.drop(columns=['Date'])
            x = self.scaler.fit_transform(x)
            y = self.yscaler.fit_transform(y)

            self.feature_count = x.shape[1]

            return x, y

        except Exception as e:
            raise Exception(f"Error in data preprocessing: {str(e)}")

    def _lstm_chunking(self, x, y, test_size=0.2, random_state=42):
        """
        Chunk input features and target values for LSTM modeling.

        Args:
            x (array): Input features
            y (array): Target values
            test_size (float): Test set size ratio
            random_state (int): Random seed for reproducibility

        Returns:
            tuple: Training and test sets (x_train, x_test, y_train, y_test)
        """
        x_stack, y_stack = [], []

        for i in range(self.timesteps, len(x)):
            x_stack.append(x[i - self.timesteps:i, :])
            y_stack.append(y[i])

        x_stack, y_stack = np.array(x_stack), np.array(y_stack)

        # Store last window for prediction
        self.x_last = x_stack[-1]

        return train_test_split(x_stack, y_stack, test_size=test_size,
                                random_state=random_state, shuffle=False)

    def _create_lstm_model(self, input_shape):
        """
        Create an LSTM neural network model.

        Args:
            input_shape (tuple): Input shape for the model
        """
        features = input_shape[-1]

        model = Sequential([
            LSTM(units=features, return_sequences=True,
                 input_shape=input_shape[1:]),
            Dropout(0.2),
            LSTM(units=features, return_sequences=True),
            Dropout(0.2),
            LSTM(units=features//2, return_sequences=True),
            Dropout(0.2),
            LSTM(units=features//4, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train(self, directory='./data/', epochs=100, batch_size=128):
        """
        Train the LSTM model using data from CSV files.

        Args:
            directory (str): Data directory path
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        try:
            # Load and preprocess data
            x, y = self._load_and_preprocess_data('GBTC', ['ETCG', 'ETHE', 'GDLC'],
                                                  directory)

            # Create training and test sets
            x_train, x_test, y_train, y_test = self._lstm_chunking(x, y)

            # Create and train model
            self._create_lstm_model(x_train.shape)

            self.model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=2
            )

        except Exception as e:
            raise Exception(f"Error during training: {str(e)}")

    def predict(self):
        """
        Predict the next day's high.

        Returns:
            float: Predicted next day's high price
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        try:
            prediction = self.model.predict(
                self.x_last.reshape(1, self.timesteps, self.feature_count)
            )
            return float(self.yscaler.inverse_transform(prediction)[0][0])
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

    def calculate_probability(self, high_pred, open_price, delta=0.01):
        """
        Calculate the probability that High >= (1 + delta) * Open.

        Args:
            high_pred (float): Predicted high price
            open_price (float): Today's open price
            delta (float): Percentage increase threshold

        Returns:
            float: Probability percentage
        """
        threshold = (1 + delta) * open_price
        return float((high_pred >= threshold) * 100)
