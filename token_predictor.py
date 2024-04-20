"""
Predicts today's trade strategy.
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class TokenPredictor:
    """
    TokenPredictor class for predicting token prices using LSTM neural network.
    """

    def __init__(self):
        """
        Initialize TokenPredictor with attributes.
        """
        self.model = None
        self.scaler = StandardScaler()
        self.timesteps = 150

    def _load_and_preprocess_data(self, main, assets, directory="./data/"):
        """
        Load and preprocess data from CSV files and JSON files.

        Args:
        - main (str): The main asset for which data is loaded.
        - assets (list): List of additional assets.
        - directory (str, optional): Directory where data files are located.

        Returns:
        - x (array): Preprocessed input features.
        - y (array): Preprocessed target values.
        """
        assets_dir = directory + "assets/{}.csv"
        todays_dir = directory + "todays/{}.json"
        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        x = pd.read_csv(assets_dir.format(main))

        for token in assets:
            df2 = pd.read_csv(assets_dir.format(token))

            columns = cols.copy()
            columns = [columns[0]] + [col + "_" + token for col in columns[1:]]

            x = x.merge(df2, on="Date", how="left", suffixes=('', '_' + token))

        y = x['Open'].shift(-1).to_list()

        todays_high_estimate = json.load(open(todays_dir.format('GBTC')))['high']
        y.pop(0)
        y.append(todays_high_estimate)

        y = np.array(y).reshape((-1, 1))

        ## Scaling
        x.drop(columns=['Date'], inplace=True)
        x = self.scaler.fit_transform(x)

        self.yscaler = StandardScaler()
        y = self.yscaler.fit_transform(y)

        return x, y

    def _lstm_chunking(self, x, y, test_size=0.2, random_state=42):
        """
        Chunk input features and target values for LSTM modeling.

        Args:
        - x (array): Input features.
        - y (array): Target values.
        - test_size (float, optional): Fraction of the dataset to include in the test split.
        - random_state (int, optional): Controls the randomness of the training and testing splits.

        Returns:
        - x_train (array): Training input features.
        - x_test (array): Testing input features.
        - y_train (array): Training target values.
        - y_test (array): Testing target values.
        """
        x_stack, y_stack = [], []

        for i in range(self.timesteps, len(x)):
            x_stack.append(x[i - self.timesteps:i,:])
            y_stack.append(y[i])
        
        x_stack, y_stack = np.array(x_stack), np.array(y_stack)

        print(x_stack.shape)
        print(y_stack.shape)
        print()

        # For prediction
        self.x_last = x_stack[-1]

        x_train, x_test, y_train, y_test = train_test_split(x_stack, y_stack, test_size=test_size, random_state=random_state)

        return x_train, x_test, y_train, y_test

    def _create_lstm_model(self, input_shape):
        """
        Create an LSTM neural network model.

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

        model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.model = model

    def _train_model(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=128):
        """
        Train the LSTM model.

        Args:
        - x_train (array): Training input features.
        - y_train (array): Training target values.
        - x_test (array): Testing input features.
        - y_test (array): Testing target values.
        - epochs (int, optional): Number of epochs to train the model.
        - batch_size (int, optional): Number of samples per gradient update.

        Returns:
        None
        """
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

    def train(self):
        """
        Train the LSTM model using data from CSV files.

        Args:
        - csv_loc (str): Location of the CSV file.

        Returns:
        None
        """
        x, y = self._load_and_preprocess_data('GBTC', ['ETCG', 'ETHE', 'GDLC'])
        x_train, x_test, y_train, y_test = self._lstm_chunking(x, y)

        self._create_lstm_model(x_train.shape)
        self._train_model(x_train, y_train, x_test, y_test)

    def _predict(self):
        """
        Predict the next day's high.

        Returns:
        - prediction (array): Predicted next day's high.
        """
        # Predicting the next day's high
        prediction = self.model.predict(self.x_last.reshape(1, -1, 24))

        return self.yscaler.inverse_transform(prediction)[0][0]

    def calculate_probability(self, high_pred, open_price, delta=0.01):
        """
        Calculate the probability that High >= (1 + delta) * Open.

        Args:
        - high_pred: Predicted high price.
        - open_price: Today's open price.
        - delta: Percentage increase threshold above the open price (default: 1%).

        Returns:
        - probability: Probability that High >= (1 + delta) * Open.
        """
        threshold = (1 + delta) * open_price
        probability = (high_pred >= threshold) * 100  # Probability in percentage
        return probability
