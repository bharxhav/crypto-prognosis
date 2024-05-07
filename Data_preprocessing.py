"""
Predicts today's trade strategy.
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Preprocessing:
    """
    Data preprocessing for feeding the models
    """

    def __init__(self, timestep):
        """
        Initialize TokenPredictor with attributes.
        """
        self.scaler = StandardScaler()
        self.timesteps = timestep
        self.yscaler = None
        self.x_last = None

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
        print(x.head())

        y = x['Open'].shift(-1).to_list()

        todays_high_estimate = json.load(open(todays_dir.format('GBTC')))['high']
        y.pop(0)
        y.append(todays_high_estimate)

        # handle missing values
        y = np.array(y).reshape((-1, 1))
        if np.isnan(y).any():
            nan_indexes = np.where(np.isnan(y))[0]
            # Replace NaN values with mean of neighboring elements
            for i in nan_indexes:
                # Calculate the mean of neighboring elements
                neighbor_mean = np.nanmean(y[max(0, i - 1):min(i + 2, len(y))])
                y[i] = neighbor_mean


        ## Scaling
        x.drop(columns=['Date'], inplace=True)
        x = self.scaler.fit_transform(x)

        self.yscaler = StandardScaler()
        y = self.yscaler.fit_transform(y)

        return x, y, self.scaler, self.yscaler

    def _chunking(self, x, y, test_size=0.05):
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

        # For prediction
        self.x_last = x_stack[-1]

        x_train, x_test, y_train, y_test = train_test_split(x_stack, y_stack, test_size=test_size)

        return x_train, x_test, y_train, y_test, self.x_last

    def _enter_trade(self, interval_pred, open_price, delta=0.01):
        """
        Determine a trading strategy depending on confidence in high prediction. 
        If the (1 + delta)*Open is in the predicted interval, we are less than 1 - alpha/2 % (default is 60%) 
            sure that the high of the day will be superior to the open.

        Args:
        - interval_pred: Predicted interval for high price at 1 - alpha confidence level.
        - open_price: Today's open price.
        - delta: Percentage increase threshold above the open price (default: 1%).

        Returns:
        - trade (bool): True if the strategy is to trade False otherwise
        """
        threshold = (1 + delta) * open_price
        trade = True
        if threshold > interval_pred:
            trade = False
        return trade

    def plot_with_interval(self, y_test, y_pred, y_pis):
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel("GBTC Value")
        x = np.linspace(0, y_pis[:,0, 0].shape[0], num=y_pis[:,0, 0].shape[0])
        ax.plot(x, self.yscaler.inverse_transform(y_test), lw=2, label="Test data", c="C1")
        ax.plot(x, y_pred, lw=2, c="C2", label="Predictions")
        ax.fill_between(
            x,
            self.yscaler.inverse_transform(y_pis[:,0, 0].reshape(-1,1)).flatten(),
            max(max(y_pred), max(y_test))+5, #max y as we want to predict a one sided interval
            color="C2",
            alpha=0.2,
            label="CV + PIs",
        )
        ax.legend()
        plt.show()

    def performance_conformal(self, y_test, y_pis):
        n =  len(y_test)
        error = 0
        predicted_lower_bound = y_pis[:,0,0]
        for i in range(n):
            if predicted_lower_bound[i]> y_test[i]:
                error += 1
        return error/n