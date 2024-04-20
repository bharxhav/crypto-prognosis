import sys
import os
import json

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from token_predictor import TokenPredictor

tp = TokenPredictor()
tp.train()
prediction = tp.predict()


