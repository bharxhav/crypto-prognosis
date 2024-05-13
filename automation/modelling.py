import sys
import os
import re

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules')))

from modules.token_predictor import TokenPredictor

tp = TokenPredictor()
tp.train(directory="../data/")
prediction = tp.predict()

with open('../README.md', 'r') as f:
    lines = f.readlines()

with open('../README.md', 'w') as f:
    for line in lines:
        if 'Expected close' in line:
            line = re.sub('(\d+(\.\d+)?)', f'{prediction}', line)
        f.write(line)
