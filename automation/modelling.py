import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from token_predictor import TokenPredictor

tp = TokenPredictor()
tp.train()
prediction = tp.predict()

with open('../README.md', 'r') as f:
    lines = f.readlines()

with open('../README.md', 'w') as f:
    for line in lines:
        if 'Expected close for GBTC: ``' in line:
            line = line.replace('Expected close for GBTC: ``', f'Expected close for GBTC: `{prediction}`')
        f.write(line)
