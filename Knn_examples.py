import sys
sys.path.append('./')
from knn_model import KNearestNeighbor
import pandas as pd
import numpy as np


nba_df = pd.read_csv("nba_2013.csv")

# The columns that we will be making predictions with.
X = ['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf']
# The column that we want to predict.
y_train = ["pts"]

# Test the model
model = KNearestNeighbor()
model.fit(nba_df)

# Return 10 nearest neighbors to fake user-input
predictions = model.prediction(y_train, k=3, metric = "jacaard")
print(predictions)

