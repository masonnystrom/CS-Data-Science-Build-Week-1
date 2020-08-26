import sys
sys.path.append('./')
from knn_model import KNearestNeighbor
import pandas as pd
import numpy as np


# Load cannabis strain dataset scraped from Weedmaps
df = pd.read_csv("https://raw.githubusercontent.com/Build-Week-Med-Cabinet-2-MP/bw-med-cabinet-2-ml/master/data/CLEAN_WMS_2020_05_24.csv")

# assign and fit model
model = KNearestNeighbor()
model.fit(df)

# Create a fake user-input 
N=58
K=5
strain_array = np.array([1] * K + [0] * (N-K))
np.random.seed(2)
np.random.shuffle(strain_array)

# Return 10 nearest neighbors to fake user-input
predictions = model.prediction(strain_array, 10, "jaccard")
print(predictions)

