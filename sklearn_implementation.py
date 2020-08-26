import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load cannabis csv from Medcabinet
df = pd.read_csv("https://raw.githubusercontent.com/Build-Week-Med-Cabinet-2-MP/bw-med-cabinet-2-ml/master/data/CLEAN_WMS_2020_05_24.csv")
df.drop(columns=["description"], inplace=True)
strain_df = df.drop(columns=["name"])

# Simulated user input for strain recommendation system
N=57
K=6
strain_array = np.array([1] * K + [0] * (N-K))
np.random.seed(2)

# assign model and fit
model = NearestNeighbors(n_neighbors=10,metric="jaccard")
model.fit(strain_df)

# Calculate nearest neighbors strains based on user input for flavors and effects (arr)
results = model.kneighbors(strain_array.reshape(1, -1))
print(results)
