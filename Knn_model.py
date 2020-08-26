# algorithm
import numpy as np
import pandas as pd
from scipy.spatial import distance

class KNearestNeighbor():
    """Unsupervise NN algorithm"""

    def fit(self, X):
        self.X = X
        self.features = self.X.drop(self.X.columns[0], axis=1)
        self.preds_labels = []
        self.distances = []

    def prediction(self, y_train, k=3, metric = "euclidian"):
        """predicts nearest neighbor for user-supplied input
         k represents the amount of neighbors to compare data with. 
         That is why it usually k is an odd number.
        the bigger the k, the less 'defined' or more smooth
         are the areas of classification.
        """
        assert k <= len(X), "[!] k can't be larger than number of samples."

        # assert len(y_train) == len(self.X.columns) - 1

        if metric == "jaccard":
            for i in range(len(self.X)):
                self.distances.append(distance.jaccard(y_train, 
                                                        self.features.iloc[i]))
                self.preds_labels.append(self.X[self.X.columns[0]].iloc[i])

            self.distance_df = pd.DataFrame([self.preds_labels, self.distances], 
                                            index=["pred_labels","distances"]).T
            return self.distance_df.sort_values("distances")[:k]

        else:

            for i in range(len(self.X)):
                self.distances.append(distance.euclidean(y_train, 
                                                        self.features.iloc[i]))
                self.preds_labels.append(self.X[self.X.columns[0]].iloc[i])

            self.distance_df = pd.DataFrame([self.preds_labels, self.distances], 
                                            index=["preds_labels","distances"]).T
            return self.distance_df.sort_values("distances")[:k]


if __name__ == "__main__":
    pass
