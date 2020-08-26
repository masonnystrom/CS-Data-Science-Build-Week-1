# algorithm 
import numpy
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
    assert len(y_train) == len(self.X.columns) - 1
    #assert k <= len(X), "[!] k can't be larger than number of samples."

    if metric == "jaccard":
        for i in range(len(self.X)):
            self.distances.append(distance.jaccard(y, self.features.iloc[i]))
            self.ids.append(self.X[self.X.columns[0]].iloc[i])

    self.distances_df = pd.DataFrame([self.id, self.distances],
                                            index = ["preds_lables", "distances"]).T 
    return self.distances_df.sort_values("distances")[:k]

    else:

        for i in range(len(self.X)):
            self.distances.append(distance.euclidean(y, self.features.iloc[i]))
            self.ids.append(self.X[self.X.columns[0]].iloc[i])

    self.distances_df = pd.DataFrame([self.ids, self.distances], 
                                            index = ["preds_labels", "distances"]).T 
    return self.distances_df.sort_values("distances")[:k]


if __name__ == "__main__":
    pass
