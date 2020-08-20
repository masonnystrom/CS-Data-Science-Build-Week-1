# algorithm 
import numpy
import scipy

class Node():
""" Helper class that will """
    pass


class DecisionTree():
    """ Decision Tree class that for fitting and predicting

    Inputs: X train, y train and a max_depth
    Outputs: Decision tree fit and prediction. 
    """
    def __ini__(self, X, y, max_depth = 5):
        self.X = X
        self.y = y
        self.depth = 0
        self.max_depth = max_depth

    def split_data(data, column, value):
        split_column_values = data[:, column]
        train = data[split_column_values <= value]
        test = data[split_column_values >= value]
    
    def best_split():
        # can use cross entropy or gini index
        # can use recurssion
        pass

    def fit():
        pass


    def predict():
        pass


if __name__ == "__main__":
    pass
