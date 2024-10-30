import numpy as np
from operations import find_mode


class PredictMode():
    def __init__(self):
        self.most_common_class = None

    def fit(self, features, labels):
        self.most_common_class = find_mode(labels)

    def predict(self, features):
        n = features.shape[0]
        return np.full(n, self.most_common_class)
    
    def visualize(self, **kwargs):
        print(f"Predict {self.most_common_class}")
