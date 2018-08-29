from dbse.pipeline.BaseScorer import BaseScorer
import numpy as np

class RMSE(BaseScorer):
    """
    Calculates the RMSE scores, e.g. for RUL
    """

    def __init__(self):
        """
        Initializes the BaseScorer using the given parameters
        """

    @staticmethod
    def score(labels: np.ndarray, predictions: np.ndarray):
        """
        does the scoring
        :param labels: actual label
        :param predictions: the model based preductions
        :return: the score as float
        """
        return np.mean(predictions - labels) ** 2
