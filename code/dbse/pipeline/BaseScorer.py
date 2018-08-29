from abc import ABCMeta, abstractmethod
import numpy as np

class BaseScorer(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Initializes the BaseScorer using the given parameters
        """

    @abstractmethod
    def score(self, labels: np.ndarray, predictions: np.ndarray):
        """
        does the scoring
        :param labels: actual label
        :param predictions: the model based preductions
        :return: the score as float
        """
        raise NotImplementedError("Must be overwritten by child-class.")
