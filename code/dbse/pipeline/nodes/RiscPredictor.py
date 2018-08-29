'''
Created on 08.08.2017

@author: Q372283, Q416435
'''
from dbse.pipeline.BaseNode import BaseNode
import numpy as np
import pandas as pd
from dbse.tools.Logging import  Logging
from dbse.tools.Visual import Visual
from dbse.pipeline.helper.PolyFitter import PolyFitter

# custom
from dbse.pipeline.helper.PredictorWrapperRegression import PredictorWrapperRegression


class RiscPredictor(BaseNode):
    """
    Node for predicting the risc score.
    """

    def __init__(self, reload_if_existing=True,
                 field_in_train_X_scaled_crit_bounded_scaled_top="train_X_scaled_crit_bounded_scaled_top",
                 field_in_test_X_scaled_crit_bounded_scaled_top="test_X_scaled_crit_bounded_scaled_top",
                 field_in_train_risc="train_risc",
                 field_in_test_risc="test_risc"
                 ):

        """
        constructor
        :param reload_if_existing: If true, model will be loaded from disc if existing.
        :param field_in_train_X_scaled_crit_bounded_scaled_top: 
        :param field_in_test_X_scaled_crit_bounded_scaled_top: 
        :param field_in_train_risc: 
        :param field_in_test_risc: 
        """

        self._reload_if_existing = reload_if_existing

        # set member variables
        self._model_risk = None
        self._model_risk_rmse = None

        # fields
        self._field_in_train_X_scaled_crit_bounded_scaled_top = field_in_train_X_scaled_crit_bounded_scaled_top
        self._field_in_test_X_scaled_crit_bounded_scaled_top = field_in_test_X_scaled_crit_bounded_scaled_top
        self._field_in_train_risc = field_in_train_risc
        self._field_in_test_risc = field_in_test_risc

        # call parent constructor
        super().__init__()

    def check_params(self):
        return

    def check_data(self, data):
        return

    def run(self, data):
        super().run(data)  # dont not remove this!

        threshold = 1

        # predict criticality
        self._model_risk, self._model_risk_rmse = PredictorWrapperRegression.train(
            data["train_X_scaled_crit_bounded_scaled_top"], data["test_X_scaled_crit_bounded_scaled_top"],
            np.array(data["train_risc"]), np.array(data["test_risc"]), data['meta_dataset_name'] + "_risc_pred",
            self._reload_if_existing)

        below_threshold = np.array(data["train_risc"] < threshold)
        above_threshold = np.array(data["train_risc"] >= threshold)

        n_above = sum(above_threshold)
        n_below = sum(below_threshold)
        Logging().log('there are {} samples above and {} below the threshold'.format(n_above, n_below))
        n_above_percentage_to_be_balanced = n_below / (n_above + n_below)
        rand_nums = np.random.choice([1, 0], size=(n_above + n_below,),
                                     p=[n_above_percentage_to_be_balanced, (1 - n_above_percentage_to_be_balanced)])
        Logging().log('picked randomly a proportion of {0:.3}% samples from all samples ({1} samples)'.format(
            n_above_percentage_to_be_balanced, (n_above + n_below)))

        above_threshold_picked = rand_nums == 1 & above_threshold

        # Plot
        Visual().plot_scatter(self._model_risk.predict(data["test_X_scaled_crit_bounded_scaled_top"]), data["test_risc"])
        Visual().plot_scatter(self._model_risk.predict(data["train_X_scaled_crit_bounded_scaled_top"]), data["train_risc"])

        # aggregate metrics
        metrics = dict()
        metrics['model_rmse'] = self._model_risk_rmse

        return data, metrics

    def _scale_to_0_1(self, feature_values):
        """
        THIS IS TOTAL BAUSTELLE
        :param feature_values: 
        :return: 
        """
        feature_max = np.max(feature_values)
        feature_min = np.min(feature_values)
        if feature_min is not feature_max:
            feature_values = (feature_values - feature_min) / (feature_max - feature_min)

        return feature_values

    def _get_criticality(self, X: np.ndarray, w: list, f, feature_weights):
        """
        THIS IS TOTAL BAUSTELLE
        :param X: features
        :param w: weights
        :param f: 
        :param feature_weights: 
        :return: 
        """

        assert X.shape[1] is len(w), 'number of features does not match ({} to {})'.format(len(f), len(feature_weights))

        n_rows = X.shape[0]  # dimension of original dataset
        n_cols = X.shape[1]  # dito
        w_array = np.array(w).reshape(n_cols, 1)  # reshape is necessary (although dim doesnt really change)
        w_array_normalized = w  # normalized the weight

        return np.dot(self._scale_to_0_1(X), w_array_normalized)

