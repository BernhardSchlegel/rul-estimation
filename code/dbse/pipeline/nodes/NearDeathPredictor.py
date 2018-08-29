'''
Created on 08.08.2017

@author: Q372283, Q416435
'''
from dbse.pipeline.BaseNode import BaseNode
import pandas as pd
from dbse.tools.Visual import Visual
from dbse.tools.Logging import Logging
from sklearn import preprocessing

# custom
from dbse.pipeline.helper.PredictorWrapperClassification import PredictorWrapperClassification


class NearDeathPredictor(BaseNode):
    """
    Calculates the risk score. Risk is binary: 0 = ok, 1 = sample is close to death
    
    train_rul holds the corresponding RUL for the train_X values. These array
    will be mapped to a near death indicator according to RUL_percent. 
    
    a sample is considered near death if less than RUL_percent are left. Example:
    If the max observed RUL is 5000 and RUL_percent is 10%, any sample with a RUL 
    below 500 is considered critical.    
    """

    def __init__(self, reload_if_existing: bool,
                 key_in_train_x='train_X',
                 key_in_test_x='test_X',
                 key_in_train_crit='train_crit',
                 key_in_test_crit='test_crit',
                 key_out_train_x='train_X_scaled_crit',
                 key_out_test_x='test_X_scaled_crit',
                 key_out_train_crit='train_rul_crit',
                 key_out_test_crit='test_rul_crit',
                 key_in_train_rul='train_rul',
                 key_in_test_rul='test_rul'):
        """
        Constructor
        :param reload_if_existing: If true, model will be loaded from disc if existing.
        """

        # set parameters
        self._reload_if_existing = reload_if_existing

        # variables
        self._model_crit = None  # holding the model
        self._model_auc = 0  # holding the model AUC after training
        self._key_in_train_x = key_in_train_x
        self._key_in_test_x = key_in_test_x
        self._key_in_train_crit = key_in_train_crit
        self._key_in_test_crit = key_in_test_crit
        self._key_in_train_rul = key_in_train_rul
        self._key_in_test_rul = key_in_test_rul
        self._key_out_train_x = key_out_train_x
        self._key_out_test_x = key_out_test_x
        self._key_out_train_crit = key_out_train_crit
        self._key_out_test_crit = key_out_test_crit
        super().__init__()

    def check_params(self):
        """
        Checks the parameter dictionary matches the requirements.
        :return: True if the parameters match the requirements, else False.
        """

    def check_data(self, data):
        """
        Checks if the given data dictionary matches the requirements.
        :param data: the data dictionary to be checked
        :return: True if the data match the requirements, else False.
        """
        assert self._key_in_train_x in data, 'field {} (features) is required'.format(self._key_in_train_x)
        assert self._key_in_test_x in data, 'field {} (features) is required'.format(self._key_in_test_x)
        assert self._key_in_train_crit in data, 'field {} (criticality) is required since being modelling target'.\
            format(self._key_in_train_crit)
        assert self._key_in_test_crit in data, 'field {} (criticality) is required since being modelling target'.\
            format(self._key_in_test_crit)


    def run(self, data):
        super().run(data)  # dont not remove this!

        # preprocessing - scaling X
        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(data[self._key_in_train_x])
        temp = dict()
        temp['train_X_scaled'] = scaler.transform(data[self._key_in_train_x])
        temp['test_X_scaled'] = scaler.transform(data[self._key_in_test_x])

        # Train model for criticality
        self._model_crit, self._model_auc = PredictorWrapperClassification.\
            train(temp['train_X_scaled'], temp['test_X_scaled'], data[self._key_in_train_crit],
                  data[self._key_in_test_crit], data['meta_dataset_name'] + "_crit_pred", self._reload_if_existing)

        # select critical samples based on model
        data[self._key_out_train_x], data[self._key_out_train_crit] = self._crit_get_varargs(temp['train_X_scaled'],
                                                                                     data[self._key_in_train_rul])
        data[self._key_out_test_x], data[self._key_out_test_crit] = self._crit_get_varargs(temp['test_X_scaled'],
                                                                                   data[self._key_in_test_rul])

        # plot histogram to show awesome results
        Logging.log("RUL histogram of complete test population")
        max_rul = data[self._key_in_test_rul].max()
        Visual.plot_hist(data[self._key_in_test_rul], max_x=max_rul)
        Logging.log("RUL histogram of sub test population labeled critical")
        Visual.plot_hist(data[self._key_out_test_crit], max_x=max_rul)

        # metrics
        metrics = dict()  # empty metrics
        metrics['model_auc'] = self._model_auc

        return data, metrics

    def _crit_gex_index(self, X, threshold=0.5):
        """
        Picks samples from X if their prediction is above the threshold
        :param X: the samples to be evaluated
        :param threshold: that has to be exceeded to be considered critical
        :return: a subset of samples
        """
        return self._model_crit.predict_proba(X)[:, 1] > threshold

    def _crit_get_varargs(self, *args, threshold=0.5):
        """
        Selects subsets of samples from all args, based on the features given by the first arg.
        
        :param args: First argument is expected to some sort of X that holds all features, needed for
                     prediction. These predictions will be used to generate indices to pick samples. Not
                     only for X, but also for all following datastructures.
        :param threshold: The threshold, if exceeded, sample is considered critical.
        :return: a subset of critical samples for all args
        """
        nd_indices = None
        return_value = []
        for arg in args:
            if nd_indices is None:
                # first iteration, do predictions
                nd_indices = self._crit_gex_index(arg, threshold)

            if isinstance(arg, pd.DataFrame):
                arg_crit = arg.loc[nd_indices]
            elif arg.ndim is 1:
                arg_crit = arg[nd_indices]
            else:
                arg_crit = arg[nd_indices, :]

            return_value.append(arg_crit)

        return tuple(return_value)
