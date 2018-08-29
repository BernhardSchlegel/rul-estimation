import numpy as np
import pickle  # dumping models
from pathlib import Path

# own
from abc import ABCMeta, abstractmethod
from dbse.pipeline.BaseNode import BaseNode
from dbse.pipeline.helper.PolyFitter import PolyFitter
from dbse.tools.Logging import Logging

class TendencyFeatureFilter(BaseNode):
    __metaclass__ = ABCMeta

    def __init__(self, select_above_rand=True, n_top_features: int=50, reload_if_existing = True,
                 model_filename = "tendency_features",
                 field_in_train_X_scaled_crit_bounded_scaled="train_X_scaled_crit_bounded_scaled",
                 field_out_train_X_scaled_crit_bounded_scaled_top="train_X_scaled_crit_bounded_scaled_top",
                 field_in_test_X_scaled_crit_bounded_scaled="test_X_scaled_crit_bounded_scaled",
                 field_out_test_X_scaled_crit_bounded_scaled_top="test_X_scaled_crit_bounded_scaled_top"
                 ):
        """        
        Initializes the PipelineNode using the given parameters
        :param select_above_rand: If this is set to true, a random feature will be generated. This will be used as 
                                  marker: All features with a higher importance will be selected, all features with 
                                  a lower importance, dropped. If this is set to true, n_top_features has no effect.
        :param n_top_features: an integer specifying the number of parameters to keep
        :param reload_if_existing: If select_above_rand was used, the selected features are stored to disk. If this is
                                   set to true, the same features will be used again.
        :param field_in_train_X_scaled_crit_bounded_scaled: Scaled, critical samples that are within RUL percentil
        :param field_out_train_X_scaled_crit_bounded_scaled_top: Same samples, but subset of features.
        :param field_in_test_X_scaled_crit_bounded_scaled: Scaled, critical samples that are within RUL percentil (test)
        :param field_out_test_X_scaled_crit_bounded_scaled_top: Same samples as above, but subset of features (test)
        """
        self._n_top_features = n_top_features
        self._select_above_rand = select_above_rand
        self._reload_if_existing = reload_if_existing
        self._model_filename = model_filename

        # fields
        self._field_in_train_X_scaled_crit_bounded_scaled = field_in_train_X_scaled_crit_bounded_scaled
        self._field_out_train_X_scaled_crit_bounded_scaled_top = field_out_train_X_scaled_crit_bounded_scaled_top
        self._field_in_test_X_scaled_crit_bounded_scaled = field_in_test_X_scaled_crit_bounded_scaled
        self._field_out_test_X_scaled_crit_bounded_scaled_top = field_out_test_X_scaled_crit_bounded_scaled_top

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
        assert 'meta_dataset_name' in data, "missing meta_dataset_name key in data dictionary"
        assert self._field_in_train_X_scaled_crit_bounded_scaled in data, "missing {} key in data dictionary".\
            format(self._field_in_train_X_scaled_crit_bounded_scaled)
        assert self._field_in_test_X_scaled_crit_bounded_scaled in data, "missing {} key in data dictionary".\
            format(self._field_in_test_X_scaled_crit_bounded_scaled)

    def run(self, data):
        super().run(data)  # dont not remove this!

        # determine feature weights
        pf = PolyFitter()

        # create temporary dictionary for data that is not passed to the next stage
        temp = dict()
        X = data['train_X_scaled_crit_bounded_scaled']
        n_samples = X.shape[0]
        n_idx_random = X.shape[1] # features are zero indexed, so the number is equal to the index of the new feature
        rand_feature = np.random.randn(n_samples)
        temp['t_X_s_c_b_s_enhanced'] = np.c_[X, rand_feature] # hstack
        data['meta_feature_weights'] = pf.get_weights(temp['t_X_s_c_b_s_enhanced'], data['train_risc'])

        # select top features
        if self._select_above_rand:
            Logging().log("selecting feature above random feature")

            # restore model from file in case existing
            model_filename = data['meta_dataset_name'] + "_" + self._model_filename
            if self._reload_if_existing is False or Path(model_filename).exists() is False:
                data['meta_feature_indices'] = pf.get_feature_idices_above_rand(data['meta_feature_weights'],
                                                                                n_idx_random=n_idx_random)
                # save model to file
                with open(model_filename, 'wb') as f:
                    pickle.dump(data['meta_feature_indices'], f)

            else:
                Logging().log("restoring model from {}".format(model_filename))
                with open(model_filename, 'rb') as fid:
                    data['meta_feature_indices'] = pickle.load(fid)

            Logging().log("selected {} features from {}".format(len(data['meta_feature_indices']), n_idx_random))
        else:
            Logging().log("selecting top {} features".format(self._n_top_features))
            data['meta_feature_indices'] = pf.get_top_feature_idices(data['meta_feature_weights'], self._n_top_features)


        data[self._field_out_train_X_scaled_crit_bounded_scaled_top] = pf.get_top_features(
            data[self._field_in_train_X_scaled_crit_bounded_scaled], data['meta_feature_indices'])
        data[self._field_out_test_X_scaled_crit_bounded_scaled_top] = pf.get_top_features(
            data[self._field_in_test_X_scaled_crit_bounded_scaled], data['meta_feature_indices'])

        # aggregate metrics
        metrics = dict()  # empty

        return data, metrics
