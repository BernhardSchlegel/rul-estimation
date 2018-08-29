import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PolyFitter:


    def _scale_to_0_1(self, feature_values):
        feature_max = np.max(feature_values)
        feature_min = np.min(feature_values)
        if feature_min is not feature_max:
            feature_values = (feature_values - feature_min) / (feature_max - feature_min)

        return feature_values

    def get_weights(self, X, y, show_plots=False):
        """
        
        :param X: in most cases, X will be the features
        :param y: in most cases, y will be the risc
        :return: the weight of all features
        """
        n_features = X.shape[1]

        polys_a = []
        polys_b = []
        polys_c = []
        rmses = []
        for i in range(n_features):
            feature_values = X[:, i]
            feature_values = self._scale_to_0_1(feature_values)

            poly2_params = np.polyfit(feature_values, y, 2, full=True)
            poly2 = np.poly1d(poly2_params[0])
            xp = np.linspace(np.min(feature_values), np.max(feature_values), 100)
            if len(poly2_params[1]) is 0:
                RMSE = max(rmses)
            else:
                RMSE = poly2_params[1][0]

            if show_plots:
                plt.plot(feature_values, y, '.', xp, poly2(xp), '-', xp, poly2(xp), '--')
                plt.xlabel('feature {0}, a={1:.3}, b={2:.3}, c={3:.3}, RMSE={4:.3}'
                           .format(i, poly2_params[0][0], poly2_params[0][1], poly2_params[0][2], RMSE))
                plt.ylabel('risk')
                plt.show()

            polys_a.append(poly2_params[0][1])
            polys_b.append(poly2_params[0][1])
            polys_c.append(poly2_params[0][2])
            rmses.append(RMSE)

        weights = np.divide(np.add(polys_a, polys_a), rmses)  # is equal to (polys_a + polys_b) / rmses
        return weights

    @staticmethod
    def get_top_feature_idices(w, num=50):
        """

        :param w: the weights to choose from
        :param num: number of weights to choose
        :return: 
        """
        assert isinstance(w, list) or isinstance(w, np.ndarray), 'w needs to be a numpy.ndarray or list'

        weights = pd.DataFrame(np.abs(w))
        weights['feature_num'] = np.arange(0, len(w))
        return np.array(weights.sort_values(by=0).tail(num)['feature_num'])

    @staticmethod
    def get_feature_idices_above_rand(w, n_idx_random):
        """
        Get feature indices sorted by their weight
        :param w: the weights to choose from
        :param n_idx_random: Index of random feature
        :return: 
        """
        assert isinstance(w, list) or isinstance(w, np.ndarray), 'w needs to be a numpy.ndarray or list'

        weights = pd.DataFrame(np.abs(w))
        weights['feature_num'] = np.arange(0, len(w))
        indices_sorted = np.array(weights.sort_values(by=0)['feature_num'])

        pos_rnd = np.where(indices_sorted == n_idx_random)[0][0]  # position of random feature
        indices_to_be_selected = indices_sorted[0:pos_rnd]  # indices of all features ranked higher than random

        return indices_to_be_selected

    @staticmethod
    def get_top_features(X, feature_indices):
        """
        get the features from X indexed by feature_indices
        :param X: the features to choose from
        :param feature_indices: 
        :return: 
        """
        assert isinstance(X, np.ndarray), 'X needs to be a numpy.ndarray'
        assert isinstance(feature_indices, np.ndarray), 'feature_indices needs to be a numpy.ndarray'
        return np.array(X)[:, feature_indices]
