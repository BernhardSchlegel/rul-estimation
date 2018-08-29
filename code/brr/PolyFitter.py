import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import scale

class PolyFitter:

    _indices = None
    _feature_names = []
    _weights = None

    def fit(self, X, y, show_plots=False, show_first_plot=False, num=50, scale_to_0_1 = False):
        self._weights = self.get_weights(X, y, show_plots=show_plots,
                                 show_first_plot=show_first_plot,
                                         scale_to_0_1 = scale_to_0_1)
        self._indices = self.get_top_feature_idices(self._weights, num=num)

    def select(self, X):
        return self.get_top_features(X, self._indices)

    @staticmethod
    def _scale_to_0_1(vals):
        feature_max = np.max(vals)
        feature_min = np.min(vals)
        if not math.isclose(feature_min, feature_max):
            vals = (vals - feature_min) / (feature_max - feature_min)
        else:
            vals = [0] * len(vals)
        return vals

    @staticmethod
    def get_weights(X, y, show_plots=True, show_first_plot=False, scaling_enabled=False, dump_plot_data_to_csv=True,
                    scale_to_0_1 = False):
        """
        
        :param X: in most cases, X will be the features
        :param y: in most cases, y will be the risc
        :return: the weight of all features
        """
        if isinstance(X, pd.DataFrame):
            X = X.as_matrix()

        if scaling_enabled:
            X = scale(X, axis=0, with_mean=True, with_std=True, copy=True)

        n_features = X.shape[1]

        polys_a = []
        polys_b = []
        polys_c = []
        quad_rmses = []
        lin_rmses = []
        dump_plot_data_to_csv = False
        for i in range(n_features):
            feature_values = X[:, i]
            feature_values = PolyFitter._scale_to_0_1(feature_values)

            if np.sum(feature_values) == 0:
                polys_a.append(0)
                polys_b.append(0)
                polys_c.append(0)
                quad_rmses.append(1000.0)
                lin_rmses.append(1000.0)
            else:
                poly2_params, res_quad, _, _, _ = np.polyfit(feature_values, y, 2, full=True)
                poly2 = np.poly1d(poly2_params)


                poly1_params, res_lin, _, _, _ = np.polyfit(feature_values, y, 1, full=True)
                poly1 = np.poly1d(poly1_params)

                xp = np.linspace(np.min(feature_values), np.max(feature_values), 100)
                if len(res_quad) is 0:
                    quad_RMSE = float(1000)
                else:
                    quad_RMSE = res_quad[0]
                if len(res_lin) is 0:
                    lin_RMSE = float(1000)
                else:
                    lin_RMSE = res_lin[0]

                if show_plots or show_first_plot:
                    plt.figure()
                    plt.plot(feature_values, y, '.',
                             xp, poly2(xp), '-',
                             xp, poly1(xp), '--')
                    plotstr = ('feature {0}\n'.format(i) +
                               'quad_fit: a={0:.3f}, b={1:.3f}, c={2:.3f}, RMSE={3:.3f}\n'.format(poly2_params[0], poly2_params[1], poly2_params[2], quad_RMSE) +
                               'lin_fit: a={0:.3}, b={1:.3f}, RMSE={2:.3f}'.format(poly1_params[0], poly1_params[1], lin_RMSE))
                    plt.xlabel("feature")
                    plt.ylabel('risk')
                    plt.title(plotstr)

                    if dump_plot_data_to_csv:

                        import uuid
                        uuid = str(uuid.uuid4())
                        from matplotlib2tikz import save as tikz_save
                        tikz_save("plots/" + uuid + "_plot.tex")
                        plt.savefig("plots/" + uuid + "_plot.png")


                        poly_x = xp
                        poly1_ys = poly1(xp)
                        poly2_ys = poly2(xp)
                        df_poly = pd.DataFrame(data={
                            "poly_x": poly_x,
                            "poly1_ys": poly1_ys,
                            "poly2_ys": poly2_ys
                        })
                        df_data = pd.DataFrame(data={
                            "feature_values": feature_values,
                            "y": y
                        })
                        df_poly.to_csv("plots/" + uuid + "_poly.csv", index=False)
                        df_data.to_csv("plots/" + uuid + "_data.csv", index=False)

                    plt.show(block=False)
                    show_first_plot = False

                polys_a.append(poly2_params[0])
                polys_b.append(poly1_params[0])
                polys_c.append(poly1_params[1])
                quad_rmses.append(quad_RMSE)
                lin_rmses.append(lin_RMSE)
        # calculate weights
        # np.divide(np.add(polys_a, polys_a), rmses)
        # which is equal to (polys_a + polys_b) / rmses
        weights = [(abs(a)+abs(b))/(rmse_quad + rmse_lin) for a, b, rmse_quad, rmse_lin in zip(polys_a, polys_b, quad_rmses, lin_rmses)]

        if scale_to_0_1:
            return PolyFitter._scale_to_0_1(weights) # scale to 0 - 1 for better visualization
        else:
            return np.array(weights)

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

        if num > weights.shape[0]:
            num = weights.shape[0]

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
