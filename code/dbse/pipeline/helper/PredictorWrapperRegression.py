import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle  # dumping models
from pathlib import Path

# own libs
from dbse.tools.Logging import Logging
from dbse.pipeline.scoring.RMSE import RMSE

class PredictorWrapperRegression:
    """
    encapsules training of models
    """

    @staticmethod
    def train(X_train, X_test, y_train, y_test, model_filename, reload_if_existing,
              modeltype="RF"):
        """
        trains and evaluates a model based on the given data
        :param X_train: Features for training, expected to a numpy.ndarray.
        :param X_test: Features for testing, expected to a numpy.ndarray.
        :param y_train: Labels for training. Expected to an one-dimesional array.
        :param y_test: Labels for testing. Expected to an one-dimesional array.
        :param model_filename: Filename of model when serialized to disk
        :param reload_if_existing: Boolean indicating if model should be restored from disk if existing.
        :param modeltype: modeltype to train (RF, SVC or LRCV). RF is recommended since being fast to train and non-
                          linear - therefore usually yielding the best results.
        :return: 
        """
        if reload_if_existing is False or Path(model_filename).exists() is False:
            Logging().log("training {}. ".format(modeltype))
            if modeltype is "RF":
                param_grid = {'max_depth': [3, 5, 10, 15, 20],
                              'n_estimators': [3, 5, 10, 20]}

                clf = GridSearchCV(RandomForestRegressor(n_jobs=-1), param_grid)

                mdl = clf.fit(X_train, y_train)

            # output model quality
            rmse = RMSE.score(y_test, mdl.predict(X_test))
            Logging().log("Mean squared error: {0:.3}".format(rmse))

            # save model to file
            with open(model_filename, 'wb') as f:
                pickle.dump((mdl, rmse), f)

        else:
            Logging().log("restoring model from {}".format(model_filename))

            with open(model_filename, 'rb') as fid:
                mdl, rmse = pickle.load(fid)

        return mdl, rmse
