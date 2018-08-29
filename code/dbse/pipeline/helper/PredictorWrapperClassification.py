

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import multiprocessing as mp
import pickle  # dumping models
from pathlib import Path

# own libs
from dbse.tools.Logging import Logging


class PredictorWrapperClassification:
    """
    encapsules training of models
    """

    @staticmethod
    def train(X_train, X_test, y_train, y_test, model_filename, reload_if_existing,
              modeltype="RF", cv_measure="roc_auc_score"):
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
        :param cv_measure: possible cv_measure are ['accuracy', 'precision', 'recall', 'roc_auc']
        :return: 
        """
        if reload_if_existing is False or Path(model_filename).exists() is False:
            Logging().log("training {}. ".format(modeltype))
            if modeltype is "LRCV":
                Logging().log("Optimizing for {}...".format(cv_measure))
                lr = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1],
                                          cv=5,
                                          penalty='l1',
                                          scoring=cv_measure,  # Changed from auROCWeighted
                                          solver='liblinear',
                                          tol=0.001,
                                          n_jobs=mp.cpu_count())
                mdl = lr.fit(X_train, y_train)
                Logging().log("cross validated {0} (train) is {1:.3}".format(cv_measure, max(
                    np.mean(mdl.scores_[1], axis=0))))  # get CV train metrics
            elif modeltype is "SVC":
                # after ~2h of training: cross validated roc_auc=0.511 on rex
                clf = SVC()
                mdl = clf.fit(X_train, y_train)
            elif modeltype is "RF":
                # after ~2h of training: cross validated roc_auc=0.511 on rex

                param_grid = {'max_depth': [3, 5, 10, 15, 20],
                              'n_estimators': [3, 5, 10, 20]}

                clf = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid)

                mdl = clf.fit(X_train, y_train)

            # output model quality
            cross_val_res = cross_val_score(mdl, X_test, y_test, scoring='roc_auc')
            auc_test = np.mean(cross_val_res)
            Logging().log("cross validated AUC (test) is {0:.3}".format(auc_test))

            # save model to file
            with open(model_filename, 'wb') as f:
                pickle.dump((mdl, auc_test), f)

        else:
            Logging().log("restoring model from {}".format(model_filename))

            with open(model_filename, 'rb') as fid:
                (mdl, auc_test) = pickle.load(fid)

        return mdl, auc_test
