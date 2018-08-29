from dbse.pipeline.BaseNode import BaseNode
import numpy as np
from dbse.tools.Logging import Logging
from dbse.tools.Visual import Visual
from sklearn import preprocessing

class RiscCalculator(BaseNode):
    """
    Calculates the risk score numerically based on the rul
    
    Parameters are
    1: first parameter needs to be of type string (\'lin\' or \'quad\'). This specifies how the risk is generated
    2: second parameter needs to be of type int (rul percentile). This specified which percentiles are filtered out.
       Recommmended value is 99
       
    Polyfit - kurve - dann Gausverteilt drumrum
    """

    def __init__(self, target_trend: str, rul_percentile: int,
                 field_in_train_rul_crit="train_rul_crit",
                 field_in_train_X_scaled_crit="train_X_scaled_crit",
                 field_in_test_rul_crit="test_rul_crit",
                 field_in_test_X_scaled_crit="test_X_scaled_crit",
                 field_out_train_risc="train_risc",
                 field_out_test_risc="test_risc",
                 field_out_train_X_scaled_crit_bounded_scaled="train_X_scaled_crit_bounded_scaled",
                 field_out_test_X_scaled_crit_bounded_scaled="test_X_scaled_crit_bounded_scaled"):
        """
        
        Constructor
        :param target_trend: interpolation of risc, options are "lin" or "quad"
        :param rul_percentile: percentile to be considered for risc (to filter out outliers)
        :param field_in_train_rul_crit: holding RUL for all samples marked critical
        :param field_in_train_X_scaled_crit: holding the scaled features of all samples marked critical
        :param field_in_test_rul_crit: holding RUL for all samples marked critical (test)
        :param field_in_test_X_scaled_crit: holding the scaled features of all samples marked critical (train)
        :param field_out_train_risc: holding the calculated risk for all samples
        :param field_out_test_risc: holding the calculated risk for all samples (test)
        :param field_out_train_X_scaled_crit_bounded_scaled: the bounded (RUL within percentile) and scaled samples 
                                                             belonging to field_out_train_risc
        :param field_out_test_X_scaled_crit_bounded_scaled: the bounded (RUL within percentile) and scaled samples 
                                                             belonging to field_out_train_risc (test)
        """
        # parameters
        self._target_trend = target_trend
        self._rul_percentile = rul_percentile

        # fields
        self._field_in_train_rul_crit = field_in_train_rul_crit
        self._field_in_train_X_scaled_crit = field_in_train_X_scaled_crit
        self._field_in_test_rul_crit = field_in_test_rul_crit
        self._field_in_test_X_scaled_crit = field_in_test_X_scaled_crit
        self._field_out_train_risc = field_out_train_risc
        self._field_out_test_risc = field_out_test_risc
        self._field_out_train_X_scaled_crit_bounded_scaled = field_out_train_X_scaled_crit_bounded_scaled
        self._field_out_test_X_scaled_crit_bounded_scaled = field_out_test_X_scaled_crit_bounded_scaled

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
        #TODO
    
    def run(self, data):
        super().run(data)  # dont not remove this!

        # temporary dictionary
        temp = dict() # use this for all data, that is not referenced by self._field

        # Assign and model risk
        rul_percentile_value = np.percentile(data[self._field_in_train_rul_crit], self._rul_percentile)
        Logging().log("any rul value larger than {0:.1f} will be dropped.".format(rul_percentile_value))

        indices_train = np.array(data[self._field_in_train_rul_crit] <= rul_percentile_value)
        temp["train_rul_crit_bounded"] = data[self._field_in_train_rul_crit][indices_train] # _bounded = only samples
                                                                                            # with RUL in percentile
        temp["train_X_scaled_crit_bounded"] = data[self._field_in_train_X_scaled_crit][indices_train]

        indices_test = np.array(data[self._field_in_test_rul_crit] <= rul_percentile_value)
        temp["test_rul_crit_bounded"] = data[self._field_in_test_rul_crit][indices_test]
        temp["test_X_scaled_crit_bounded"] = data[self._field_in_test_X_scaled_crit][indices_test]

        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(temp["train_X_scaled_crit_bounded"])

        data[self._field_out_train_X_scaled_crit_bounded_scaled] = scaler.transform(temp["train_X_scaled_crit_bounded"])
        data[self._field_out_test_X_scaled_crit_bounded_scaled] = scaler.transform(temp["test_X_scaled_crit_bounded"])

        # first, we calculate the risk for all critical samples based on the rul
        rul_min = np.min(temp["train_rul_crit_bounded"])
        rul_max = np.max(temp["train_rul_crit_bounded"])
        Logging().log("max RUL in bounded training dataset is {} (RISK = 1), min is {} (RISK = 0).".format(rul_max, rul_min))

        data[self._field_out_train_risc] = self._get_risc_target(temp["train_rul_crit_bounded"])
        data[self._field_out_test_risc] = self._get_risc_target(temp["test_rul_crit_bounded"])

        Visual().plot_scatter(temp["train_rul_crit_bounded"], data[self._field_out_train_risc])

        #for field in ["train", "test", "valid"]:
        #    field_real = "rul_" + field
        #    if field_real in data:
        #        data["risk_" + field] = self._get_risc_target(data[field_real])
        #        Visual().plot_scatter(data[field_real], data["risk_" + field])

        # metrics
        metrics = dict()  # empty metrics

        return data, metrics

    def _get_risc_target(self, ruls ):
        """
        gets the risc for the ruls
        :param ruls: the ruls to get the risc for
        :return: the riscs
        """
        rul_max = np.max(ruls)
        mode = self._target_trend
        if mode is 'lin':   # linear
            return 1 - (ruls/rul_max)
        if mode is 'quad':  # quadratic
            return ((1/rul_max)*(ruls - rul_max)**2)/rul_max

