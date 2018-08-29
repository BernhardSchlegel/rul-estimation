import time
import datetime
import csv
import os.path

from common.tools import check_var_exists

"""
import like from common.log import log, ping, pong
"""

SUPRESS_LOG = True

def log(text, force = False):
    if not SUPRESS_LOG or (SUPRESS_LOG and force):
        print(time.strftime('%Y.%m.%d, %H:%M:%S') + ': ' + text)

def ping():
    return datetime.datetime.now()

def pong(dt):
    now = datetime.datetime.now()
    diff = now - dt
    ms = round(diff.total_seconds() * 1000)
    return ms


class ResultLogger:

    _header = ["TIME",              # the current time, automatically filled
               "DATASET_NAME",      # e.g. "nasa-phm", "nasa-turbofan", "bmw-rex", "bmw-battery", "weather"
               "DATASET_SUBTYPE",   # e.g. "test", "fina_test", e.g.
                "ALGORITHM_NAME",     # e.g. "wang_et_al", "artur"
                "TRAINING_TIME_MS",
                "TESTING_TIME_MS",
                "SCORE_RMSE",
                "SCORE_PHM",
                "SCORE_PHM_ALL",      # PHM score if all elements have been used for scoring, not just the last from each object
                "USE_CLUSTERS_FROM_DATASET"]  # 0 no clusters are used, 1 = cluster from dataset are used

    def __init__(self, approach_name="random_approach", additional_header_fields=[]):
        """
        Constructor
        :param approach_name: Set the name of the evaluated approach, e.g. "wang_et_al" or "artur", etc.
        :param additional_header_fields: you can specify additional header fields here, e.g. 
               ["HYPER_PETER1_SCALE_MODE", "HYPER_PETER_CHEATING"]
        """

        result_folder = "simulation_results"
        self._filename = result_folder + "/" + approach_name + "_results.csv"

        # add custom fields to default header
        self._header += additional_header_fields

        # check if file exists

        file_existed = False
        if os.path.isfile(self._filename):
            log("logfile {} already existing, attaching.".format(self._filename))
            file_existed = True
        else:
            log("logfile {} not existing, creating header.".format(self._filename))

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        if not file_existed:
            with open(self._filename, 'w') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=self._header, lineterminator="\n")
                csv_writer.writeheader()

    def log(self, values={"DATASET_NAME": "PHM", "ALGORITHM_NAME": "WANG_ET_AL"}):
        values["TIME"] = str(datetime.datetime.now())
        if len(values) != len(self._header):
            log("WARNING: LENGTH OF HEADER ({}) DOES NOT MATCH THE NUMBER OF VALUES ({})".format(len(self._header),
                                                                                                 len(values)) +
                ". SOMETHING IS WRONG.")

        with open(self._filename, 'a') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=self._header, lineterminator="\n")
            csv_writer.writerow(values)

