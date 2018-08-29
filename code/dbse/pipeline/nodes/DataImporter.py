from dbse.pipeline.BaseNode import BaseNode
import pandas as pd
from dbse.tools.Visual import Visual
from dbse.tools.Logging import Logging
from sklearn.model_selection import train_test_split # sklearn 2.18+
from sklearn.decomposition import PCA
import numpy
from binstar_client.tests.urlmock import rule
from matplotlib import pyplot as plt
import common.data as dt
import os

class DataImporter(BaseNode):
    '''
    Loads a dataset. please use relative paths.

    Output is a dictionary

    mandatory

    - X[_test|_train]: Trainingdata
    - t[_test|_train]: T column
    - crit[_test|_train]: Criticality index, based on RUL
    - rul[_test|_train]: RUL column

    optional (if a field is existing, please make sure to specify an additional 
    $FIELD_NAME$_existing entry with a boolean saying "True")

    - X_final_test: E.g. from challenges, where this was the non public dataset
    - id[_test|_train]: object id in case existing

    
    Parameters:
        1: first argument needs to be of type string (name of dataset) specifiying the dataset
        2: second argument needs to be of type float (lowest X percent of RUL will be marked critical)
    '''

    def __init__(self, data_tag: str, rul_percent: float, enable_pca: bool):
        """
        Constructor
        :param data_tag: supported datatags are atm "rex" and "phm2008"
        :param rul_percent: the rul_percentage, that defines which samples are marked critical
        :param enable_pca: should a PCA be performed on the features?
        """

        # set parameters
        self._data_tag = data_tag
        self._rul_percent = rul_percent
        self._do_pca = enable_pca

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
        pass

    def run(self,  data=None):
        """
        does actual work
        :param data: 
        :return: 
        """
        super().run(dict())  # dont not remove this!
 
        if self._data_tag == "bmw_cells":
            train, test = dt.DataSource.load_bmw_cells()
            train_df = train.X
            train_df["t"] = train.t
            train_df["id"] = train.objectid
            train_df["rul"] = train.rul
            train_df["cluster_id"] = train.cluster
            train_df["ts"] = train.ts
            train_df = train_df[train_df["rul"] < 2500000]

            # R�ckwandlung - nur train_df test_df is ok
            train.t = train_df["t"] 
            train.objectid = train_df["id"]
            train.rul = train_df["rul"]
            train.cluster = train_df["cluster_id"]
            train.ts = train_df["ts"]
            train.X = train_df.drop(["t", "id", "rul", "cluster_id", "ts"], axis = 1)

            # filter df
            train.X = train.X[['STAT_FAKTOR_RS_WERT','STAT_TIME_TOTAL_WERT','STAT_ANZAHL_KUEHLVORGAENGE_WERT','STAT_HIS_PROG_LADEZEIT_5_WERT','STAT_HIS_PROG_LADEZEIT_6_WERT','STAT_HIS_PROG_LADEZEIT_7_WERT','STAT_HIS_SOC_WARN_GRENZEN_3_WERT','STAT_HISTO_SYM_DAUER_3_WERT','STAT_HISTO_SYM_ZELLANZAHL_4_WERT','STAT_HISTO_SYM_ZELLANZAHL_5_WERT','STAT_HISTO_SYM_ZELLANZAHL_6_WERT','STAT_I_HISTO_2_WERT','STAT_I_HISTO_3_WERT','STAT_I_HISTO_4_WERT','STAT_I_HISTO_6_WERT','STAT_I_HISTO_7_WERT','STAT_ENTLADUNG_KUEHLUNG_WERT','STAT_SCHUETZ_K2_RESTZAEHLER_WERT','STAT_MAX_SOC_GRENZE_WERT','STAT_ZEIT_SOC_12_WERT','STAT_ZEIT_TEMP_NO_OP_7_WERT','STAT_ZEIT_TEMP_TOTAL_5_WERT','STAT_HIS_EFF_CURR_CHG_1_TMID_WERT','STAT_FAKT_P2_T3_SOC5_WERT','STAT_RELATIVZEIT_4_WERT','STAT_ZEIT_POWER_CHG_2_WERT','MV___FAHRZEUGALTER']]
            test.X = test.X[['STAT_FAKTOR_RS_WERT','STAT_TIME_TOTAL_WERT','STAT_ANZAHL_KUEHLVORGAENGE_WERT','STAT_HIS_PROG_LADEZEIT_5_WERT','STAT_HIS_PROG_LADEZEIT_6_WERT','STAT_HIS_PROG_LADEZEIT_7_WERT','STAT_HIS_SOC_WARN_GRENZEN_3_WERT','STAT_HISTO_SYM_DAUER_3_WERT','STAT_HISTO_SYM_ZELLANZAHL_4_WERT','STAT_HISTO_SYM_ZELLANZAHL_5_WERT','STAT_HISTO_SYM_ZELLANZAHL_6_WERT','STAT_I_HISTO_2_WERT','STAT_I_HISTO_3_WERT','STAT_I_HISTO_4_WERT','STAT_I_HISTO_6_WERT','STAT_I_HISTO_7_WERT','STAT_ENTLADUNG_KUEHLUNG_WERT','STAT_SCHUETZ_K2_RESTZAEHLER_WERT','STAT_MAX_SOC_GRENZE_WERT','STAT_ZEIT_SOC_12_WERT','STAT_ZEIT_TEMP_NO_OP_7_WERT','STAT_ZEIT_TEMP_TOTAL_5_WERT','STAT_HIS_EFF_CURR_CHG_1_TMID_WERT','STAT_FAKT_P2_T3_SOC5_WERT','STAT_RELATIVZEIT_4_WERT','STAT_ZEIT_POWER_CHG_2_WERT','MV___FAHRZEUGALTER']]


        elif self._data_tag == "weather":
            train, test = dt.DataSource.load_weather()
        
        elif self._data_tag == "bmw_rex":
            train, test = dt.DataSource.load_bmw_rex()
            
        elif self._data_tag == "phm2008":
            train, test, tx = dt.DataSource.load_phm()
            
        elif self._data_tag == "phm2008_final":
            train, tx, test = dt.DataSource.load_phm()
            
        elif self._data_tag == "turbofan":
            train, test, tx = dt.DataSource.load_turbofan()
                  
        elif self._data_tag == "turbofan_final":
            train, tx, test = dt.DataSource.load_turbofan()        
        
        else:
            raise ImportError("No valid tag given in DataImporterNode - Valid tags are: rex, phm2008\nWas given:",self._data_tag)
        data = {}
        data["train_t"] = train.t
        data["train_X"] = train.X
        data["train_id"] = train.objectid
        data["train_rul"] = train.rul
        data["train_cluster_id"] = train.cluster
        data["train_ts"] = train.ts
        data["test_t"] = test.t
        data["test_X"] = test.X
        data["test_id"] = test.objectid
        data["test_rul"] = test.rul
        data["test_cluster_id"] = test.cluster
        data["test_ts"] = test.ts
        data["meta_dataset_name"] = self._data_tag
        
        # add other datasets here
        data["train_crit"] = numpy.array([1]* len(data["train_X"]))
        data["test_crit"] = numpy.array([1]* len(data["test_X"]))
        
        # define  risk based on linear mapping
        # training
        rul_max = data["train_rul"].max() 
        m = 1.0/float(rul_max)
        data["train_risc"] = 1 - m* data["train_rul"]

        # muss selbe Umrechnung sein wie oben - rueckrechnung auf RUL sollte immer noch gleich sein
        data["test_risc"] = 1 - m* data["test_rul"]
        
        metrics = dict()  # empty metrics
        #Visual().plot_scatter(data["train_rul"], data["train_risc"])
        #plt.show()
        return data, metrics

