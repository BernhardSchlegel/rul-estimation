import pandas as pd
import numpy as np

"""
import like
from common.data import DataSource, DataStructure
"""
import numpy

class DataStructure:

    X = pd.DataFrame()         # the features
    y = pd.DataFrame()         # the labels
    t = pd.DataFrame()         # the t values (timesteps until failure are negative)
    rul = pd.DataFrame()       # the rul of each sample
    objectid = pd.DataFrame()  # the objectid of all samples
    cluster = pd.DataFrame()   # the cluster, to which the sample belongs
    ts = pd.DataFrame()        # the timestamp of each sample

class DataSource:

    @staticmethod
    def to_ingest_set(train, test):
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

        # add other datasets here
        data["train_crit"] = numpy.array([1]* len(data["train_X"]))
        data["test_crit"] = numpy.array([1]* len(data["test_X"]))

        # define  risk based on linear mapping
        # training
        rul_max = data["train_rul"].max()
        m = 1.0/float(rul_max)
        data["train_risc"] = 1 - m* data["train_rul"]

        # test
        data["test_risc"] = 1 - m* data["test_rul"]

        return data

    @staticmethod
    def load_clean(X):

        X = pd.DataFrame(X)
        X = X.sort_values(["OBJECTID", "TS"], ascending=[True, True])

        t = X['T']
        rul = X['RUL']
        objectid = X['OBJECTID']
        cluster = X['OPERATIONAL_SETTING']
        ts = X['TS']

        del X['T']
        del X['RUL']
        del X['OBJECTID']
        del X['OPERATIONAL_SETTING']
        del X['TS']

        return X, t, rul, objectid, cluster, ts

    @staticmethod
    def _load_dataset_nasa_helper(X):

        X = pd.DataFrame(X)
        X = X.sort_values(["OBJECTID", "TS"], ascending=[True, True])

        t = X['T']
        rul = X['RUL']
        objectid = X['OBJECTID']
        cluster = X['OPERATIONAL_SETTING']
        ts = X['TS']

        del X['T']
        del X['RUL']
        del X['OBJECTID']
        del X['OPERATIONAL_SETTING']
        del X['TS']

        return X, t, rul, objectid, cluster, ts

    @staticmethod
    def _load_dataset_bmw_helper(X):

        X = pd.DataFrame(X)
        X = X.sort_values(["OBJECTID", "TS"], ascending=[True, True])

        t = X['T']
        rul = X['RUL']
        objectid = X['OBJECTID']
        cluster = [0] * X.shape[0]
        ts = X['TS']

        del X['T']
        del X['RUL']
        del X['OBJECTID']
        del X['TS']

        return X, t, rul, objectid, cluster, ts

    @staticmethod
    def _load_dataset_weather_helper(X):

        X = pd.DataFrame(X)
        X = X.sort_values(["OBJECTID", "TS"], ascending=[True, True])

        t = X['T']
        rul = X['RUL']
        objectid = X['OBJECTID']
        cluster = X['CLUSTER']
        ts = X['TS']

        del X['T']
        del X['RUL']
        del X['OBJECTID']
        del X['CLUSTER']
        del X['TS']

        return X, t, rul, objectid, cluster, ts

    @staticmethod
    def load_turbofan(base_path="../../Data/nasa-turbofan/ready/",
                      train_path="turbofan_phm_simulation_train_218.csv",
                      test_path="turbofan_phm_simulation_test_218.csv",
                      final_test_path="turbofan_phm_simulation_final_test_435.csv"):

        data_objects = []
        for pth in np.core.defchararray.add([base_path] * 3, [train_path, test_path, final_test_path]):
            #("importing {}".format(pth))

            df = pd.DataFrame.from_csv(pth, index_col=None)
            df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
            # print(df[["OBJECTID", "TS"]])
            #("loaded {} samples and {} features.".format(df.shape[0], df.shape[1]))

            do = DataStructure()
            do.X, do.t, do.rul, do.objectid, do.cluster, do.ts = DataSource._load_dataset_nasa_helper(df)
            data_objects.append(do)

        return data_objects

    @staticmethod
    def load_phm(base_path="../../Data/phm-2008-data-challange/ready/",
                 train_path="out_mod_train.txt",
                 test_path="out_mod_test.txt",
                 final_test_path="out_mod_final_test.txt"):

        data_objects = []
        for pth in np.core.defchararray.add([base_path] * 3, [train_path, test_path, final_test_path]):
            #("importing {}".format(pth))

            df = pd.DataFrame.from_csv(pth, index_col=None)
            df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
            #("loaded {} samples and {} features.".format(df.shape[0], df.shape[1]))

            do = DataStructure()
            do.X, do.t, do.rul, do.objectid, do.cluster, do.ts = DataSource._load_dataset_nasa_helper(df)
            data_objects.append(do)

        return data_objects

    @staticmethod
    def load_bmw_rex(base_path="../../Data/bmw-rex/ready/",
                     train_path="bmw_rex_train.csv",
                     test_path="bmw_rex_test.csv"):
        data_objects = []
        for pth in np.core.defchararray.add([base_path] * 2, [train_path, test_path]):
            #("importing {}".format(pth))

            df = pd.DataFrame.from_csv(pth, index_col=None)
            df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
            #("loaded {} samples and {} features.".format(df.shape[0], df.shape[1]))

            do = DataStructure()
            do.X, do.t, do.rul, do.objectid, do.cluster, do.ts = DataSource._load_dataset_bmw_helper(df)
            data_objects.append(do)

        return data_objects


    @staticmethod
    def load_bmw_cells(base_path="../../Data/bmw-cells/ready/",
                       train_path="bmw_cells_train.csv",
                       test_path="bmw_cells_test.csv"):
        data_objects = []
        for pth in np.core.defchararray.add([base_path] * 2, [train_path, test_path]):
            #("importing {}".format(pth))

            df = pd.DataFrame.from_csv(pth, index_col=None)
            df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
            #("loaded {} samples and {} features.".format(df.shape[0], df.shape[1]))

            do = DataStructure()
            do.X, do.t, do.rul, do.objectid, do.cluster, do.ts = DataSource._load_dataset_bmw_helper(df)
            data_objects.append(do)

        return data_objects

    @staticmethod
    def load_weather(base_path="../../Data/weather-forecast/ready/",
                     train_path="weather_train.csv",
                     test_path="weather_test.csv"):
        data_objects = []
        for pth in np.core.defchararray.add([base_path] * 2, [train_path, test_path]):
            #("importing {}".format(pth))

            df = pd.DataFrame.from_csv(pth, index_col=None)
            df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
            #("loaded {} samples and {} features.".format(df.shape[0], df.shape[1]))

            do = DataStructure()
            do.X, do.t, do.rul, do.objectid, do.cluster, do.ts = DataSource._load_dataset_bmw_helper(df)
            data_objects.append(do)

        return data_objects
