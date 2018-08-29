import pandas as pd
import numpy as np
from .log import log, ping, pong

"""
import like
from common.data import DataSource, DataStructure
"""

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
    def _clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].reset_index().astype(np.float64)

    @staticmethod
    def _load_dataset_nasa_helper(X):

        X = pd.DataFrame(X)
        X = DataSource._clean_dataset(X)
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
    def _replace_strings_by_factors(df):

        df_sub_no_str = df.select_dtypes(include=[np.number])
        df_sub_str = df.select_dtypes(exclude=[np.number])

        for column in df_sub_str:
            df_sub_str[column] = pd.Categorical.from_array(df_sub_str[column]).codes

        result = pd.concat([df_sub_str, df_sub_no_str], axis=1)

        return result

    @staticmethod
    def _load_dataset_bmw_helper(X):

        X = pd.DataFrame(X)
        X = DataSource._replace_strings_by_factors(X)
        X = DataSource._clean_dataset(X)
        X = X.sort_values(["OBJECTID", "TS"], ascending=[True, True])

        t = X['T']
        rul = X['RUL']
        objectid = X['OBJECTID']
        cluster = pd.Series([0] * X.shape[0])
        ts = X['TS']

        del X['T']
        del X['RUL']
        del X['OBJECTID']
        del X['TS']

        return X, t, rul, objectid, cluster, ts

    @staticmethod
    def _load_dataset_weather_helper(X):

        X = pd.DataFrame(X)
        X = DataSource._clean_dataset(X)
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
            log("importing {}".format(pth))

            df = pd.DataFrame.from_csv(pth, index_col=None)
            df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
            # print(df[["OBJECTID", "TS"]])
            log("loaded {} samples and {} features.".format(df.shape[0], df.shape[1]))

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
            log("importing {}".format(pth))

            df = pd.DataFrame.from_csv(pth, index_col=None)
            df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
            log("loaded {} samples and {} features.".format(df.shape[0], df.shape[1]))

            do = DataStructure()
            do.X, do.t, do.rul, do.objectid, do.cluster, do.ts = DataSource._load_dataset_nasa_helper(df)
            data_objects.append(do)

        return data_objects

    @staticmethod
    def load_bmw_rex(base_path="../../Data/bmw-rex/ready/",
                     train_path="bmw_rex_train.csv",
                     test_path="bmw_rex_test_err.csv"):
        data_objects = []
        for pth in np.core.defchararray.add([base_path] * 2, [train_path, test_path]):
            log("importing {}".format(pth))

            df = pd.DataFrame.from_csv(pth, index_col=None)
            df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
            log("loaded {} samples and {} features.".format(df.shape[0], df.shape[1]))

            do = DataStructure()
            do.X, do.t, do.rul, do.objectid, do.cluster, do.ts = DataSource._load_dataset_bmw_helper(df)
            data_objects.append(do)

        return data_objects


    @staticmethod
    def load_bmw_cells(base_path="../../Data/bmw-cells/ready/",
                       train_path="bmw_cells_train.csv",
                       test_path="bmw_cells_test_err.csv"):
        data_objects = []
        for pth in np.core.defchararray.add([base_path] * 2, [train_path, test_path]):
            log("importing {}".format(pth))

            df = pd.DataFrame.from_csv(pth, index_col=None)
            df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
            log("loaded {} samples and {} features.".format(df.shape[0], df.shape[1]))

            do = DataStructure()
            do.X, do.t, do.rul, do.objectid, do.cluster, do.ts = DataSource._load_dataset_bmw_helper(df)
            data_objects.append(do)

        return data_objects

    @staticmethod
    def load_weather(base_path="../../Data/weather-forecast/ready/",
                     train_path="weather_train.csv",
                     test_path="weather_test_err.csv"):
        data_objects = []
        for pth in np.core.defchararray.add([base_path] * 2, [train_path, test_path]):
            log("importing {}".format(pth))

            df = pd.DataFrame.from_csv(pth, index_col=None)
            df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
            log("loaded {} samples and {} features.".format(df.shape[0], df.shape[1]))

            do = DataStructure()
            do.X, do.t, do.rul, do.objectid, do.cluster, do.ts = DataSource._load_dataset_weather_helper(df)
            data_objects.append(do)

        return data_objects