#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

from dbse.api import Parameter as P, Tester
import time
from dbse.data import DataSource, DataStructure
from dbse.pipeline.nodes.HeatmapConvolutionTrainer import HeatmapConvolutionTrainer
import pandas as pd

def set_features_to_skip():
    '''
    Note: any feature that is to be skipped requires the prefix scaled_
    A dictionary is passed where the key indicates the cluster id and the value is a list of features to
    skip during test
    '''

    skip_features = {}
    skip_features[0] = dict() # main model features
    skip_features[1] = dict() # fine tuner model left features
    skip_features[2] = dict() # fine tuner model mid features
    skip_features[3] = dict() # fine tuner model right features

    # one per cluster e.g.
    feats = dict()
    feats[0] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_4','scaled_FEATURE_7','scaled_FEATURE_8','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_14','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21','scaled_FEATURE_12']
    feats[1] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_4','scaled_FEATURE_7','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_14','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21','scaled_FEATURE_12']
    feats[2] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_4','scaled_FEATURE_7','scaled_FEATURE_8','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_14','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21','scaled_FEATURE_12']
    feats[3] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_7','scaled_FEATURE_8','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_14','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21','scaled_FEATURE_12']
    feats[4] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_4','scaled_FEATURE_7','scaled_FEATURE_8','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_14','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21']
    feats[5] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_7','scaled_FEATURE_8','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21','scaled_FEATURE_12']

    # here use same features for all models
    skip_features[0] = feats
    skip_features[1] = feats
    skip_features[2] = feats
    skip_features[3] = feats

    return skip_features

if __name__ == "__main__":

    # Parameters
    P.te_feature_selection_on = True # If true features are selected according to the features given
    P.tr_remove_outliers = True # If true outliers are removed per feature heat map
    P.te_smoothing_side = 51 # smoothing parameter
    P.tr_kernel_size = 11 # Kernle size
    P.te_percentage_side_fine = 0.1 # Percentage
    visualize_heatmap = False # If true after each trained cluster a heatmap is shown
    read_from_file = True
    file = "trained_model.dbs"

    # -----------------------------------------------------------------------------------------------------
    #   Load Dataset
    # -----------------------------------------------------------------------------------------------------

    # load training data into structure
    df = pd.DataFrame.from_csv(r"data/train_set_test.csv", index_col=None)
    df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
    ti = DataStructure()
    ti.X, ti.t, ti.rul, ti.objectid, ti.cluster, ti.ts = DataSource.load_clean(df)

    # load test data
    df = pd.DataFrame.from_csv(r"data/train_set_test.csv", index_col=None)
    df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
    ts = DataStructure()
    ts.X, ts.t, ts.rul, ts.objectid, ts.cluster, ts.ts = DataSource.load_clean(df)

    # convert to ingest set
    data = DataSource.to_ingest_set(ti, ts)

    # -----------------------------------------------------------------------------------------------------
    #   Run Training and Test
    #   Per cluster given in ti.cluster one model is trained - if no clusters are available this parameter
    #   can be a constant variable
    # -----------------------------------------------------------------------------------------------------

    # create and run trainer
    trainer = HeatmapConvolutionTrainer(test_mode = False, whole_data_set= True, remove_outliers=P.tr_remove_outliers, visualize_heatmap = visualize_heatmap, grid_area=P.tr_grid_area, interpol_method=P.tr_interpol_method, kernel_size = P.tr_kernel_size, std_gaus = P.tr_std_gaus)
    t = time.time()
    print("\nStart Training...\nKernel: " + str(P.tr_kernel_size)) # train model
    trained_model, _ = trainer.run(data)
    print("Training Time: " + str(time.time() - t))



    # 4. run test
    tester = Tester()
    tester.skip_features = set_features_to_skip()# optionally can pass features to skip e.g. found via feature selection
    t = time.time()
    list_of_results = tester.do_testing(data, smooth_per_feat = P.te_smooth_per_feat, csv_pathh = P.te_csv_path, test_mode=False)
    testing_time = time.time() - t
    print("Total Testing Time: " + str(testing_time) + "\n------------------------------")

    # 5. print results
    y_pred = pd.DataFrame(list_of_results, columns=["object_id", "cluster_id", "rul", "risk", "predicted_rul", "predicted_risk", "invalid", "invalid2", "testing_time"])
    #y_pred.to_csv("predictions.csv") # STORE RESULTS TO PANDAS DF IF REQUIRED

    for result in list_of_results:
        print("\n\nObject Id: %s" % str(result[0]))
        print("Cluster Id: %s" % str(result[1]))
        print("Real RUL: %s" % str(result[2]))
        print("Real Risk: %s" % str(result[3]))
        print("Predicted RUL: %s" % str(result[4]))
        print("Predicted Risk: %s" % str(result[5]))







