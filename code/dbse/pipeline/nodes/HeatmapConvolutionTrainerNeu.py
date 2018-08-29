#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
import pandas as pd
from pipeline.BaseNode import BaseNode
from sklearn import preprocessing
import numpy as np
import copy
from tools.Visual import Visual
from tools.Logging import Logging
from scipy import signal
from scipy import misc
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from scipy.interpolate import griddata
import math
import matplotlib.pyplot as plt
import traceback
import warnings; warnings.simplefilter('ignore')
from pipeline.nodes.HeatmapConvolutionTester import HeatmapConvolutionTester
from pipeline.helper.iron_tool_belt import *
import numpy
import random
import os
import csv
import All
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.signal.signaltools import medfilt

class HeatmapConvolutionTrainerNeu(BaseNode):
    '''
    Attention: THIS has to be done per cluster id! As per cluster id one model results
    
    This class contains the training of the prediction model. The model works as follows:
        1. scale data 
        2. remove outliers
        3. remove empty or constant features
        4. split the 2D area of risk vs. feature value into a grid of size "grid_area x grid_area" 
           by interpolating all points on this grid
        5. per feature perform a 2D convolution with a gaussian curve on this 2D area to generate a distribution of heat over the whole area
        
        -> The resulting model is stored for prediction
        
    Parameters are
    1: ???
    2: ??? second parameter needs to be of type int (rul percentile). This specified which percentiles are filtered out.
       Recommmended value is 99
    '''
    

    def __init__(self, visualize_outlier: bool,
                 visualize_heatmap: bool,
                  grid_area: int, interpol_method: str,
                  kernel_size: int, std_gaus: float,
                  remove_empty_features = True,
                  nr_outlier_iterations = 1,
                  outlier_window_size = 10,
                  whole_data_set = False,
                  outlier_std_threshold = 1,
                  remove_outliers = True,
                  test_mode = False,
                 field_in_train_t="train_t",
                 field_in_train_X="train_X",
                 field_in_train_crit="train_crit",
                 field_in_train_risc="train_risc",
                 field_in_train_id="train_id",
                 field_in_train_rul="train_rul",
                 field_in_meta_dataset_name = 'meta_dataset_name',
                 field_in_train_cluster_id="train_cluster_id",
                 field_out_train_model = "train_model",
                 field_out_train_model_grid_area = "train_model_grid_area", 
                 field_out_train_model_trained_scalers = "train_model_trained_scalers"):
        
        """
        Constructor
        :param whole_data_set: if true the evaluation is performed not only on critical parts but on the whole dataset
        :param visualize_outlier: boolean that describes if the result of the outlier removal should be visualized
        :param visualize_heatmap: boolean that determines if the produced heatmap per feature should be plotted
        :param grid_area:  The heat map is split into a grid of grrd_area x grid_area big field 
        :param interpol_method: Method used for interpolation in grid area linear, nearest or cubic
        :param kernel_size: Gauss kernel used for convolution - has to be odd!
        :param std_gaus: standard deviation of gauss kernel used for filtering/convolution
        :param remove_empty_features: Boolean - true if empty features should be removed during outlier detection
        :param nr_outlier_iterations: Number of iterations to be performed per outlier removal
        :param outlier_window_size: Size of window (in # elements) that is used to remove 
        :param outlier_std_threshold: number of std deviations over which a data point is considered outlier
        :param train_range_x: x axis of rul estimated data during training (from risk-min to risk-max)
        :param test_mode: if true only a subset of data is evaluated
        :param remove_outliers: if True outliers are removed prior to training
        :param field_in_train_t: Timestamp of the training data
        :param field_in_train_X: Features of the training data
        :param field_in_train_crit: 1 if row is in critical area and 0 otherwise
        :param field_in_train_risc: risk score at each point of time
        :param field_in_train_id: object identifier of training object
        :param field_in_train_rul:RUL of training data
        :param field_in_train_cluster_id: cluster identifier assigned to each object (numeric from 0 to numberOfClusters-1)        
        """
        # parameters
        self._visualize_outlier = visualize_outlier
        self._visualize_heatmap = visualize_heatmap
        self._grid_area = grid_area
        self._interpol_method = interpol_method
        self._kernel_size = kernel_size
        self._std_gaus = std_gaus
        self._remove_empty_features = remove_empty_features
        self._nr_outlier_iterations = nr_outlier_iterations
        self._outlier_window_size = outlier_window_size
        self._outlier_std_threshold = outlier_std_threshold
        self._test_mode = test_mode
        self._remove_outliers = remove_outliers
        self._whole_data_set = whole_data_set
                
        # fields
        self._field_in_train_t = field_in_train_t
        self._field_in_train_t = field_in_train_t
        self._field_in_train_X = field_in_train_X
        self._field_in_train_crit = field_in_train_crit
        self._field_in_train_risc = field_in_train_risc
        self._field_in_train_id = field_in_train_id
        self._field_in_train_rul = field_in_train_rul
        self._field_in_meta_dataset_name = field_in_meta_dataset_name
        self._field_in_train_cluster_id = field_in_train_cluster_id
        self._field_out_train_model = field_out_train_model
        self._field_out_train_model_grid_area = field_out_train_model_grid_area
        self._field_out_train_model_trained_scalers = field_out_train_model_trained_scalers
        
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
    
    def _split_to_subsets(self, whole_df, percentage_train, iter_idx):
        ''' This method splits the whole dataframe into a training and a testset multiple times        
        '''
        
        object_indices = list(whole_df["id"].unique())
        idx_thrshld = int(percentage_train * (len(object_indices)-1))
        
        res_list = []
        for i in range(iter_idx):
            
            # shuffle the list 
            random.shuffle(object_indices)
            shuffled_indices = object_indices
            
            # take first percentage percent as training 
            train_indices = shuffled_indices[:idx_thrshld]
            train_df = whole_df[whole_df['id'].isin(train_indices)]
            
            # take second percentage-1 as test
            test_indices = shuffled_indices[idx_thrshld:]
            test_df = whole_df[whole_df['id'].isin(test_indices)]
            
            res_list.append([train_df, test_df])
            
        return res_list
        
    def run(self, data_in):
        super().run(data_in)  # do not remove this!
        Logging.log("Training da heat...")
        
        # cross validation params
        iter_idx = 2 # Monte Carlo cross-validation - randomly assign training and test set x times
        percentage_train = 0.8 # percentage of data being trainingset
                
        # 1. transform to df and keep critical
        whole_df = self._extract_critical_data_frame(data_in) 
        whole_df[self._field_in_train_cluster_id] = data_in[self._field_in_train_cluster_id]
        lst_of_train_n_test = self._split_to_subsets(whole_df, percentage_train, iter_idx) # split frame multiple times
        # each element being [train_df, test_df]
        
        # 2. distance - scoring quality of a feature
        dist_score = {} # key: clusterid_featureid e.g. c1 value: list of dict: key: feature_id, value score

        for train_test in lst_of_train_n_test:
            train_df = train_test[0]
            test_df = train_test[1]
            test_df["cluster_id"] = test_df["train_cluster_id"]
            
            for cluster_id in list(train_df["train_cluster_id"].unique()): # per cluster own model
                #if cluster_id in [1, 5, 4, 0, 3]: # den hab ich schon
                #    continue               
                
                if self._test_mode and not (cluster_id == 3  or cluster_id == 1):
                    continue
                
                print("\n\n TRAINING CLUSTER: "+str(cluster_id))
                cur_train_df = train_df[train_df[self._field_in_train_cluster_id]==cluster_id]
                cur_test_df = test_df[test_df["cluster_id"]==cluster_id]
            
                # 2. scale data and remove outliers
                output_dfs, trained_scalers = self._preprocess_data(cur_train_df, self._remove_empty_features, self._nr_outlier_iterations, self._outlier_window_size, self._outlier_std_threshold)
                data_in["CL_" + str(cluster_id) + "_" + self._field_out_train_model_trained_scalers] = trained_scalers
    
                # 3. Train the model
                model_per_feature = self._build_heat_map_parallel(output_dfs)
                data_in["CL_" + str(cluster_id) + "_" + self._field_out_train_model_grid_area] = self._grid_area

                # 4. Store the models
                data_in["CL_" + str(cluster_id) + "_" + self._field_out_train_model] = model_per_feature
                                
                # 5. score the feature quality for cross validation
                score_dict = self._score_feature_quality(cur_test_df, whole_df, model_per_feature, cluster_id, data_in)
                print("Found scores: " + str(score_dict))
                idfr = "c"+str(cluster_id)
                if idfr not in dist_score:
                    dist_score[idfr] = [score_dict]
                else: 
                    dist_score[idfr].append(score_dict)
                    
                try:
                    pathh = os.path.join(r"C:\Users\q416435\Desktop\scores", "cluster_" + str(cluster_id)+".csv")
                    print("Writing file to "+ pathh)
                    self._csv_file = open(pathh, 'w')
                    for ke in score_dict.keys():                    
                        self._csv_writer = csv.writer(self._csv_file, delimiter=';')
                        self._csv_writer.writerow([ke, str(score_dict[ke])])
                    self._csv_file.close()
                except: 
                    pass
                    
        # 3. Perform training for whole model now
        # get final model
                
        # 4. keep only optimal models now based on dist_score
        #T.B.D.        
        # 5. empty metrics
        metrics = dict()

        return data_in, metrics
    
    
    def _score_feature_quality(self, test_df, whole_df, model_per_feature, cluster_id, data_in, r_min, r_max, finetuner_index ):
        
        # wenn einmal berechnet wegspeichern und im Zweifel wieder laden
        print("_________SCORING: "+ str(r_min) + " to " + str(r_max))
        
        # jedes model is ein heat map
        tester = HeatmapConvolutionTester(smooth_per_feature = True, enable_all_print = False, visualize_summed_curve = False, visualize_per_feature_curve = False)
        abs_max_rul = whole_df["RUL"].max() # 217
        segment_thrshld = 0.33 *abs_max_rul
        distances = {} # key feat name, value list_dist
        phm_scores = {}
        rmse_scores = {}
        tot = len(list(test_df["id"].unique()))
        oo = 0
        for object_id in list(test_df["id"].unique()):  
            oo += 1          
            Logging.log(str(oo) + " of " + str(tot) +" - Optimizing based on - OBJECT ID: "+str(object_id))
            cur_df1 = test_df[test_df['id'] == object_id]
            
            # predict immer einmal random aus erste 33% dann zwischen 33% und 66% und dann zwischen 66 und 100%
            le = int(numpy.ceil(len(cur_df1)/3))
            z_to_33 = list(range(le))
            random.shuffle(z_to_33)
            if 2*le > len(cur_df1): 
                t_to_66 = []
                s_to_100 = []
                thrshlds = [z_to_33[0]]
            else: 
                t_to_66 = list(range(le, 2*le))
                s_to_100 = list(range(2*le, len(cur_df1)))
                random.shuffle(t_to_66)
                random.shuffle(s_to_100)                
                thrshlds = [z_to_33[0], t_to_66[0], s_to_100[0]]
            
            cur_df3 = cur_df1.sort_values("RUL", ascending=False)
            for thrshld in thrshlds:
                current_test_df = cur_df3.iloc[:thrshld]
                                
                dist = current_test_df["RUL"].max()- current_test_df["RUL"].min()
                if dist > segment_thrshld:
                    print("SHORTENED RUL AREA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
                    thrs = current_test_df["RUL"].min() + segment_thrshld
                    current_test_df = current_test_df[current_test_df["RUL"] < thrs]
                
                # do prediction
                try:
                    predicted_risk, predicted_rul, m, all_feature_sum, per_feature_sum, feature_favorites = tester._predict_RUL(data_in, current_test_df, cluster_id, None, [], 0 , current_test_df["RUL"].min(), test=True, fine = finetuner_index)
                except: 
                    print("No prediction to short shit")
                    continue
                true_RUL = current_test_df.iloc[-1]["RUL"]
                true_risk = 1 + m * true_RUL
                
                # post process
                
                
                # asses per_feature_sum
                for col_name in per_feature_sum.keys():
                    #print("--- Feature: " + col_name)
                    cur = per_feature_sum[col_name]
                    cur[1] = cur[1][0]
                    
                    #m_idx = numpy.argmax(cur[1])
                    #da_risk_found = cur[0][m_idx]
                    #predicted_rul_feature = (da_risk_found - 1)/m
                    
                    if numpy.count_nonzero(cur[1]) == 0:
                        if col_name in distances: 
                            distances[col_name].append([1]) # this curve did not help at all
                            phm_scores[col_name].append(["None"])
                            rmse_scores[col_name].append(["None"])
                        else: 
                            distances[col_name] = [[1]]
                            phm_scores[col_name] = [["None"]]
                            rmse_scores[col_name] = [["None"]]
                        #print(" - Distance - 1")
                        continue
                    
                    ten_perc = math.ceil(0.05 * len(cur[1]))
                    subs = int(0.1 * len(cur[1]))
                    ind = sorted(numpy.argpartition(cur[1], -ten_perc)[-ten_perc:]) # indices of x percent of highest values
                    
                    
                    # if gap bigger than subs indices == 0.1 risk -> split to to regions
                    gaps = numpy.where(numpy.diff(ind)>subs)
                    if not gaps: runner = [None]
                    if gaps: runner = sorted(list(gaps[0]))
                    prev = 0
                    multi_dist = []
                    phm_scores_lst = []
                    rmse_scores_lst = []
                    for gap_idx in runner: 
                        if gap_idx == None: 
                            cur_subset_selection = ind
                        else:
                            gap_idx += 1
                            cur_subset_selection = ind[int(prev):int(gap_idx)]
                        
                        values  = cur[0][cur_subset_selection]
                        
                        avg = numpy.average(values)
                        dist = avg - true_risk
                        
                        
                        # bevorzuge praktisch frühere weil das den index nicht so zerlegt 
                        multi_dist.append(dist)
                        #print(" - Distance - " + str(dist))
                        
                        # analog find phm and risk
                        da_risk_found = avg
                        predicted_rul_feature = (da_risk_found - 1)/m
                        
                        phmScore = self.score_phm(pd.DataFrame([[true_RUL, predicted_rul_feature, -1]], columns = ["RUL", "predicted_RUL", "object_id"]))
                        rmse = self.score_rmse(numpy.array([true_RUL]), numpy.array([predicted_rul_feature]))
                        phm_scores_lst.append(phmScore)
                        rmse_scores_lst.append(rmse)
                        
                        
                        prev = gap_idx 
                    
                        '''
                        print("\nFeature: "+ str(col_name)+ "\nplot all - true risk: " + str(true_risk))
                        plt.plot(cur[0], cur[1])
                        plt.plot(cur[0][cur_subset_selection], cur[1][cur_subset_selection], color='red')
                        #plt.plot(cur[0], medfilt(cur[1], 61), color = "green") # KERNEL MUST BE ODD 
                        plt.xlabel("risk - true risk = " + str(true_risk))
                        plt.ylabel("heat - "+str(col_name))
                        plt.show()
                        '''
                        
                    # use this for evaluation
                    #phmScore = self.score_phm(pd.DataFrame([[true_RUL, predicted_rul_feature, -1]], columns = ["RUL", "predicted_RUL", "object_id"]))
                    #rmse = self.score_rmse(numpy.array([true_RUL]), numpy.array([predicted_rul_feature]))
                    
                    if col_name in distances: 
                        distances[col_name].append(multi_dist) # this curve did not help at all
                        phm_scores[col_name].append(phm_scores_lst)
                        rmse_scores[col_name].append(rmse_scores_lst)
                    else: 
                        distances[col_name] = [multi_dist]
                        phm_scores[col_name] = [phm_scores_lst]
                        rmse_scores[col_name] = [rmse_scores_lst]
                    
        return distances, phm_scores, rmse_scores


    def score_phm(self, df, bound_phm = False, plot_heatmap_enabled=True):
    
        # groupby preserves the order of rows within each group, 
        # see http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
        #df = df.sort_values(["object_id", "RUL"], ascending=[True, False]).groupby('object_id').last().reset_index().sort_values(by="object_id")
        # THIS MUST BE ONE LINE - BUG?
        
        scores = []
        cnt = 0
        
            
        for pred, actual, objectid in zip(df["predicted_RUL"], df["RUL"], df["object_id"]):
            
            print("comparing pred={}/rul={} for object {}".format(pred, actual, objectid))

            a1 = 13
            a2 = 10
            d = pred - actual
    
            if d > 50 and bound_phm:
                d = 50
            if d < -50 and bound_phm:
                d = -50
    
            if d < 0:
                scores.append(np.exp(-d / a1) - 1)
            elif d >= 0:
                scores.append(np.exp(d / a2) - 1)
                
            cnt += 1

         
        #print("scored {} elements".format(cnt))
        #print("\nPhm Scores: "+str(scores))
        #print("\nDavon ueber 100: "+str(len([s for s in scores if s>100])))
        return sum(scores)

    def score_rmse(self, y_pred, y_true):
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        return rmse
    
    def run_old(self, data_in):
        super().run(data_in)  # do not remove this!
        Logging.log("Training da heat...")
        
        # cross validation params
        iter_idx = 2 # Monte Carlo cross-validation - randomly assign training and test set x times
        percentage_train = 0.8 # percentage of data being trainingset
                
        # 1. transform to df and keep critical
        train_df = self._extract_critical_data_frame(data_in)        
        train_df[self._field_in_train_cluster_id] = data_in[self._field_in_train_cluster_id]
        
        for cluster_id in list(train_df["train_cluster_id"].unique()): # per cluster own model
            if self._test_mode and not (cluster_id == 3  or cluster_id == 1):
                continue      
            
            print("\n\n TRAINING CLUSTER: "+str(cluster_id))
            cur_train_df = train_df[train_df[self._field_in_train_cluster_id]==cluster_id]
        
            # 2. scale data and remove outliers
            output_dfs, trained_scalers = self._preprocess_data(cur_train_df, self._remove_empty_features, self._nr_outlier_iterations, self._outlier_window_size, self._outlier_std_threshold)
            data_in["CL_" + str(cluster_id) + "_" + self._field_out_train_model_trained_scalers] = trained_scalers

            # 3. Train the model
            model_per_feature = self._build_heat_map_parallel(output_dfs)
            data_in["CL_" + str(cluster_id) + "_" + self._field_out_train_model_grid_area] = self._grid_area
                                    
            # 4. Store the models
            data_in["CL_" + str(cluster_id) + "_" + self._field_out_train_model] = model_per_feature

        # TODO: Training oben muss mit 80% der Dategeschehen und Testlauf dann mit 20 %
        # 5. Durchlaufe Testlauf hier 

        # 5. empty metrics
        metrics = dict()

        return data_in, metrics
        
        
    def _build_one_heat_map(self, feature_df, risk_min, feature_min, feature_max):
        Logging().log("Processing Feature: "+feature_df.columns[1])
                
        try:
            values = np.empty(len(feature_df))
            values.fill(1)
    
            # Assign X Y Z
            X = feature_df.RISK.as_matrix()
            Y = feature_df[feature_df.columns[1]].as_matrix()
            Z = values
    
            # create x-y points to be used in heatmap of identical size
            #risk_min = min([rm for rm in [df.RISK.min() for df in output_dfs] if not math.isnan(rm)])
            risk_max = 1
            
            xi = np.linspace(risk_min, risk_max, self._grid_area)
            yi = np.linspace(feature_min, feature_max, self._grid_area)
    
            # Z is a matrix of x-y values interpolated (!)
            zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method=self._interpol_method)
            zmin = 0
            zmax = 1
            zi[(zi<zmin) | (zi>zmax)] = None
    
            # Convolve each  point with a gaussian kernel giving the heat value at point xi,yi being Z
            # Advantage: kee horizontal and vertical influence 
            grid_cur = np.nan_to_num(zi)

            # Smooth with a Gaussian kernel 
            kernel = Gaussian2DKernel(stddev=self._std_gaus, x_size=self._kernel_size, y_size=self._kernel_size)
            grad = scipy_convolve(grid_cur, kernel, mode='same', method='direct')
    
            # Store the model in memory
            feature_name = feature_df.columns[1]
            result = [feature_name, [copy.deepcopy(np.absolute(grad)), copy.deepcopy(xi), copy.deepcopy(yi)], grid_cur]
            
        except:
            #traceback.print_exc()
            feature_name = feature_df.columns[1]
            Logging().log(str(feature_df.columns[1])+": No heat map created due to error")
            result = [feature_name, None, None]
        
        return result
                       
    def _build_heat_map_parallel(self, output_dfs):        
        
        # Parameter: nehme nur die 21 features fürs erste 
        output_dfs = output_dfs[:21]
        parallel_processes = numpy.ceil(len(output_dfs))
        
        # TEST - spielt eigentlich gar keine Rolle! 
        #if self._test_mode:
        #    output_dfs = output_dfs[:2]
        #    parallel_processes = numpy.ceil(len(output_dfs))
        
        # output ist dimensions
        dimensions = {}
        print("Building " + str(len(output_dfs)) + " heat maps (if all work out)")
        
        # 1. create input arguments = output_dfs
        risk_min = min([rm for rm in [df.RISK.min() for df in output_dfs] if not math.isnan(rm)])
        feature_min = min([rm for rm in [df[df.columns[1]].min() for df in output_dfs] if not math.isnan(rm)])
        feature_max = max([rm for rm in [df[df.columns[1]].max() for df in output_dfs] if not math.isnan(rm)])
        res_list = parallelize_stuff([[l, risk_min, feature_min, feature_max] for l in output_dfs], self._build_one_heat_map, simultaneous_processes = parallel_processes)
        
        # 2. output dimensions[feature_df.columns[1]] = output
        for res in res_list:
            name = res[0]
            dim = res[1]
            grid_cur = res[2]
            
            dimensions[name] = dim
            
            if self._visualize_heatmap:     
                print("Feature " + str(name))
                try:      
                    if grid_cur != None:                                 
                        fig, (ax_orig, ax_mag) = plt.subplots(1, 2)
                        ax_orig.imshow(grid_cur[::-1,::-1], cmap='RdYlGn')
                        ax_orig.set_title('Original '+ str(name))
                        ax_mag.imshow(dim[0][::-1,::-1], cmap='RdYlGn')# https://matplotlib.org/examples/color/colormaps_reference.html
                        ax_mag.set_title('Heat '+ str(name))
                        fig.show()
                        plt.show()
                except:
                    print("not plottable")
        
        return dimensions
        
        
    def _build_heat_map(self, output_dfs):
        ''' using convolution for each point of a 2d array risk vs. feature value per feature
            a heat map is generated 
            :param output_dfs: list of dataframes with each having a column scaled_FEATURE_X (that is outlierfree and scaled now) and a column 
                               risk which is the risk for that feature at its row
            :return a dictionary is returned that contains the feature name as key and its 2d heatmap as output
        '''
        dimensions = {}
        for feature_df in output_dfs: # each output_df has one risk and value     
            Logging().log("Processing Feature: "+feature_df.columns[1])
            
            # Testmode
            if self._test_mode and (feature_df.columns[1] == "scaled_FEATURE_5"): 
                print("Testing thus, break now!")
                break
            
            try:
                values = np.empty(len(feature_df))
                values.fill(1)
        
                # Assign X Y Z
                X = feature_df.RISK.as_matrix()
                Y = feature_df[feature_df.columns[1]].as_matrix()
                Z = values
        
                # create x-y points to be used in heatmap of identical size
                risk_min = min([rm for rm in [df.RISK.min() for df in output_dfs] if not math.isnan(rm)])
                risk_max = 1
                feature_min = min([rm for rm in [df[df.columns[1]].min() for df in output_dfs] if not math.isnan(rm)])
                feature_max = max([rm for rm in [df[df.columns[1]].max() for df in output_dfs] if not math.isnan(rm)])
                
                xi = np.linspace(risk_min, risk_max, self._grid_area)
                yi = np.linspace(feature_min, feature_max, self._grid_area)
        
                # Z is a matrix of x-y values interpolated (!)
                zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method=self._interpol_method)
                zmin = 0
                zmax = 1
                zi[(zi<zmin) | (zi>zmax)] = None
        
                # Convolve each  point with a gaussian kernel giving the heat value at point xi,yi being Z
                # Advantage: kee horizontal and vertical influence 
                grid_cur = np.nan_to_num(zi)

                # Smooth with a Gaussian kernel 
                kernel = Gaussian2DKernel(stddev=self._std_gaus, x_size=self._kernel_size, y_size=self._kernel_size)
                grad = scipy_convolve(grid_cur, kernel, mode='same', method='direct')
        
                # Store the model in memory
                dimensions[feature_df.columns[1]] = [copy.deepcopy(np.absolute(grad)), copy.deepcopy(xi), copy.deepcopy(yi)]
                #print("GRAD")
                #print(str(grad))
                
                
                if self._visualize_heatmap:                
                    fig, (ax_orig, ax_mag) = plt.subplots(1, 2)
                    ax_orig.imshow(grid_cur[::-1,::-1], cmap='RdYlGn')
                    ax_orig.set_title('Original')
                    ax_mag.imshow(np.absolute(grad)[::-1,::-1], cmap='RdYlGn')# https://matplotlib.org/examples/color/colormaps_reference.html
                    ax_mag.set_title('Heat')
                    fig.show()
                    plt.show()

            except:
                Logging().log("No chance")
                #traceback.print_exc()
                dimensions[feature_df.columns[1]] = None
                
        return dimensions, xi

    def _extract_critical_data_frame(self, data):
        ''' Creates a central dataframe from the given training data fusing 
            existing subelements of the data dictionary and keeps critical 
            examples only
        :param data dictionary as passed by the main execution chain
        :return dataframe concatenated with time criticality ris id and rul 
        '''
        
        train_df = data[self._field_in_train_t].to_frame().join(data[self._field_in_train_X])
        train_df["CRIT"] = 1# data[self._field_in_train_crit].to_frame()
        train_df["RISK"] = data[self._field_in_train_risc].to_frame()
        train_df["id"] = data[self._field_in_train_id].to_frame()
        train_df["RUL"] = data[self._field_in_train_rul].to_frame()
        if not self._whole_data_set:
            train_df = train_df[train_df["CRIT"]>0]
        train_df["TS"] = data["train_ts"]
        return train_df
              
    def _preprocess_data(self, train_df, remove_empty, nr_iterations, split_windows = 10, std_threshold = 1):
        ''' data is scaled and outliers are removed to reduce noise 
        :param: train_df: Dataframe that contains the training data
        :param: remove_empty: Boolean - if true empty features are removed
        :param: nr_iterations: Number of iterations that are repeated to remove outliers per window
        :param: split_windows: Data is split into split_windows equal length window that are between minimal risk and 1
        :param: std_threshold: data that is further away than std_threshold * std of the feature is removed
        :return: output_dfs: list of dataframes with each having a column scaled_FEATURE_X (that is outlierfree and scaled now) and a column 
                             risk which is the risk for that feature at its row
        '''
        
        # 1. Scale
        train_df, trained_scalers = self._scale(train_df)
        
        
        # 2. Remove outliers
        output_dfs = self._outlier_removal(train_df, remove_empty, nr_iterations, split_windows, std_threshold)
        
        return output_dfs, trained_scalers
           
    def _outlier_removal(self, train_df, remove_empty, nr_iterations, split_windows, std_threshold):
        ''' outliers are removed from the training dataframe per feature by windowing and removing
            all values per window that are further away than std_threshold times the standard 
            deviation
        :param: train_df: Dataframe that contains the training data
        :param: remove_empty: Boolean - if true empty features are removed
        :param: nr_iterations: Number of iterations that are repeated to remove outliers per window
        :param: split_windows: Data is split into split_windows equal length window that are between minimal risk and 1
        :param: std_threshold: data that is further away than std_threshold * std of the feature is removed
        :return: output_dfs: list of dataframes with each having a column scaled_FEATURE_X that is outlierfree now and a column risk which is the risk 
                             for that feature at its row
        '''
        if not self._remove_outliers:
            print("Outlier removal disabled!")
        # 1. Initialize
        output_dfs = []
        iteration = range(nr_iterations)
        
        first = True
        
        
        # Per feature and window
        for col in train_df.columns:
                        
            # 2. only scaled features are considered
            if not col.startswith("scaled_FEATURE"): continue        
            #Logging().log("CURRENT -> "+ col)
            result_df = train_df.sort_values("RISK")
        
            # 3. iterate multiple times over window 
            #   on each iteration remove outliers
            for i in iteration:        
                sub_dfs = []
                indices = []
                rs = 0
                # 4. iterate over windows
                for r in np.linspace(result_df["RISK"].min(),1,10):

                    sub_df = result_df[(rs <= result_df["RISK"]) & (r > result_df["RISK"])]     
                    if self._remove_outliers:
                        sub_df = sub_df[((sub_df[col] - sub_df[col].mean()) / sub_df[col].std()).abs() < std_threshold]
                    sub_dfs.append(sub_df) 
                    rs = r
                result_df = pd.concat(sub_dfs)    
                
                
            # 5. Merge result to common dataframe
            output_dfs.append(result_df[["RISK", col]])
                        
            # 6. Remove empty
            if (remove_empty and len(result_df[col].unique())<2): 
                continue

            # 7. Plot results
            if self._visualize_outlier:
                Logging().log("Pre - Standard Deviation vorher: "+str(train_df[col].std()))
                Visual().plot_scatter(train_df["RISK"], train_df[col])#, "RISK", "feature")  
                Logging().log("Post - Standard Deviation nachher: "+str(result_df[col].std()))
                Visual().plot_scatter(result_df["RISK"], result_df[col])

        return output_dfs

    def _scale(self, train_df):
        ''' centers the data around 0 and scales it 
        :param: train_df: Dataframe that contains the training data
        :return: dataframe with additional column scaled_FEATURE_X containing scaled features
        :return: trained_scalers: dictionary - Per feature stores scaler object that is needed in the testing 
                 phase to perform identical scaling, with key as column name
        '''
        
        Logging().log("Scaling Features...")
        trained_scalers = {}
        for col in train_df.columns:

            # 1. consider only relevant columns
            if not col.startswith("FEATURE"):       
                continue 
        
            # 2. standard scaler 
            scaler = preprocessing.StandardScaler()
            scaler = scaler.fit(train_df[col])
            
            train_df['scaled_'+col] = scaler.transform(train_df[col])
            trained_scalers[col] = copy.deepcopy(scaler)
                
        
        return train_df, trained_scalers
        
        
        