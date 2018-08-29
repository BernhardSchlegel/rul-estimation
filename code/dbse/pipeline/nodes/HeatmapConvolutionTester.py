#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
import pandas as pd
from dbse.pipeline.BaseNode import BaseNode
import matplotlib.pyplot as plt
import traceback
import warnings; warnings.simplefilter('ignore')
from dbse.pipeline.nodes.ClusterTraining import ClusterTraining
from dbse.pipeline.nodes.HeatmapConvolutionTrainer import HeatmapConvolutionTrainer
import warnings; warnings.simplefilter('ignore')
import traceback
from bisect import bisect_left
from dbse.tools.Visual import *
import math
from dbse.tools.Logging import Logging
from numpy import mean
import numpy
from dbse.pipeline.helper.iron_tool_belt import *
from scipy.signal.signaltools import medfilt
import dill
import copy


class HeatmapConvolutionTester(BaseNode):
    '''
    This class contains the testing phase of the prediction model. 
    For each Object the RUL is estimated based on the trained model and a new field predicted_rul is added
    The model works as follows:
        1. load test sequence
        2. assign a cluster to the current object (scale, pca then cluster predictor)
        3. remove empty or constant features
        4. then per object and per feature and per time element of a feature - take y value of this feature and cut the heat map 
           horizontally to get a heat curve, for delayed time elements do the same but then shift by the time difference to the current 
           element
           Then sum up all curves per feature and per time element of each feature
        5. Iterate over object ids and there over all RULs (i.e. first use 1 then 2 then 3 RULs etc.)        
           add the result
    '''
    PROCESS_INSTANCE= 0
    
    def __init__(self, 
                 visualize_summed_curve = True,
                 visualize_per_feature_curve = True,
                 visualize_pre_post_risk = False,
                 enable_all_print = True,
                 write_csv = False,
                 csv_path = r"tested_result.csv",
                 test_mode = False,
                 optimize_via_average = False,
                 whole_data_set = False,
                 field_in_test_t = "test_t", 
                field_in_test_crit = "test_crit", 
                field_in_test_X = "test_X",
                field_in_test_id = "test_id",
                field_in_test_rul = "test_rul",
                field_in_train_cluster_stored = "train_cluster_stored",
                field_in_meta_dataset_name = "meta_dataset_name",
                field_in_train_model_grid_area = "train_model_grid_area",
                field_in_train_model_trained_scalers = "train_model_trained_scalers",
                field_in_train_model = "train_model",
                field_in_train_risc = "train_risc",
                field_in_train_rul = "train_rul",
                field_out_predicted_rul = "predicted_rul",
                smooth_per_feature = False,
                smoothing_side = 81, 
                percentage_side_fine = 0.1, 
                feature_selection_on = True):        
        """
        Constructor
        :param whole_data_set: if true the evaluation is performed not only on critical parts but on the whole dataset
        :param optimize_via_average: this optimization finds the average of all feature favorites, then removes the ones most distant (i.e. outlier removal) and keeps the rest
        :param visualize_summed_curve: visualize sum of all heat curves
        :param visualize_per_feature_curve: visualize sum of heats per feature
        :param visualize_pre_post_risk: visualize heat prior to and after shift
        :param write_csv: if true then write directly to CSV per result retrieved
        :param field_in_test_t: time stamp of the test data
        :param field_in_test_crit: 1 if row is in critical area and 0 otherwise
        :param field_in_test_X: Features of the training data
        :param field_in_test_id: object identifier of test object
        :param field_in_test_rul: RUL of test data - used for evaluation purpose only!
        :param field_in_train_cluster_stored: cluster identifier assigned to each object (numeric from 0 to numberOfClusters-1)
        :param field_in_meta_dataset_name: name of the current data set (for feature extraction in clustering)
        :param field_in_train_model_grid_area: grid specifics for the heat map
        :param field_in_train_model_trained_scalers: scaling factors of the training data
        :param field_in_train_model: heat map of the trained model
        :param field_in_train_risc: risk values of the training data (used to get rul, risk ratio)
        :param field_in_train_rul: rul values of the training data (used to get rul, risk ratio)
        """
        # parameters
        self.percentage_side_fine = percentage_side_fine
        self._smooth_per_feature = smooth_per_feature
        self._visualize_summed_curve = visualize_summed_curve
        self._visualize_per_feature_curve = visualize_per_feature_curve
        self._visualize_pre_post_risk = visualize_pre_post_risk
        self._test_mode = test_mode
        self._optimize_via_average = optimize_via_average
        self._write_csv = write_csv
        self._csv_path = csv_path
        self._whole_data_set = whole_data_set
        self._enable_all_print = enable_all_print
        self.skip_never = False
        self.skip_never_whole = False
        self.deviation_dict = {}
        self.feature_selection_on = feature_selection_on
                
        # fields
        self.smoothing_side = smoothing_side
        self._field_in_test_t = field_in_test_t
        self._field_in_test_crit = field_in_test_crit
        self._field_in_test_X = field_in_test_X
        self._field_in_test_id = field_in_test_id
        self._field_in_test_rul = field_in_test_rul
        self._field_in_train_cluster_stored = field_in_train_cluster_stored
        self._field_in_meta_dataset_name = field_in_meta_dataset_name
        self._field_in_train_model_grid_area = field_in_train_model_grid_area
        self._field_in_train_model_trained_scalers = field_in_train_model_trained_scalers
        self._field_in_train_model = field_in_train_model
        self._field_in_train_risc = field_in_train_risc
        self._field_in_train_rul = field_in_train_rul
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
    
    def run(self, data_in):
        super().run(data_in)  # do not remove this!
        Logging.log("Testing da heat...")
                
        # 1. transform to df and keep critical
        test_df = self._extract_critical_data_frame(data_in)  

        # 2. assign cluster id, add column with id
        test_df = self._assign_cluster(data_in, test_df)
        test_df["predicted_rul"] = -1
        test_df["predicted_risk"] = -1
        
        abs_max_rul = test_df["RUL"].max() # 217
        segment_thrshld = 0.33 *abs_max_rul
        if self._enable_all_print: print("THE MAXIMUM RUL IN THE DATA SET IS " + str(abs_max_rul))

        # 3. extract current relevant data - do this for all and append
        for object_id in list(test_df["id"].unique()):
            all_feature_sum = False
            cur_df1 = test_df[test_df['id'] == object_id]
            print("Current: OBJECT ID: "+str(object_id))
            
            timestamp_gap = 0 # PER Cluster need to shift incoming data else I cannot sum it up
            last_ts = 0
            expected_rul = 99999999
            all_feature_favorites = []
            for cluster_id in list(cur_df1["cluster_id"].unique()):
                if self._test_mode and not (cluster_id == 3):
                    continue
                Logging.log("--------> Eval: CLUSTER ID: "+str(cluster_id))
                cur_df2 = cur_df1[cur_df1['cluster_id'] == cluster_id]
                cnt = 0
                cur_df3 = cur_df2.sort_values("RUL", ascending=False)

                # per object predict only the maximal
                first = True
                for i in range(len(cur_df3)):
                    
                    # 0. parallelize only estimate last one                                      
                    current_test_df = cur_df3
                    if not first:
                        continue
                    if first:
                        first = False
                    Logging.log("--------> Eval: RUL RANGE: "+str(current_test_df["RUL"].max())+ " to "+str(current_test_df["RUL"].min()))                      
                    
                    # 1. OPTIMIERUNG - nehme nicht alles sondern nur die maximal letzten 120 (ansonsten verzerrt weil ich ja nur bis 200 gelernt hab)    
                    dist = current_test_df["RUL"].max()- current_test_df["RUL"].min()
                    if dist > segment_thrshld:
                        if self._enable_all_print: print("SHORTENED RUL AREA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
                        thrshld = current_test_df["RUL"].min() + segment_thrshld
                        current_test_df = current_test_df[current_test_df["RUL"] < thrshld]
                    
                    
                    # 4. run tester for this data frame and add column predicted
                    try: skip = skip_features[int(cluster_id)]
                    except: skip = []
                    
                    # 5. shift the input curve to align with the one processed next 
                    if last_ts != 0:
                        cur_ts = current_test_df["TS"].max()
                        timestamp_gap = cur_ts - last_ts
                        
                        
                    # 6. store last Timestamp for shifting if it is more urgent
                    if current_test_df["RUL"].min() < expected_rul:
                        expected_rul = current_test_df["RUL"].min()
                        
                        
                    predicted_risk, predicted_rul, m, all_feature_sum, per_feature_sum, feature_favorites = self._predict_RUL(data_in, current_test_df, cluster_id, all_feature_sum, skip, timestamp_gap, expected_rul)
                    all_feature_favorites += feature_favorites
                       
                       
                    # VARIANTE 1 - weighted average mit 1/x
                    print("USING WEIGHTED AVERAGE")
                    total_amount = 0
                    total_count = 0
                    for feat in all_feature_favorites:
                        weight = 1 / feat
                        total_count += (weight* feat)
                        total_amount += weight
                        
                    wAvg = total_count/total_amount
                    predicted_risk = wAvg
                    predicted_rul = (predicted_risk - 1)/m
                    print("\n->>>>>> Estimated predicted RUL FINAL FINAL: " + str(predicted_rul) + "\nUPDATE RISK: "+ str(predicted_risk))

                    # 7. wenn mehr als 2 features kleiner 0.53 sind dann nehme average dieser
                    rego = [a for a in all_feature_favorites if a < 0.53]
                    if len(rego) > 2:
                        predicted_risk = numpy.average(rego)
                        predicted_rul = (predicted_risk - 1)/m
                        print("Estimated predicted RUL UPDATED: " + str(predicted_rul) + "\nUPDATE RISK: "+ str(predicted_risk))

                    # 5. result should be at location of test_df WHERE current_test_df["RUL"].min()
                    test_df = test_df.set_value(current_test_df.index[-1], "predicted_risk", predicted_risk)
                    test_df = test_df.set_value(current_test_df.index[-1], "predicted_rul", predicted_rul)
                    
                    # 6. store last Timestamp for shifting if it is more urgent
                    if current_test_df["TS"].max() > last_ts:
                        last_ts = current_test_df["TS"].max()
                    
                    # 3. store to file
                    if self._write_csv:
                        cnt += 1
                        
                        object_id = str(object_id)
                        cluster_id = str(cluster_id)

        # 5. metrics
        metrics ={}
        return data_in, metrics
        

        
    def _predict_RUL(self, data, current_test_df, cluster_id, all_feature_sum, features_to_skip, timestamp_gap, expected_rul, test=False, fine=-1, dev_dict_path = ""):
        '''
        based on the trained model extract the RUL for the given test dataframe
        :param data_in data dictionary as passed by the main execution chain
        :param current_test_df: Dataframe to use for prediction
        :return column: predicted_rul - remaining useful life as determined by the predictor
        '''
        
        # 1. extract information
        grid_area, trained_scalers, dimensions, train_df, m = self._extract_prediction_information(data, cluster_id, test)
        
        # 2. run prediction adding column predicted_rul
        predicted_risk, predicted_rul, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict = self._do_prediction(current_test_df, grid_area, trained_scalers, dimensions, train_df, m, all_feature_sum, features_to_skip, timestamp_gap, expected_rul, fine, dev_dict_path)
                
        return predicted_risk, predicted_rul, m, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict
        
    def _initialize_feature(self, grid_area, col):

        # initialize empty array 
        summ = np.empty(grid_area) 
        summ.fill(0)
        all_good = True
        
        return summ, all_good
        
        
    def find_integrals(self, xi, yi):
        all_integrals = []
        # integral 0 bis 0.5 
        fst = numpy.where(xi <= 0.5)
        x1 = xi[fst]
        y1 = yi[fst]
        all_integrals.append(numpy.trapz(y1, x1))
        
        # integral 0.25 bis 0.75
        scd = numpy.where(xi>0.25)
        xtmp = xi[scd] 
        ytmp = yi[scd]
        scd = numpy.where(xtmp <= 0.75)
        x2 = xtmp[scd]
        y2 = ytmp[scd]
        all_integrals.append(numpy.trapz(y2, x2))
        
        # integral 0.5 bis 1.0
        thrd = numpy.where(xi>0.5)
        xtmp = xi[thrd]
        ytmp = yi[thrd]
        thrd = numpy.where(xtmp <= 1.0)
        x3 = xtmp[thrd]
        y3 = ytmp[thrd]
        all_integrals.append(numpy.trapz(y3, x3))
        
        return all_integrals
        
    def _do_whole_model_prediction(self, cur_test_df, grid_area, dimensions, m, all_feature_sum, features_to_skip, timestamp_gap, dev_dict_path):
    
        # 0 loading deviations
        if not self.skip_never_whole:
            try:
                with open(dev_dict_path, 'rb') as in_strm:
                    dev_dict = dill.load(in_strm)
            except:
                dev_dict = {}
        
    
        # 1. Initialize
        feature_favorites, per_feature_sum, found_risk, weight_b, feature_favs_dict = [], {}, -1, True, {}
        
        for col in cur_test_df.columns:
            if HeatmapConvolutionTrainer.scaled_relevant_columns(col) or col in features_to_skip:  continue            
            summ, all_good = self._initialize_feature(grid_area, col)
            
            try:                
                
                l_iter_e = len(cur_test_df[col])
                l_iter_s = 0
                if l_iter_e > 6:
                    l_iter_s = len(cur_test_df[col]) - 6
                    l_iter_e = len(cur_test_df[col])
                    
                # 2. Per Sample of Training data determine heat curve and add it shifted to most recent
                this_feature_tops = []
                for cur_row in list(range(l_iter_s, l_iter_e)):
                    try:
                        cur_heat_values, xi = self._get_current_heat(cur_test_df, dimensions, cur_row, m, grid_area, col)
                    except:
                        pass#print("John McSkippo")
                        #print(traceback.format_exc())
                        continue
                    
                    if len(numpy.where(cur_heat_values != 0)[0]) == 0: continue
                    if self._smooth_per_feature: 
                        cur_heat_values = medfilt(cur_heat_values, 41)
                    summ = summ + cur_heat_values                               
                    if self._visualize_pre_post_risk: 
                        self._visualize_pre_post_risk_m(xi, cur_model, y_rel_row_idx, cur_heat_values) 

                     # 2.1 Gather information per point
                    ten_per_idx =  math.floor(len(cur_heat_values)*0.1)
                    ind = self.largest_indices(cur_heat_values, ten_per_idx)
                    weight = cur_heat_values[ind]
                    max_val = numpy.max(weight)
                    normalized_weight = weight/max_val              
                    weighted_res = numpy.sum(xi[ind] * normalized_weight)/numpy.sum(normalized_weight)
                    this_risk = weighted_res
                    this_feature_tops.append(this_risk)
                
                
                avg = numpy.average(this_feature_tops)
                remaining = this_feature_tops#[f for f in this_feature_tops if (avg+0.2)>f and f>(avg-0.2)]  
                
                if summ.argmax() != 0:
                    feature_favs_dict[col] = avg
                try:
                    avg += dev_dict[cur_test_df["cluster_id"].iloc[0]][col]
                except:
                    pass
                    
                    
                #3. Timestamp gap: die Cluster sagen nicht alle die selbe RUL voraus muss sie also aufeinander schieben
                summ, xi = self._shift_array(xi, summ, timestamp_gap, m, grid_area)
                
                # 4. store and plot result+
                per_feature_sum[col] = deepcopy([xi, cur_heat_values])
                all_feature_sum = all_feature_sum + summ
                if self._visualize_per_feature_curve: 
                    self._visualize_per_feature_curve_m(xi, summ)
                    
                try:
                    two_max = numpy.array(self.find_integrals(xi, summ)).argsort()[-2:][::-1]
                    two_max = two_max.tolist()
                    if 1 in two_max:
                        # vote
                        two_max.remove(1)
                        self.voting[two_max[0]] += 2
                        self.voting[1] += 1
                    else:
                        self.voting[two_max[0]] += 1
                    self.integral_sum = self.find_integrals(xi, all_feature_sum)
                except:
                    print(traceback.format_exc())

                # 5.4. Variante 4: Average of current heats without outliers
                found_risk = avg
                if not math.isnan(found_risk):
                    feature_favorites.append(found_risk)

            except:
                found_risk = -1
        
        # 6. print results
        found_rul = (found_risk - 1)/m
        try:
            self._print_likeliness(found_risk, cur_test_df, xi, m, found_rul, expected_rul)
            if self._visualize_summed_curve: self._visualize_summed_curve_m(xi, all_feature_sum)
        except: 
            pass#print("No")
        
        return found_risk, found_rul, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict   
    
    def _do_prediction(self, cur_test_df, grid_area, trained_scalers, dimensions, train_df, m, all_feature_sum, features_to_skip, timestamp_gap, expected_rul, fine = -1, dev_dict_path = ""):
        # 1. initialize
        try:
            if not all_feature_sum:
                all_feature_sum = np.empty(grid_area) # resulting curve is sum of subcurves
                all_feature_sum.fill(0)
        except: pass
    
        # 2. scale - adding column scaled
        cur_test_df = self._scale_with_scaler(cur_test_df, trained_scalers)
        feature_favs_dict = {}
    
        # 3. Prediction total model
        if fine == -1:
            found_risk, found_rul, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict = self._do_whole_model_prediction( cur_test_df, grid_area, dimensions, m, all_feature_sum, features_to_skip, timestamp_gap, dev_dict_path)
            
            return found_risk, found_rul, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict
        
        # 3. Prediction side model        
        if fine != -1:
            weight_b, feature_favorites, per_feature_sum  = True, [], {}
            if self.skip_never: features_to_skip = []

            for col in cur_test_df.columns:
                # 3.1. Initialize                
                info_key = "fine_" + str(fine) + "_" + col
                if HeatmapConvolutionTrainer.scaled_relevant_columns(col) or col in features_to_skip:  continue            
                summ, all_good = self._initialize_feature(grid_area, col)
                cur_dev = 0
              
                try:                    
                    l_iter_e = len(cur_test_df[col])
                    l_iter_s = 0
                    if l_iter_e > 6:
                        l_iter_s = len(cur_test_df[col]) - 6
                        l_iter_e = len(cur_test_df[col])
                    
                    # 3.2. iterate over rows
                    for cur_row in list(range(l_iter_s, l_iter_e)):
                                               
                        try:
                            cur_heat_values, xi = self._get_current_heat(cur_test_df, dimensions, cur_row, m, grid_area, col, info_key)
                        except:
                            continue
                        if self._smooth_per_feature: cur_heat_values = medfilt(cur_heat_values, self.smoothing_side)
                        summ = summ + cur_heat_values
                        if self._visualize_pre_post_risk: self._visualize_pre_post_risk_m(xi, cur_model, y_rel_row_idx, cur_heat_values) 
                
                    # 3.3. Shift by bias that was determined
                    bias = cur_dev/m
                    summ, xi  = self._shift_array(xi, summ, bias, m, grid_area)

                
                    # 3.4. Timestamp gap and store
                    summ, xi  = self._shift_array(xi, summ, timestamp_gap, m, grid_area)
                    per_feature_sum[col] = deepcopy([xi, cur_heat_values])
                    #if summ.argmax() == 0: continue #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    all_feature_sum = all_feature_sum + summ
                    if self._visualize_per_feature_curve: self._visualize_per_feature_curve_m(xi, summ)
                    
                    #if self._enable_all_print: print(str(col)+ " - Most-likely risk (Gesamtheit): " + str(xi[np.unravel_index(summ.argmax(), summ.shape)[0]]))
                    if summ.argmax() != 0:                                                
                        feature_favs_dict[col] = xi[np.unravel_index(summ.argmax(), summ.shape)[0]] # WIRD ANDERS BESTIMMT IM ENDERGEBNIS -> WAHL SOLLTE NACH TOP 10 Erfolgen
                    else:
                        pass
                    
                    # 4.2. Variante 2 - top 10 % weighted average
                    ten_per_idx =  math.floor(len(all_feature_sum)*self.percentage_side_fine)
                    ind = self.largest_indices(all_feature_sum, ten_per_idx)
                    weight = all_feature_sum[ind]
                    max_val = numpy.max(weight)
                    normalized_weight = weight/max_val                        
                    weighted_res = numpy.sum(xi[ind] * normalized_weight)/numpy.sum(normalized_weight)
                    found_risk = [xi[ind], normalized_weight]

                    # 4.3. Variante 3 - Maximum of curve
                    feature_favorites.append(xi[np.unravel_index(summ.argmax(), summ.shape)[0]]) 

                    
                except:
                    found_risk = -1
                 
            # 5. print results
            found_rul = -1
            try:
                self._print_likeliness(found_risk, cur_test_df, xi, m, found_rul, expected_rul)
                if self._visualize_summed_curve: self._visualize_summed_curve_m(xi, all_feature_sum)
            except:
                pass

            
            return found_risk, found_rul, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict
    
    def largest_indices(self, ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = numpy.argpartition(flat, -n)[-n:]
        indices = indices[numpy.argsort(-flat[indices])]
        return numpy.unravel_index(indices, ary.shape)
    
    
    def _apply_average_optimization(self, feature_favorites):
        '''
        optimizes the result: 
        1. determine average of all favorites per feature
        2. remove fields that are further away then 0.05 
        3. risk is average of remaining
        '''
        
        # 1. if more than 33% are within a significant interval this interval is likely to be risk
        risk, applicable = self._optimize_by_confidence(pref)
        
        # 2. take average of remaining elements
        if not applicable:
            # 1. outlier detection
            mean_val = mean(feature_favorites)
            remaining = [f for f in feature_favorites if (math.fabs(f - mean_val) < 0.05)]
            
            # 2. return result 
            risk = mean(remaining)
        
        return risk
    
    def _print_likeliness(self, found_risk, test_df_first, xi, m, found_rul, expected_rul):
        #if self._enable_all_print: print("Most likely RUL: " + str(found_rul))
        #if self._enable_all_print: print("Most-likely risk: " + str(found_risk))
        expected_risk = 1 + m*expected_rul
        #if self._enable_all_print: print("Expected risk: " + str(expected_risk))
        if self._enable_all_print: print("Error (related to critical area): " + str(math.fabs(100*((expected_risk - found_risk)/(max(xi)-min(xi))))) + " %")

    def _optimize_by_confidence(self, feature_favorites):
        if self._enable_all_print: print("No Confidence Optimization")
        return 0, False
        
         
        s_perf = sorted(perf)
        gaps = np.diff(s_perf)
        split_idx = np.where(gaps>0.02)
        prev_idx = -1
        
        # confidence variable
        conf_interval = 0.4

        indices = split_idx[0]
        el_list = []
        for idx in indices:
            idx += 1    
            if idx == indices[-1]+1:
                el_list.append(s_perf[idx:])                
            if idx == indices[0]+1:
                el_list.append(s_perf[:idx])    
            else:
                el_list.append(s_perf[prev_idx:idx])
            prev_idx = idx
        
        # within each list split by 0.02 - wenn sich die Dinger einig sind dann muss ich reagieren
        res_list = []
        cnt = 0
        for cur_list in el_list: 
            if self._enable_all_print: print("run "+str(cnt));cnt+=1
            s_cur_list = sorted(cur_list)
            
            # split idx
            steps = np.arange(min(s_cur_list), max(s_cur_list), 0.02)
            if len(steps)<1:
                res_list.append(s_cur_list)
                continue
            i = 0
            cList = []
            for el in s_cur_list:        
                if i == len(steps) and el == s_cur_list[-1]:
                    cList.append(el)
                    res_list.append(cList)
                    cList = []
                else: 
                    if i != len(steps) and el >= steps[i]:
                        i += 1
                        if cList:
                            res_list.append(cList)
                        cList = []
                    cList.append(el)
        
        # if now any container has more than 33% of all elements confidence is high, thus, return its value
        thr_elements = conf_interval*len(s_perf)
        lst = [np.mean(l) for l in res_list if len(l)>=thr_elements]
        if lst:
            mean_risk_elems_with_highest_confidence = np.mean(lst)
            if self._enable_all_print: print("Found optimized risk " + str(mean_risk_elems_with_highest_confidence))
            return mean_risk_elems_with_highest_confidence, True
        else: 
            if self._enable_all_print: print("NORMAL RISK BETTER")
            return 0, False

    def _visualize_summed_curve_m(self, xi, all_feature_sum):
        ''' visualize total curve'''        
        if self._enable_all_print: print("Gesamtheit aller Feature: ")
        plt.plot(xi, all_feature_sum)
        plt.xlabel("risk")
        plt.ylabel("all feature total heat")
        plt.show()
          
    def _visualize_per_feature_curve_m(self, xi, summ):
        '''
        Plot feature result 
        '''
        if self._enable_all_print: print("Total sum for this Feature: ")
        plt.plot(xi, summ)
        plt.xlabel("risk")
        plt.ylabel("total heat")
        plt.show()
          
    def _get_current_heat(self, cur_test_df, dimensions, cur_row, m, grid_area, col, dim_info = False):
        ''' compute the heat values of the current row and shift it if required
        
        :param cur_test_df: input test dataframe
        :param dimensions: trained model with heat map
        :param cur_row: index of current row in test dataframe
        :param grid_area: size in x and y of the heatmap grid
        :return heat value curve as determined
        '''
        
        
        # 1. initialize
        if not dim_info:
            dim_info = col
        cur_model = dimensions[dim_info][0]
        xi = dimensions[dim_info][1]
        yi = dimensions[dim_info][2]
        
        # 2. get feature row - assume yi to be sorted and get index with y closest to our value
        y_val = list(cur_test_df[col])[cur_row] 
        y_rel_row_idx = bisect_left(yi, y_val)
        
        
        # 3. Shift curves e.g. kriege Kurve 1,2,3,2,1 - d.h. die Wschlkt das ich an RUL der Stelle 3 bin     und bin 144 vor Ende 
        rul_shift = (-1) * (cur_test_df["TS"].max() - cur_test_df["TS"].iloc[cur_row])
        shift = m*rul_shift             # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IN ANDERE RICHTUNG?
        avg_abstand_x = (max(xi)-min(xi))/grid_area
        idx_shifts = math.floor((shift/avg_abstand_x)+0.499)   
        
        # 4. extract heat curve from model - shift nach recht
        if y_rel_row_idx == 1000: y_rel_row_idx = 999
        cur_heat_values = deepcopy(cur_model[:][y_rel_row_idx]) 
        if idx_shifts >0:
            if idx_shifts > len(cur_heat_values): # shifting to far
                arr = np.empty(len(cur_heat_values))# fulle idx_shifts neue Werte rechts dazu dann habe 0000 val val val
                arr.fill(0)
                cur_heat_values = arr
            else:
                cur_heat_values = cur_heat_values[:-idx_shifts] # hinten abschneiden
                arr = np.empty(idx_shifts)# fülle idx_shifts neue Werte links dazu dann habe 0000 val val val
                arr.fill(0)
                cur_heat_values = np.concatenate([arr, cur_heat_values]) 

        # shift nach links
        elif idx_shifts <0:
            if idx_shifts< -len(cur_heat_values): # shifting to far
                arr = np.empty(len(cur_heat_values))# fulle idx_shifts neue Werte rechts dazu dann habe 0000 val val val
                arr.fill(0)
                cur_heat_values = arr
            else:
                idx_shifts = (-1)*idx_shifts
                cur_heat_values = cur_heat_values[:-idx_shifts] # hinten abschneiden
                arr = np.empty(idx_shifts)# fülle idx_shifts neue Werte rechts dazu dann habe 0000 val val val
                arr.fill(0)
                cur_heat_values = np.concatenate([cur_heat_values, arr])
        return cur_heat_values, xi
    
    def _shift_array(self, xi, target_array, rul_shift, m, grid_area):
        yi = target_array
        
        # 3. Shift curves
        shift = numpy.abs(m * rul_shift)
        avg_abstand_x = (max(xi)-min(xi))/grid_area
        idx_shifts = math.floor((shift/avg_abstand_x)+0.499)   
        
        # 4. shift nach links neo neg
        if rul_shift <0:
            target_array = target_array[idx_shifts:] # hinten abschneiden
            arr = np.empty(idx_shifts)# fülle idx_shifts neue Werte links dazu dann habe 0000 val val val
            arr.fill(0)
            cur_heat_values = np.concatenate([target_array, arr]) 
        # shift nach rechts
        elif rul_shift >0:
            target_array = target_array[:-idx_shifts] # hinten abschneiden
            arr = np.empty(idx_shifts)# fülle idx_shifts neue Werte rechts dazu dann habe 0000 val val val
            arr.fill(0)
            cur_heat_values = np.concatenate([arr, target_array])
        else:
            cur_heat_values = target_array
        return cur_heat_values, xi
        
                                
    def _scale_with_scaler(self, cur_test_df, trained_scalers):
        '''
        to make the test data comparable to the training data a scaling needs to be performed 
        in the same way
        :param cur_test_df: Test data frame that has the features that are to be scaled
        :param trained_scalers: Dictionary mapping ids of features to the scaler used in training
        :return test_df: tested dataframe with additional column scaled_FEATURE_XX
        '''
        for col in cur_test_df.columns:
            if HeatmapConvolutionTrainer.relevant_columns(col):       
                continue 
            #print("CURRENT -> "+ col)
        
            # standard scaler (zentrieren und skalieren)
            try:
                cur_test_df['scaled_'+col] = trained_scalers[col].transform(cur_test_df[col])
            except:
                cur_test_df['scaled_'+col] = trained_scalers[col].transform(cur_test_df[col].reshape(-1,1))
        
        return cur_test_df
        
    
    def _visualize_pre_post_risk_m(self, xi, cur_model, y_rel_row_idx, cur_heat_values):
        '''
        visualize result prior and after shifting
        '''
        if self._enable_all_print: print("Before shift")
        plt.plot(xi, cur_model[:][y_rel_row_idx])
        plt.xlabel("risk")
        plt.ylabel("heat")
        plt.show()
        
        if self._enable_all_print: print("After shift")
        plt.plot(xi, cur_heat_values)
        plt.xlabel("risk")
        plt.ylabel("heat")
        plt.show()
    
    def _extract_prediction_information(self, data, cluster_id, test = False):
        '''
        for the prediction data needs to be stored during training, the stored values 
        are loaded here
        :param data: data dictionary as passed by the main execution chain
        :return grid_area: size of the grid of the heat map (grid_area x grid_area)
        :return trained_scalers: scaling of each feature
        :return dimensions: the heat map
        :return train_df: dataframe of the training data
        :return m: slope of ratio rul to risk i.e. risk = rul * m
        '''
        
        if test: 
            grid_area = data["test_CL_" + str(cluster_id) + "_" + "train_model_grid_area"]
            trained_scalers = data["test_CL_" + str(cluster_id) + "_" + "train_model_trained_scalers"]
            dimensions = data["test_CL_" + str(cluster_id) + "_" + "train_model"]
        else:
            
            grid_area = data["CL_" + str(cluster_id) + "_" + "train_model_grid_area"]
            trained_scalers = data["CL_" + str(cluster_id) + "_" + "train_model_trained_scalers"]
            dimensions = data["CL_" + str(cluster_id) + "_" + "train_model"]
        
        
        train_df = data["train_risc"].to_frame()
        if 'rul' in (train_df.columns):
            train_df = train_df.rename(columns={'rul': "RISK"})
        else:
            train_df = train_df.rename(columns={"RUL": "RISK"})
        train_df["RUL"] = data["train_rul"].to_frame()
        
        # 1. Determine linear mapping rul to risk
        train_df = train_df.dropna().sort_values("RUL").drop_duplicates()
        m = (train_df.iloc[5]["RISK"] - train_df.iloc[10]["RISK"])/(train_df.iloc[5]["RUL"] - train_df.iloc[10]["RUL"])

        
        return grid_area, trained_scalers, dimensions, train_df, m 
        
    def _extract_dataframe(self, test_df, object_id, rul_thr_min, rul_thr_max, cluster_id):
        ''' From the test dataframe extract the part with RULs between min and max and with 
            the given object id, as well as the specified cluster_id
        
        :param test_df: Dataframe  with testdata
        :param object_id: Id of the current object to be extracted
        :param rul_thr_min: Only RUL values bigger or equal than this are extracted
        :param rul_thr_max: Only RUL values smaller or equal than this are extracted
        :param cluster_id: Identifier of the cluster to be extracted (that is assigned to the object_id)
        :return test_df: Dataframe satisfying given conditions
        '''
        
        test_df = test_df[test_df['cluster_id']==cluster_id]
        test_df = test_df.sort_values("TS")
        df1 = test_df[test_df["id"]==object_id]
        test_df_first = df1[(df1["RUL"] >= rul_thr_min) & (df1["RUL"] <= rul_thr_max)] # FOR Evaluation only - in reality not needed
        
        return test_df_first
        
        
        
    def _assign_cluster(self, data, test_df):
        ''' 
        From the clustering algorithm performed in the training phase 
        get the cluster id by first scaling each row of the data then applying
        a pca to it and then using the cluster predictor of the training phase
        :param data dictionary as passed by the main execution chain
        :param test dataframe with all features
        :return test dataframe with additional column cluster_id indicating the number of the assigned cluster
        '''
        try:
            test_df["cluster_id"] = data["test_cluster_id"]
        except:
            test_df["cluster_id"] = 0
            print("No cluster assigned")
        return test_df
        
        
        # 1. Determine Cluster - ACHTUNG: immernoch wird nur cluster 1 trainiert!
        cluster_scaler = data["train_cluster_stored"][0]
        cluster_pca = data["train_cluster_stored"][1]
        cluster_predictor = data["train_cluster_stored"][2]
        
        # 2. add cluster ids
        feature_id_df = test_df[test_df.columns[pd.Series(test_df.columns).str.startswith('FEATURE_')]]
        feature_id_df = feature_id_df.join(test_df["id"])
        grouped_feature_id_df  = feature_id_df.groupby("id")
        
        # 3. Map to selected features per id
        dataset = data['meta_dataset_name']
        prep_features_df = grouped_feature_id_df.apply(ClusterTraining.feature_extraction, dataset)
        prep_features_df = prep_features_df[prep_features_df.columns[pd.Series(prep_features_df.columns).str.startswith('CL_FEATURE')]]
        
        # 4. apply scale, pca and cluster
        c_data = cluster_scaler.transform(prep_features_df.as_matrix())
        c_data = cluster_pca.transform(c_data)
        c_data = cluster_predictor.predict(c_data)
        test_df["cluster_id"] = c_data
        
        return test_df
        
    def _extract_critical_data_frame(self, data):
        '''
        extracts the critical part of the testing dataframe
        :param data dictionary as passed by the main execution chain
        :return dataframe concatenated with features, ruls, ids etc.
        '''
        test_df = data["test_t"].to_frame().join(data["test_X"])
        test_df['test_crit'] = 1#data['test_crit']#.to_frame().join(test_df)
        
        test_df = test_df.rename(columns={"RUL": "CRIT"})
        test_df = test_df.rename(columns={"RUL": "RISK"})
        
        test_df = data['test_id'].to_frame().join(test_df)
        test_df = data['test_rul'].to_frame().join(test_df)
        test_df["id"] = data['test_id']
        if not self._whole_data_set:
            test_df = test_df[test_df["CRIT"]>0]
        test_df["TS"] = data["test_ts"]
        return test_df
    


    def run_initialize(self, object_id, test_df,  test_mode):
        
        #Logging.log("\n\nOo oO --------> Eval: OBJECT ID: "+str(object_id))
        all_feature_sum = False
        cur_df1 = test_df[test_df['id'] == object_id]    
        timestamp_gap = 0 # PER Cluster need to shift incoming data else I cannot sum it up
        last_ts = 0
        expected_rul = 99999999
        all_feature_favorites = []
        predicted_risk, predicted_rul = -1, -1
        cnt = 4
        
        if  test_mode: 
            # remove random number of last RULs
            all_rul = list(cur_df1["RUL"])
            random.shuffle(all_rul )
            thrshld = all_rul[0]
            if len(cur_df1) > 1:
                cur_df1 = cur_df1[cur_df1["RUL"]>thrshld]
            
        return all_feature_sum, cur_df1, timestamp_gap, last_ts, expected_rul, all_feature_favorites, predicted_risk, predicted_rul, cnt
        
    
    def get_cluster_values(self, cur_df1, cluster_id):
        
        #Logging.log("\n\n                --------> Eval: CLUSTER ID: "+str(cluster_id))
        cur_df2 = cur_df1[cur_df1['cluster_id'] == cluster_id]
        
        cur_df3 = cur_df2.sort_values("RUL", ascending=False)

        # TEST - bestimmt über Feature selection
        skip_features = {}
        if not self.skip_never_whole and self.feature_selection_on:
            try:
                skip_features = self.skip_features[0]
                print("Skip WHOLE")
                print(str(skip_features))
            except:
                pass


        # per object predict only the maximal
        first = True
        return cur_df3, skip_features, first
    
    def shorten_segment(self, current_test_df):        
        dist = current_test_df["RUL"].max()- current_test_df["RUL"].min()
        if dist > segment_thrshld:
            print("SHORTENED RUL AREA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
            thrshld = current_test_df["RUL"].min() + segment_thrshld
            current_test_df = current_test_df[current_test_df["RUL"] < thrshld]
            
        return current_test_df
    
    def get_skip_feats(self, feature_skipping, skip_features, cluster_id):
        if not feature_skipping: return []
        try:
            skip_features = self.skip_features[0]
        except:
            pass
        # 4. run tester for this data frame and add column predicted
        try: skip = skip_features[int(cluster_id)]
        except: skip = []
        return skip

    def get_skip_features(self, finetuner, cluster_id):
        if self.feature_selection_on:
            try:
                skip_features = self.skip_features[finetuner+1]
            except:
                pass
        try: skip = skip_features[int(cluster_id)]
        except: skip = []

        return skip
    
    def whole_model_prediction(self, cur_df1,  expected_rul, predicted_risk, predicted_rul, all_feature_sum, timestamp_gap, data, all_feature_favorites, only_cluster_id, only_object_id, last_ts, dev_dict_path1, test_mode = False):
        
        # CONFIGURATION
        segmentation_shortening = False
        feature_skipping = True
        whole_model_features = []
        m = -1
        all_votes = []
        self.integral_sum = [0,0,0]
        for cluster_id in list(cur_df1["cluster_id"].unique()):
            if test_mode and not (cluster_id == 3):
                continue
            if only_object_id and cluster_id not in only_cluster_id: continue                
            current_test_df, skip_features, first = self.get_cluster_values(cur_df1, cluster_id)
            self.voting = [0.0, 0.0, 0.0]
            
        
            # 1. OPTIMIERUNGEN
            #Logging.log("--------> Eval: RUL RANGE: "+str(current_test_df["RUL"].max())+ " to "+str(current_test_df["RUL"].min()))
            if segmentation_shortening: current_test_df = self.shorten_segment(current_test_df)
            if feature_skipping: skip = self.get_skip_feats(feature_skipping, skip_features, cluster_id)
            if self.skip_never_whole: skip = []
    
            # 2. shift the input curve to align with the one processed next 
            if last_ts != 0:
                cur_ts = current_test_df["TS"].max()
                timestamp_gap = cur_ts - last_ts
            if current_test_df["RUL"].min() < expected_rul:  # store last Timestamp for shifting if it is more urgent
                expected_rul = current_test_df["RUL"].min()
    
            # 3. run prediction
            prev_risk, prev_rul = predicted_risk, predicted_rul
            predicted_risk, predicted_rul, m, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict = self._predict_RUL(data, current_test_df, cluster_id, all_feature_sum, skip, timestamp_gap, expected_rul, dev_dict_path = dev_dict_path1)
            
            # count in timestamp bias            
            risk_gap = numpy.absolute(timestamp_gap*m)
            if timestamp_gap > 0:
                for p in range(len(whole_model_features)):                    
                    for f in whole_model_features[p][1]:
                        whole_model_features[p][1][f] += risk_gap
            if timestamp_gap < 0:
                for f in feature_favs_dict:
                    feature_favs_dict[f] += risk_gap             
            whole_model_features.append([cluster_id,  feature_favs_dict])
            
            
            try:
                if predicted_risk == -1: predicted_risk, predicted_rul = prev_risk, prev_rul
            
                # 4. resulting prediction
                f_in = [f for f in feature_favorites if f >= 0]
                all_feature_favorites += f_in
                predicted_risk = numpy.average(all_feature_favorites)

            except: 
                predicted_risk, predicted_rul = prev_risk, prev_rul

            # 5. optimize
            print("RISK of STAGE 1: " + str(predicted_risk))


            all_votes.append(copy.deepcopy(self.voting))
            
        fst = sum([a[0] for a in all_votes])
        scd = sum([a[1] for a in all_votes])
        thr = sum([a[2] for a in all_votes])
        # nehme von den 2 siegern den pessimistischeren
        two_max = numpy.array([fst, scd, thr]).argsort()[-2:][::-1].tolist()
        # pessimistic
        model = sorted(two_max)[-1]
        fine_indices = [model]

        # CASE: pessimistic ist Mitte + summ of integrals ist am größten in der Mitte - Wschl dann an rechter Ecke
        # Wegen PHM versuche so pessimistic zu sein wie möglich
        if numpy.argmax(numpy.array(self.integral_sum)) == 1:
            fine_indices = [2]
        #print("RESULTING VOTE: "+ str([[fst, scd, thr], [model]]))

            
        return fine_indices, predicted_risk, whole_model_features, m, expected_rul
    


        
