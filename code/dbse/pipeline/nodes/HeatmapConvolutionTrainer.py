#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
import pandas as pd
from dbse.pipeline.BaseNode import BaseNode
from sklearn import preprocessing
import numpy as np
import copy
from dbse.tools.Visual import Visual
from dbse.tools.Logging import Logging
from scipy import signal
from scipy import misc
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from scipy.interpolate import griddata
import math
import matplotlib.pyplot as plt
import traceback
import warnings; warnings.simplefilter('ignore')
from dbse.pipeline.helper.iron_tool_belt import *
import numpy

class HeatmapConvolutionTrainer(BaseNode):
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
    '''
    

    def __init__(self,
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
                 field_out_train_model_trained_scalers = "train_model_trained_scalers",
                 visualize_outlier= False):
        
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
    
    def run(self, data_in, extract_frame_override = False, train_dff = None):
        super().run(data_in)  # do not remove this!
        Logging.log("Training da heat...")
                
        # 1. transform to df and keep critical
        if not extract_frame_override:
            train_df = self._extract_critical_data_frame(data_in)   
            train_df[self._field_in_train_cluster_id] = data_in[self._field_in_train_cluster_id]
        else: 
            train_df = train_dff
            
        
        for cluster_id in list(train_df["train_cluster_id"].unique()): # per cluster own model
            if self._test_mode and not (cluster_id == 3):
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

        # 5. empty metrics
        metrics = dict()

        return data_in, metrics
    
    def _remove_one_outlier(self, feature_df, col_name, nr_iterations =3, std_threshold=1):  
        if not self._remove_outliers:
            print("Outlier removal disabled!")
        
        # 1. Initialize
        iteration = range(nr_iterations)
        first = True
        
        # Per feature and window        
        result_df = feature_df.sort_values("RISK")
    
        # 3. iterate multiple times over window 
        #   on each iteration remove outliers
        for i in iteration:        
            sub_dfs = []
            indices = []
            rs = 0
            # 4. iterate over windows
            for r in np.linspace(result_df["RISK"].min(),1,20):

                sub_df = result_df[(rs <= result_df["RISK"]) & (r > result_df["RISK"])]     
                if self._remove_outliers:
                    sub_df = sub_df[((sub_df[col_name] - sub_df[col_name].mean()) / sub_df[col_name].std()).abs() < std_threshold]
                sub_dfs.append(sub_df) 
                rs = r
            result_df = pd.concat(sub_dfs)    

        return result_df
    
    def _build_one_heat_map(self, feature_df, risk_min, feature_min, feature_max, fine_tune = -1):
        Logging().log("Processing Feature: "+feature_df.columns[1])

        if fine_tune == -1:
            try:
                values = np.empty(len(feature_df))
                values.fill(1)
    
                # Assign X Y Z
                X = feature_df.RISK.as_matrix()
                Y = feature_df[feature_df.columns[1]].as_matrix()
                Z = values
        
                # create x-y points to be used in heatmap of identical size
                risk_min = feature_df.RISK.min()
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
            
                # horizontal interpolation
                for r in range(len(grad)):
                    # per dimension get first and last nonzero value
                    cur_line = grad[:, r]
                    nonzeros = numpy.where(cur_line > 0.0001)[0]
                    if list(nonzeros): 
                        a = 4
                        # fill von 0 bis nonzeros[0]
                        v = numpy.average(cur_line[nonzeros[0]:(nonzeros[0]+a)])
                        replacement = numpy.linspace(0, v, nonzeros[0]+a)[:(nonzeros[0])]
                        grad[:len(replacement), r] = replacement
                        
                        # fill von nonzeros[-1] bis len(grid)-1
                        v = numpy.average(cur_line[nonzeros[-1]-a:(nonzeros[-1])])
                        replacement = numpy.linspace(0, v, len(cur_line) - nonzeros[-1])[::-1]
                        grad[nonzeros[-1]:, r] = replacement

        
                # Store the model in memory
                feature_name = feature_df.columns[1]
                result = [feature_name, [copy.deepcopy(np.absolute(grad)), copy.deepcopy(xi), copy.deepcopy(yi)], grid_cur]
                
            except:
                feature_name = feature_df.columns[1]
                Logging().log(str(feature_df.columns[1])+": Feature skipped")
                result = [feature_name, None, None]
        
        else:
            if fine_tune == 0:
                feature_df = feature_df[feature_df["RISK"]<0.5] # hier changed!!!!!!!!!!!!!
            if fine_tune == 1:
                feature_df = feature_df[feature_df["RISK"] > 0.25]
                feature_df = feature_df[feature_df["RISK"] < 0.75]
            if fine_tune == 2:
                feature_df = feature_df[feature_df["RISK"] > 0.5]
                
            try:
                feature_df = self._remove_one_outlier(feature_df, feature_df.columns[1])
                
                values = np.empty(len(feature_df))
                values.fill(1)

                # Assign X Y Z
                X = feature_df.RISK.as_matrix()
                Y = feature_df[feature_df.columns[1]].as_matrix()
                Z = values
                risk_min = feature_df.RISK.min()
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
            
                # vertikale interpolation bis an Rand
                for r in range(len(grad)):
                    # per dimension get first and last nonzero value
                    cur_line = grad[:, r]
                    nonzeros = numpy.where(cur_line > 0.0001)[0]
                    if list(nonzeros): 
                        a = 4
                        # fill von 0 bis nonzeros[0]
                        v = numpy.average(cur_line[nonzeros[0]:(nonzeros[0]+a)])
                        replacement = numpy.linspace(0, v, nonzeros[0]+a)[:(nonzeros[0])]
                        grad[:len(replacement), r] = replacement
                        
                        # fill von nonzeros[-1] bis len(grid)-1
                        v = numpy.average(cur_line[nonzeros[-1]-a:(nonzeros[-1])])
                        replacement = numpy.linspace(0, v, len(cur_line) - nonzeros[-1])[::-1]
                        grad[nonzeros[-1]:, r] = replacement
        
                # Store the model in memory
                feature_name = feature_df.columns[1]
                result = ["fine_"+str(fine_tune)+"_" +feature_name, [copy.deepcopy(np.absolute(grad)), copy.deepcopy(xi), copy.deepcopy(yi)], grid_cur]
                
            except:
                #traceback.print_exc()
                feature_name = feature_df.columns[1]
                result = [feature_name, None, None]
            
        return result
                       
    def _build_heat_map_parallel(self, output_dfs):        
        
        # Parameter: nehme nur die 21 features fürs erste 
        output_dfs = output_dfs[:21]
        parallel_processes = numpy.ceil(len(output_dfs))
        
        # output ist dimensions        
        dimensions = {}
        print("Building " + str(len(output_dfs)) + " heat maps (if all work out)")
        
        # 1. create input arguments = output_dfs
        print([rm for rm in [df.RISK.min() for df in output_dfs] if not math.isnan(rm)])
        risk_min = min([rm for rm in [df.RISK.min() for df in output_dfs] if not math.isnan(rm)])
        feature_min = min([rm for rm in [df[df.columns[1]].min() for df in output_dfs] if not math.isnan(rm)])
        feature_max = max([rm for rm in [df[df.columns[1]].max() for df in output_dfs] if not math.isnan(rm)])
        input_args = []
        
        for l in output_dfs:
            for i in [-1,0,1,2]:
                input_args.append([l, -1, feature_min, feature_max, i])


        parallel_processes = math.ceil(len(input_args)/2)+2
        if parallel_processes>multiprocessing.cpu_count():
            parallel_processes=multiprocessing.cpu_count()

        res_list = parallelize_stuff(input_args, self._build_one_heat_map, simultaneous_processes = parallel_processes)
        
        # 2. output dimensions[feature_df.columns[1]] = output
        for res in res_list:
            name = res[0]
            dim = res[1]
            grid_cur = res[2]
            
            dimensions[name] = dim
            
            if True and self._visualize_heatmap:
                print("Feature " + str(name))
                try:
                    dim[0][::-1,::-1]
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
                risk_min = 0
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
        
                # no constant/zero values shall be allowed -> first - horizontal 
                # horizontal interpolation bis an Rand
                Logging.log("I AM NEW")
                for r in range(len(grad)):
                    # per dimension get first and last nonzero value
                    cur_line = grad[:, r]
                    nonzeros = numpy.where(cur_line > 0.0001)[0]
                    if list(nonzeros): 
                        a = 20
                        # fill von 0 bis nonzeros[0]
                        v = numpy.average(cur_line[nonzeros[0]:(nonzeros[0]+a)])
                        replacement = numpy.linspace(0, v, nonzeros[0]+a)[:(nonzeros[0])]
                        grad[:len(replacement), r] = replacement
                        
                        # fill von nonzeros[-1] bis len(grid)-1
                        v = numpy.average(cur_line[nonzeros[-1]-a:(nonzeros[-1])])
                        replacement = numpy.linspace(0, v, len(cur_line) - nonzeros[-1])[::-1]
                        grad[nonzeros[-1]:, r] = replacement
                
                # vertikale interpolation bis an Rand
                for r in range(len(grad)):
                    # per dimension get first and last nonzero value
                    cur_line = grad[r, :]
                    nonzeros = numpy.where(cur_line > 0.0001)[0]
                    if list(nonzeros): 
                        a = 20
                        # fill von 0 bis nonzeros[0]
                        v = numpy.average(cur_line[nonzeros[0]:(nonzeros[0]+a)])
                        replacement = numpy.linspace(0, v, nonzeros[0]+a)[:(nonzeros[0])]
                        grad[r, :len(replacement)] = replacement
                        
                        # fill von nonzeros[-1] bis len(grid)-1
                        v = numpy.average(cur_line[nonzeros[-1]-a:(nonzeros[-1])])
                        replacement = numpy.linspace(0, v, len(cur_line) - nonzeros[-1]+1)[::-1]
                        grad[r, nonzeros[-1]-1:] = replacement
        
        
                # Store the model in memory
                dimensions[feature_df.columns[1]] = [copy.deepcopy(np.absolute(grad)), copy.deepcopy(xi), copy.deepcopy(yi)]
                
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
        train_df["CRIT"] = data[self._field_in_train_crit]
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
            if HeatmapConvolutionTrainer.scaled_relevant_columns(col): continue
            result_df = train_df.sort_values("RISK")
        
            # 3. iterate multiple times over window 
            #   on each iteration remove outliers
            for i in iteration:
                sub_dfs = []
                indices = []
                rs = 0
                # 4. iterate over windows
                for r in np.linspace(result_df["RISK"].min(),1,split_windows):

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
    
    
    def relevant_columns(col):
        ''' skipping the ones given here e.g. if not FEATURE then skip '''
        return not col.startswith("FEATURE") and not col.startswith("SC___RDME") and not col.startswith("RO___") and not col.startswith("MV___EDME") and not col.startswith("CP___") and not col in ['STAT_FAKTOR_RS_WERT','STAT_TIME_TOTAL_WERT','STAT_ANZAHL_KUEHLVORGAENGE_WERT','STAT_HIS_PROG_LADEZEIT_5_WERT','STAT_HIS_PROG_LADEZEIT_6_WERT','STAT_HIS_PROG_LADEZEIT_7_WERT','STAT_HIS_SOC_WARN_GRENZEN_3_WERT','STAT_HISTO_SYM_DAUER_3_WERT','STAT_HISTO_SYM_ZELLANZAHL_4_WERT','STAT_HISTO_SYM_ZELLANZAHL_5_WERT','STAT_HISTO_SYM_ZELLANZAHL_6_WERT','STAT_I_HISTO_2_WERT','STAT_I_HISTO_3_WERT','STAT_I_HISTO_4_WERT','STAT_I_HISTO_6_WERT','STAT_I_HISTO_7_WERT','STAT_ENTLADUNG_KUEHLUNG_WERT','STAT_SCHUETZ_K2_RESTZAEHLER_WERT','STAT_MAX_SOC_GRENZE_WERT','STAT_ZEIT_SOC_12_WERT','STAT_ZEIT_TEMP_NO_OP_7_WERT','STAT_ZEIT_TEMP_TOTAL_5_WERT','STAT_HIS_EFF_CURR_CHG_1_TMID_WERT','STAT_FAKT_P2_T3_SOC5_WERT','STAT_RELATIVZEIT_4_WERT','STAT_ZEIT_POWER_CHG_2_WERT','MV___FAHRZEUGALTER']
    
    def scaled_relevant_columns(col):
        ''' skipping the ones given here e.g. if not scaled_FEATURE then skip '''
        return not col.startswith("scaled_FEATURE") and not col.startswith("scaled_SC___RDME") and not col.startswith("scaled_RO___") and not col.startswith("scaled_MV___EDME") and not col.startswith("scaled_CP___") and col not in ['scaled_STAT_FAKTOR_RS_WERT', 'scaled_STAT_TIME_TOTAL_WERT', 'scaled_STAT_ANZAHL_KUEHLVORGAENGE_WERT', 'scaled_STAT_HIS_PROG_LADEZEIT_5_WERT', 'scaled_STAT_HIS_PROG_LADEZEIT_6_WERT', 'scaled_STAT_HIS_PROG_LADEZEIT_7_WERT', 'scaled_STAT_HIS_SOC_WARN_GRENZEN_3_WERT', 'scaled_STAT_HISTO_SYM_DAUER_3_WERT', 'scaled_STAT_HISTO_SYM_ZELLANZAHL_4_WERT', 'scaled_STAT_HISTO_SYM_ZELLANZAHL_5_WERT', 'scaled_STAT_HISTO_SYM_ZELLANZAHL_6_WERT', 'scaled_STAT_I_HISTO_2_WERT', 'scaled_STAT_I_HISTO_3_WERT', 'scaled_STAT_I_HISTO_4_WERT', 'scaled_STAT_I_HISTO_6_WERT', 'scaled_STAT_I_HISTO_7_WERT', 'scaled_STAT_ENTLADUNG_KUEHLUNG_WERT', 'scaled_STAT_SCHUETZ_K2_RESTZAEHLER_WERT', 'scaled_STAT_MAX_SOC_GRENZE_WERT', 'scaled_STAT_ZEIT_SOC_12_WERT', 'scaled_STAT_ZEIT_TEMP_NO_OP_7_WERT', 'scaled_STAT_ZEIT_TEMP_TOTAL_5_WERT', 'scaled_STAT_HIS_EFF_CURR_CHG_1_TMID_WERT', 'scaled_STAT_FAKT_P2_T3_SOC5_WERT', 'scaled_STAT_RELATIVZEIT_4_WERT', 'scaled_STAT_ZEIT_POWER_CHG_2_WERT', 'scaled_MV___FAHRZEUGALTER']

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
            if HeatmapConvolutionTrainer.relevant_columns(col):     
                continue 
        
            # 2. standard scaler 
            scaler = preprocessing.StandardScaler()
            try:
                scaler = scaler.fit(train_df[col])
            except:
                scaler.fit(train_df[col].reshape(-1,1))
            try:
                train_df['scaled_'+col] = scaler.transform(train_df[col])
            except:
                train_df['scaled_'+col] = scaler.transform(train_df[col].reshape(-1,1))

            trained_scalers[col] = copy.deepcopy(scaler)
                
        
        return train_df, trained_scalers
        
