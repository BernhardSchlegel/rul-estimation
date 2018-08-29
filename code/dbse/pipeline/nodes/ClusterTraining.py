from dbse.pipeline.BaseNode import BaseNode
import numpy as np
from dbse.tools.Logging import Logging
from dbse.tools.Visual import Visual
from sklearn import preprocessing
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn        
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P
import math
import numpy


class ClusterTraining(BaseNode):
    """
    Assigns a cluster to the training data for all critical values only (i.e. where CRIT=1)
    
    Parameters are
    1: ???
    2: ??? second parameter needs to be of type int (rul percentile). This specified which percentiles are filtered out.
       Recommmended value is 99
    """

    def __init__(self, target_algorithm: str, algorithm_params: list,
                 visualize_cluster: bool, 
                 visualize_series: bool,
                 whole_data_set = False,
                 field_in_train_t="train_t",
                 field_in_train_X="train_X",
                 field_in_train_crit="train_crit",
                 field_in_train_risc="train_risc",
                 field_in_train_id="train_id",
                 field_in_train_rul="train_rul",
                 field_in_meta_dataset_name = 'meta_dataset_name',
                 field_out_train_cluster_id="train_cluster_id", 
                 field_out_cluster_storage = "train_cluster_stored"):
        """
        Constructor
        :param whole_data_set: if true the evaluation is performed not only on critical parts but on the whole dataset
        :param target_algorithm: Target algorithm for clustering. Possible values are 'kmeans'
        :param algorithm_params: Parameters for the target algorithm given as list, for kmeans those are (number_of_expected_clusters)
        :param visualize_cluster: Boolean specifiying if a visualization of the clustering result is to be shown  
        :param visualize_series: Boolean specifiying if a visualization of each feature in terms of assigned cluster is to be plotted 
        :param field_in_train_t: Timestamp of the training data
        :param field_in_train_X: Features of the training data
        :param field_in_train_crit: 1 if row is in critical area and 0 otherwise
        :param field_in_train_risc: risk score at each point of time
        :param field_in_train_id: object identifier of training object
        :param field_in_train_rul:RUL of training data
        :param field_out_train_cluster_id: cluster identifier assigned to each object (numeric from 0 to numberOfClusters-1)
        :param field_out_cluster_storage: list [cluster_scaler, pca, clusterpredictor] to predict cluster_id for unknown data i.e. scale, pca then predict
                                         -> ATTENTION need to determine Features as in feature_extraction and scale and do PCA 
                                            
        """
        # parameters
        self._target_algorithm = target_algorithm
        self._algorithm_params = algorithm_params
        self._visualize_series = visualize_series
        self._visualize_cluster = visualize_cluster
        self._whole_data_set = whole_data_set
        
        # fields
        self._field_in_train_t = field_in_train_t
        self._field_in_train_X = field_in_train_X
        self._field_in_train_crit = field_in_train_crit
        self._field_in_train_risc = field_in_train_risc
        self._field_in_train_id = field_in_train_id
        self._field_in_train_rul = field_in_train_rul
        self._field_in_meta_dataset_name = field_in_meta_dataset_name
        self._field_out_train_cluster_id = field_out_train_cluster_id
        self._field_out_cluster_storage = field_out_cluster_storage
        
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
        
        # 1. transform to dataframe
        train_df = self._extract_data_frame(data_in)
        if not self._whole_data_set:
            train_df = train_df[train_df["CRIT"]>0]
        
        # 2. extract dataframe with only features per object id 
        dataset = data_in[self._field_in_meta_dataset_name]
        prep_features_df = self._extract_cluster_features_per_id(train_df, dataset)
        
        # 3. scale and transform to numpy array      
        cluster_scaler = StandardScaler()
        cluster_scaler = cluster_scaler.fit(prep_features_df.as_matrix())
        data = cluster_scaler.transform(prep_features_df.as_matrix())
        data_in[self._field_out_cluster_storage] = [cluster_scaler]
        
        # 4. apply clustering: returns a column with cluster ids for all critical examples
        if self._target_algorithm == "kmeans":
            data_in[self._field_out_train_cluster_id], cluster_storage = self._cluster_kmeans(data, dataset)
            
        data_in[self._field_out_cluster_storage] += cluster_storage

        # 5. per object extend
        '''
        for obj_id in list(train_df["id"].unique()):
            cur_train_df = train_df[train_df["id"] == obj_id]
            # per feature column do polyfit and extrapolate to a rul of 400
            for col in cur_train_df.columns:
                if not col.startswith("FEATURE"): continue
                
                # plot before
                plt.plot(cur_train_df["RUL"], cur_train_df[col])
                
                # fit curve
                c,d = P.polyfit(cur_train_df["RUL"], cur_train_df[col], 1, full=True)
                
                # extrapolated value
                rul = numpy.linspace(cur_train_df["RUL"].max(), 400, 1+400-cur_train_df["RUL"].max())
                ts = numpy.linspace(cur_train_df["TS"].max(), 400, 1+400-cur_train_df["RUL"].max())
                
                new_y_wert = c[1] * rul + c[0] # c[2]*numpy.power(rul, 2) + c[1] * rul + c[0]
                plt.plot(rul, new_y_wert, color ="red")
                plt.show()
                
                # add to the end of list 
                print("TODO ADD TO END OF LIST")
                
                # add to train_X
                #field_in_train_t="train_t",
                #field_in_train_X="train_X",
                #field_in_train_crit="train_crit",
                #field_in_train_risc="train_risc",
                #field_in_train_id="train_id",
                #field_in_train_rul="train_rul",
        '''
        # 6. visual result
        if self._visualize_series:
            self._visualize_feature_series(train_df, data_in[self._field_out_train_cluster_id])

        # 7. empty metrics
        metrics = dict() 
        
        return data_in, metrics 

    def _visualize_feature_series(self, train_df, cluster_ids):
        ''' plots each feature of the reduced training set with the 
            color of its assigned cluster
            :param train_df: dataframe of prepared features (only critical values!)
            :param cluster_ids: array of cluster ids corresponding to the reduced rows
        '''
        seaborn.set(style='ticks')
        train_df["train_cluster_id"] = cluster_ids
        
        for col in train_df.columns:
            if not col.startswith("FEATURE"): continue 
            Logging().log("CURRENT -> "+ col)
            _order = list(set(cluster_ids))
            fg = seaborn.FacetGrid(data=train_df, hue='train_cluster_id', hue_order=_order, aspect=1.61)
            fg.map(plt.scatter, 'RISK', col).add_legend()
            plt.show()

    def _cluster_kmeans(self, data, dataset):
        
        # 1. Set Parameters for K Means
        n_samples, n_features, n_clusters = self._kmeans_parse(data)
                
        # 2. Run k Means
        cluster_result, kmeans, reduced_data, cluster_storage = self._kmeans_run(n_samples, n_features, n_clusters, data)
        
        # 3. visualize
        if self._visualize_cluster:
            self._kmeans_visualize(kmeans, reduced_data)        

        return cluster_result, cluster_storage
        
    def _kmeans_visualize(self, kmeans, reduced_data):
        '''
        Visualize the result with a cluster plot
        
        :param kmeans: Object holding the kmean execution
        :param reduced_data: training data features after PCA was applied
        '''
        
        #1. Step size of the mesh /point in the mesh [x_min, x_max]x[y_min, y_max].
        h = .02

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z2 = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]) 

        # Put the result into a color plot
        Z2 = Z2.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z2, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')    
        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
        plt.title('K-means clustering (PCA-reduced data)\nCentroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def _kmeans_run(self, n_samples, n_features, n_clusters, data, pca_components = 2):
        '''
        Run k means algorithm with PCA prior to execution
        :param n_samples: Number of input examples 
        :param n_features: Number of features per example
        :param n_clusters: Number of expected target clusters  
        :param data: 2D array containing features in shape array([[ f1 f2 f3 f4, ...], [ f1 f2 f3 f4, ...], [ f1 f2 f3 f4, ...], ...])   
        :param pca_components: Number of PCA components used for clustering
        :return: cluster_result: Array of Cluster id corresponding to the given data
        :return: kmeans: Object holding the kmean execution
        :return: reduced_data: Data after PCA was applied
        '''
        # 1. Apply PCA
        pca_to_store1 = PCA(n_components = pca_components)
        pca_to_store = pca_to_store1.fit(data)
        reduced_data = pca_to_store1.fit_transform(data)
        
        # 2. Run K Means
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        kmeans.fit(reduced_data)    
        
        # 3. Retrieve Clustering results
        cluster_result = kmeans.predict(reduced_data)
        
        # 4. store clustering information
        cluster_storage = [pca_to_store, kmeans]
        
        return cluster_result, kmeans, reduced_data, cluster_storage

    def _kmeans_parse(self, data):
        ''' extract all required parameters for k means clustering
        
        :param data: 2D array containing features in shape array([[ f1 f2 f3 f4, ...], [ f1 f2 f3 f4, ...], [ f1 f2 f3 f4, ...], ...])   
        
        :return n_samples: Number of input examples 
        :return n_features: Number of features per example
        :return n_clusters: Number of expected target clusters  
        '''        
        
        expected_cluster_number = self._algorithm_params[0]
        
        np.random.seed(42)    
        n_samples, n_features = data.shape
        n_clusters = expected_cluster_number # Anzahl Cluster
        Logging().log("n_clusters: %d, \t n_samples %d, \t n_features %d" % (n_clusters, n_samples, n_features))
        return n_samples, n_features, n_clusters
        
    def _extract_cluster_features_per_id(self, train_df, dataset):
        ''' From the training dataframe per object id extract features based on which 
        common clusters are assigned       
                
        :param train_df Training dataframe containing all features
        :param dataset: name of input dataset
        :return dataframe with extracted features for this training data
        '''
        
        # 1. Group by id
        feature_id_df = train_df[train_df.columns[pd.Series(train_df.columns).str.startswith('FEATURE_')]]
        feature_id_df = feature_id_df.join(train_df["id"])    
        grouped_feature_id_df  = feature_id_df.groupby("id")
        
        # 2. Map to selected features per id
        prep_features_df = grouped_feature_id_df.apply(ClusterTraining.feature_extraction, dataset)
        prep_features_df = prep_features_df[prep_features_df.columns[pd.Series(prep_features_df.columns).str.startswith('CL_FEATURE')]]
        
        return prep_features_df
        
    def _extract_data_frame(self, data):
        ''' Creates a central dataframe from the given training data fusing 
            existing subelements of the data dictionary
        :param data dictionary as passed by the main execution chain
        :return dataframe concatenated with time criticality ris id and rul 
        '''
        
        train_df = data[self._field_in_train_t].to_frame().join(data[self._field_in_train_X])
        train_df["CRIT"] = data[self._field_in_train_crit]#.to_frame()
        train_df["RISK"] = data[self._field_in_train_risc].to_frame()
        train_df["id"] = data[self._field_in_train_id].to_frame()
        train_df["RUL"] = data[self._field_in_train_rul].to_frame()
        
        return train_df
        
    def feature_extraction(input_df, dataset):
        """
        Extracts features from an input dataframe, which holds all features of the 
        system. Those features can vary depending on the dataset. For the phm2008 dataset it is sufficient
        to choose the values of the least noisy features. However, for other datasets extracted features
        could be needed (e.g. max, min, mean, trend,...)
        :param input_df: dataframe holding feature values in each column 
        :param dataset: Name of the dataset, form which features are to be extracted
        :return: input_df dataframe extended by feature columns (CL_FEATURE_X) that can be used for clustering
        """
        
        if dataset == "phm2008":
            # Best Features found by visual inspection
            input_df["CL_FEATURE_1"] = input_df["FEATURE_1"] # ACHTUNG !!!!!!!!!!!!!!! DAS GILT NICHT fuR ALLE DATENSaTZE
            input_df["CL_FEATURE_2"] = input_df["FEATURE_2"]
            input_df["CL_FEATURE_3"] = input_df["FEATURE_5"]
            input_df["CL_FEATURE_4"] = input_df["FEATURE_6"]
            input_df["CL_FEATURE_5"] = input_df["FEATURE_5"]
            input_df["CL_FEATURE_6"] = input_df["FEATURE_5"]
            
        if dataset == "turbofan":
            input_df["CL_FEATURE_5"] = input_df["FEATURE_5"]
            input_df["CL_FEATURE_6"] = input_df["FEATURE_6"]
            input_df["CL_FEATURE_7"] = input_df["FEATURE_7"]
            input_df["CL_FEATURE_12"] = input_df["FEATURE_12"]
            
        else:
            # for unknown datsets use all feature columns
            for col in input_df.columns:
                if not col.startswith("FEATURE"): continue                
                else: input_df["CL_"+col] = input_df[col]
                
        return input_df
    
