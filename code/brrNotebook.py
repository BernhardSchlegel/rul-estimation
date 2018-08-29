
# coding: utf-8

# ## Modular Pipeline overview
# 
# Pipeline consists of the following steps:
# 
# 1. 
# 
# 1. Train a scaler, normalize (mu = 0, std = 1) all samples (suffixed with `_scaled`)
# 1. Clustern, **HOW?** simple approach would be kmeans
# 1. **Assign criticality (CRIT)**
#     - A object is considered critical (`CRIT = 1`) if all of the following conditions are met:
#         - Object close to end of life (EOL), assessed `RUL < RUL_max * percentage`
#         - Object was destroyed at some point
#     - Training is done using all features (including TS) 
# 1. **Model CRIT**
#     - CRIT is modelled using a LR, training is done using all or a representative subset of samples
# 1. **Assign RISK**
#     - RISK is assigned **only** for critical samples (`_crit`), all other samples will have RISK = 0
#     - All critcal samples of objects will have a RISK assigned. 
#     - The risk ranged from 0 (samples of objects that just made it to being a critical one) to 1 (EOL reached).
#     - Risk is assigned using 
# 1. **Model RISK**
#     - Question does the previous RISK model serve as model? If not: What is a good fit here?
# 1. **Model RUL**
#     - Using RISK or without

# In[ ]:

# Unresolved question: How to determine when degradation starts?


# In[ ]:

SUPRESS_LOG = True # set this to true if you are badge evaluating something 


# In[ ]:

from common.log import log, ping, pong, ResultLogger
from common.data import DataSource, DataStructure
from common.score import score_rmse, output_phm_file, score_phm, score_phm_all


# In[ ]:

from common.log import ResultLogger
custom_fields = [
    "SCORE_RMSE_TRAIN",
    "SCORE_PHM_TRAIN",
    "TRAINING_TIME_MS_SOLELY_TRAINING",
    "TRAINING_TIME_MS_SOLELY_EVAL",
    "HYPER_BERNHARD_RUL_CAP",
    "HYPER_BERNHARD_RUL_CAP_DROP_ABOVE",
    "HYPER_BERNHARD_FEATURE_SELECT_MANUAL",
    "HYPER_BERNHARD_RUL_MODELTYPE",
    "HYPER_BERNHARD_RUL_ADD_DERIVED",
    "HYPER_BERNHARD_CRIT_SEPARATE_RUL",
    "HYPER_BERNHARD_ADD_HISTORY",
    "HYPER_BERNHARD_ADD_HISTORY_MODE",
    "META_BERNHARD_TRAIN_ID",
    "META_BERNHARD_TRAIN_POINT"]
rs = ResultLogger(approach_name = "bernhard_phd_redo_train", additional_header_fields = custom_fields)


# In[ ]:

# utils

def is_percentage(string_to_check):
    if isinstance(string_to_check, str) and "%" in string_to_check:
        return True
    return False
    

def percent_to_num(percent):
    if isinstance(percent, str):
        #log("using percentage {}".format(percent))
        percent_times_100 = int(percent.replace("%", ""))
        return percent_times_100 * 0.01
    else:
        return percent


# In[ ]:

import math

def cap_rul(ds : DataStructure, rul_cap, drop_above):
    rul_max = np.max(ds.rul)    
    if isinstance(rul_cap, str):
        rul_cap_used = rul_max * percent_to_num(rul_cap)
    else:
        rul_cap_used = rul_cap    
    log("used rul cap is {}".format(rul_cap_used))
    
    if drop_above:
        idxs = ds.rul <= rul_cap_used

        ds.X = ds.X[np.array(idxs)]
        if len(ds.y) > 0: ds.y = ds.y[idxs]
        ds.t = ds.t[idxs]
        ds.rul = ds.rul[idxs]
        ds.objectid = ds.objectid[idxs]
        ds.cluster = ds.cluster[idxs]
        ds.ts = ds.ts[idxs]
    else:
        rul_copy = ds.rul.copy()
        rul_copy[ds.rul >= rul_cap_used] = rul_cap_used
        ds.rul = rul_copy
        # for some strange reason, this adds RUL as a colum to x ... WTF ?! memory issue i guess
        # this is only the case, if the above lines are NOT executed
        # ds.rul[ds.rul >= rul_cap_used] = rul_cap_used

    return ds, rul_cap_used


# In[ ]:

# FEATURE FILTER
from bernhard.PolyFitter import PolyFitter


# In[ ]:

# RUL
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV # WORKAROUND, delete this, uncomment line above
from sklearn.metrics import make_scorer
import pickle  # dumping models
from pathlib import Path
import math

def add_derivatives(X):
    X = pd.DataFrame(X)
    dot = X.diff(axis=0)
    dotdot = dot.diff(axis=0)
    return np.array(pd.concat((X, dot, dotdot), axis=1).fillna(value=0))

def add_history(X, n_hist, mode="add_zero"):
    """
    add_zero or add copy (not yet supported)
    """
    df = pd.DataFrame(X)
    
    # create a copy once
    df_shifted = df.copy()
    
    res = df.copy()
    for shift in range(0, n_hist):
        if mode == "add_zero":
            df_shifted.loc[-1] = [0]*df.shape[1]  # adding a new, zero row
        elif mode == "copy":
            df_shifted.loc[-1] = df_shifted.loc[0]
            
        df_shifted.index = df_shifted.index + 1  # shifting index
        df_shifted = df_shifted.sort_index()  # sorting by index
        df_shifted = df_shifted[0:df.shape[0]] # back to original number of rows, drop last
    
        res = pd.concat([res, df_shifted], axis=1, join='inner') # merge to new dataframe

    return res

def regression_predict(X, regression_mdls, add_derived, selector_mdls=None, clusters=None, USE_CLUSTERS_FROM_DATASET = 0):
    print("DEPRECATED, DO NOT USE")
    1/0
    
    if add_derived > 0:
        log("adding derivatives to dataset")
        X = add_derivatives(X)
        
    if USE_CLUSTERS_FROM_DATASET is 0:
        # there was no clustering, predict using one model
        if selector_mdls is not None:
            X_sub = selector_mdls[0].select(X.as_matrix())
        else:
            X_sub = X
        return regression_mdls[0].predict(X_sub)
    else:
        assert clusters is not None, "I need cluster information when USE_CLUSTERS_FROM_DATASET is 1"
        
        all_predictions = np.array([None] * len(X))
        for cluster_id in range(0, len(np.unique(clusters))):
            log("predicting cluster {}".format(cluster_id))
            regression_model = regression_mdls[cluster_id]
            indices = clusters == cluster_id
            
            if selector_mdls is not None:
                selector_model = selector_mdls[cluster_id]
                
                if isinstance(X[indices], pd.DataFrame):
                    X_pre = X[indices].as_matrix()
                else:
                    X_pre = X[indices]
                X_sub = selector_model.select(X_pre)
            else:
                X_sub = X[indices]
                
            preds = regression_model.predict(X_sub)
            
            all_predictions[np.array(indices)] = np.array(preds)      
            
        return all_predictions
        
def scale_to_0_1(vals):
    feature_max = np.max(vals)
    feature_min = np.min(vals)
    if not math.isclose(feature_min, feature_max):
        vals = (vals - feature_min) / (feature_max - feature_min)
    else:
        vals = [0] * len(vals)
    return vals
    
def regression_fit(X, y, 
                     model_filename, 
                     reload_if_existing,
                     modeltype="RF", 
                     scoring="default",
                  use_sample_weight=True):
    model_filename = "./models/" + modeltype + "_" + model_filename
    
    if use_sample_weight:
        y_scaled = np.array(scale_to_0_1(y))
        w = np.subtract(np.ones_like(y_scaled ), y_scaled )
    else:
        w = np.ones_like(y)
    
    if reload_if_existing is False or Path(model_filename).exists() is False:
        log("training {} using {} samples for training.".format(
            modeltype, X.shape[0]))
        if modeltype == "RF":
            param_grid = {'max_depth': [5, 10, 20, 40],
                          'n_estimators': [3, 5, 10, 100]}

            clf = None
            if scoring == "phm":
                loss = make_scorer(PredictorWrapperRegression.phm_loss_function, greater_is_better=False)
                clf = GridSearchCV(RandomForestRegressor(n_jobs=-1), param_grid, scoring=loss)
            elif scoring == "default":
                clf = GridSearchCV(RandomForestRegressor(n_jobs=-1), param_grid)
            else:
                raise ValueError('PredictorWrapperRegression.train: unknown scoring mode.')

            mdl = clf.fit(X, y, w)
        if modeltype == "SVR":
            param_grid = {"C": [1e0, 1e1, 1e2, 1e3],
                          "gamma": np.logspace(-2, 2, 5)}
            clf = None
            if scoring == "phm":
                loss = make_scorer(PredictorWrapperRegression.phm_loss_function, greater_is_better=False)

                clf = GridSearchCV(svm.SVR(kernel='rbf'), cv=5,
                                   param_grid=param_grid,
                                   scoring=loss,
                                   n_jobs=-1)
            elif scoring == "default":
                clf = GridSearchCV(svm.SVR(kernel='rbf'), cv=5,
                                   param_grid=param_grid,
                                   n_jobs=-1)
            else:
                raise ValueError('PredictorWrapperRegression.train: unknown scoring mode.')

            mdl = clf.fit(X, y, w)

        # save model to file
        with open(model_filename, 'wb') as f:
            pickle.dump(mdl, f)

    else:
        log("restoring model from {}".format(model_filename))

        with open(model_filename, 'rb') as fid:
            mdl = pickle.load(fid)
        log("restored model")

    return mdl

def classification_fit(X, y, model_filename, reload_if_existing,
          modeltype="RF", cv_measure="recall"):
    """
    trains and evaluates a model based on the given data
    :param X_train: Features for training, expected to a numpy.ndarray.
    :param X_test: Features for testing, expected to a numpy.ndarray.
    :param y_train: Labels for training. Expected to an one-dimesional array.
    :param y_test: Labels for testing. Expected to an one-dimesional array.
    :param model_filename: Filename of model when serialized to disk
    :param reload_if_existing: Boolean indicating if model should be restored from disk if existing.
    :param modeltype: modeltype to train (RF, SVC or LRCV). RF is recommended since being fast to train and non-
                      linear - therefore usually yielding the best results.
    :param cv_measure: possible cv_measure are ['accuracy', 'precision', 'recall', 'roc_auc']
    :return: 
    """
    model_filename = "./models/" + modeltype + "_" + model_filename

    if reload_if_existing is False or Path(model_filename).exists() is False:
        log("training {} using {} samples and {} features ".format(
            modeltype, X.shape[0], X.shape[1]))
        if modeltype == "LRCV":
            log("Optimizing for {}...".format(cv_measure))
            lr = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1],
                                      cv=5,
                                      penalty='l1',
                                      scoring=cv_measure,  # Changed from auROCWeighted
                                      solver='liblinear',
                                      tol=0.001,
                                      n_jobs=mp.cpu_count())
            mdl = lr.fit(X, y)
        elif modeltype == "RF":

            param_grid = {'max_depth': [3, 5, 10, 15, 20],
                          'n_estimators': [3, 5, 10, 20]}

            clf = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, scoring=cv_measure)

            mdl = clf.fit(X, y)

        # save model to file
        with open(model_filename, 'wb') as f:
            pickle.dump(mdl, f)

    else:
        log("restoring model from {}".format(model_filename))

        with open(model_filename, 'rb') as fid:
            mdl = pickle.load(fid)

    return mdl


# In[ ]:

import numpy as np
import matplotlib.pyplot as plt

def filter_kalman(values, initial_value=0.0, Q=1e-3, R=0.1**2):
    """
    Kalman filter implementation based on 
    
    - http://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html by Andrew D. Straw
    - http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    
    param Q: process variance.
    param R: estimate of measurement variance, change to see effect. Unscientifically put: "The smaller, the faster
             the kalman filter will adapt the "mean"".
    """
    
    if isinstance(values, pd.core.series.Series):
        values = values.as_matrix()
    
    # intial parameters
    n_iter = len(values)
    sz = (n_iter,) # size of array

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    # intial guesses
    xhat[0] = initial_value
    P[0] = 1.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(values[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]

    return xhat

def demo_kalman(measurements, initial_value = 0.0,  Q=1e-5, R=0.05**2, real_values = None):
    xhat = filter_kalman(measurements, initial_value, Q, R)
    plt.figure()
    plt.plot(measurements,'k+',label='noisy measurements')
    plt.plot(xhat,'b-',label='a posteri estimate')
    if real_values is not None:
        plt.plot(real_values,'b--',label='real_values')
    #plt.axhline(x,color='g',label='truth value')
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.show()

demo = False
if demo:
    measurements = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 2, 3, 1, 2, 3, 4, 5, 6, 9, 5, 7, 10, 12, 9, 15]
    demo_kalman(measurements)


# In[ ]:

# SCORE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
def predict(X,
            clusters,
            regression_mdls,  
            selector_mdls,
            USE_CLUSTERS_FROM_DATASET,
            critical_models):
    # extract information
    n_critical_levels = regression_mdls.shape[1]
    
    # preparations
    criticality = np.ones_like(clusters) - 2 # should lead to -1
    criticality_confidence = np.zeros([len(X), n_critical_levels])
    all_predictions = np.array([None] * len(X))   
    all_predictions_crit = np.zeros([len(X), n_critical_levels])   
    if isinstance(X, pd.DataFrame):
        X = X.as_matrix()
    use_all_pred = False      
    
    # get number of clusters
    if USE_CLUSTERS_FROM_DATASET > 0:
        n_clusters = len(np.unique(clusters))
    else:
        n_clusters = 1  # there is at least one
    
    # for all clusters
    for cluster_id in range(0, n_clusters):
        if n_clusters > 1:
            indices = np.array(clusters == cluster_id)
        else:
            indices = np.ones_like(clusters) == 1

        # use selector model
        if selector_mdls is not None:
            selector_model = selector_mdls[cluster_id]
            
            # enhance data
            if RUL_ADD_DERIVED > 0:
                log("adding derivatives to dataset")
                X_enhanced = add_derivatives(X[indices])
            else:
                X_enhanced = X[indices]
            
            X_selected = selector_model.select(X_enhanced)
        else:
            selector_model = None
            
            # enhance data
            if RUL_ADD_DERIVED > 0:
                log("adding derivatives to dataset")
                X_selected = add_derivatives(X[indices])
            else:
                X_selected = X[indices]
            
        if ADD_HISTORY > 0:
            log("adding history to dataset")
            X_selected = add_history(X_selected, ADD_HISTORY, mode = ADD_HISTORY_MODE)
            
        # check if critical models are used
        if critical_models is not None:
            
            criticality_confidence[indices,:] = critical_models[cluster_id,0].predict_proba(X_selected)

            # for all criticalities
            for current_crit in range(0, n_critical_levels):
                #if selector_model is not None:
                #    X_selected = selector_model.select(X_selected)
                #else:
                #    X_selected = X[indices]

                # predict
                log("shape of regression_mdls is {}, {}".format(regression_mdls.shape[0], regression_mdls.shape[1]))
                preds = regression_mdls[cluster_id, current_crit].predict(X_selected)
                
                # add
                all_predictions_crit[np.array(indices),current_crit] = np.array(preds)
                use_all_pred = True
        else:
            # predict using on model using default crit (0)
            preds = regression_mdls[cluster_id, 0].predict(X_selected)

            # add
            all_predictions[np.array(indices)] = np.array(preds)    
            
    if use_all_pred:
        all_predictions = np.sum(np.multiply(all_predictions_crit,criticality_confidence), axis = 1)
            
    return all_predictions

def generate_frc(evalobject, regression_models, selector_models, critical_models):
    y_pred = predict(evalobject.X, 
                     clusters=evalobject.cluster, 
                     regression_mdls = regression_models,  
                     selector_mdls = selector_models,
                     USE_CLUSTERS_FROM_DATASET=USE_CLUSTERS_FROM_DATASET,
                     critical_models=critical_models)
    """y_pred = regression_predict(evalobject.X, 
                                add_derived = RUL_ADD_DERIVED, 
                                regression_mdls = regression_models,  
                                selector_mdls = selector_models,
                                clusters=evalobject.cluster, 
                                USE_CLUSTERS_FROM_DATASET=1,
                               critical_models=critical_models)"""
    final_result_combined = {'RUL_REAL': evalobject.rul, 
                             'RUL_PRED': y_pred, 
                             'OBJECTID': evalobject.objectid, 
                             'TS': evalobject.ts,
                             'CLUSTER': evalobject.cluster,
                             'RUL_PRED_SMOOTHED_CLUSTER': np.ones_like(y_pred),
                             'RUL_PRED_SMOOTHED_OBJECT': np.ones_like(y_pred)}
    
    return pd.DataFrame(data=final_result_combined)


def eval(evalobject: DataStructure, regression_models, selector_models=None, critical_models=None,
         offset=0, to_file=False, train_id=None):
    final_result_combined = generate_frc(evalobject, regression_models, selector_models, critical_models)

    final_result_combined["RUL_PRED"] = final_result_combined["RUL_PRED"] + offset
    test_phm = score_phm(final_result_combined, plot_heatmap_enabled=False, train_id=train_id)
    test_phm_all = score_phm_all(final_result_combined, plot_heatmap_enabled=False)
    test_rmse = score_rmse(final_result_combined["RUL_REAL"], final_result_combined["RUL_PRED"])
    log("test: rmse={}, phm={}".format(test_rmse, test_phm))

    if to_file:
        output_phm_file(final_result_combined, filename=to_file)

    return final_result_combined, test_phm, test_phm_all, test_rmse

def plot_heatmap(x, y):
    if not SUPRESS_LOG:
        heatmap, xedges, yedges = np.histogram2d(np.array(x), y, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.figure()
        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.show(block=False)
    
def plot_scatter(x, y, xlabel = "x", ylabel="y", title="default title"):    
    if not SUPRESS_LOG: 
        plt.scatter(x, y)       
        plt.xlabel(xlabel)         
        plt.ylabel(ylabel)     
        plt.title(title)
        plt.show(block=False)   


# In[ ]:

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn import metrics
import pandas as pd
from brr.ClusterFitter import ClusterFitter

def feature_selection_wang(X):
    """
    Authors in paper just used feature 2, 3, 4, 7, 11, 12, 15. So these features will be selected manually until
    sophisticated clustering is available.
    :param data: the data to do the calculation
    :return: enhanced dataset
    """
    feature_list = ['FEATURE_2', 'FEATURE_3', 'FEATURE_4', 'FEATURE_7', 'FEATURE_11', 'FEATURE_12', 'FEATURE_15']
    X = X[feature_list]

    return X


def do_all():
    
    global train
    global test
    global final_test
    global regression_models
    global selector_models
    global critical_models
    
    # data
    global train
    global test
    global final_test
    

    # RUL CAP
    # cap the RUL, early life samples are not distinguishable
    # in
    #    rul_cap: 0=Auto, any other value: cap for the RULs, if you pass "45%" (a string), 45% of max RUL will be used
    # out:
    #    rul_cap_used: Used, if rul_cap is a percentage or Auto
    if RUL_CAP != 0:
        train, cap_used = cap_rul(train, RUL_CAP, RUL_CAP_DROP_ABOVE)
        log("capped RUL to {}".format(cap_used))
    else:
        cap_used = False

    # CLUSTER
    # in
    #     X,y
    # out
    #     clusters
    if USE_CLUSTERS_FROM_DATASET > 0:
        log("using cluster given from csv...")
        n_clusters = len(np.unique(train.cluster))
    else:
        n_clusters = 1  # there is at least one
        

    # RUL
    # in
    #    X, y
    # out
    #    RUL
    n_critical = 2
    selector_models = np.ones([n_clusters], dtype=object)# dim 1: clusters, dim 2: criticality
    critical_models = np.ones([n_clusters,n_critical - 1], dtype=object)
    regression_models = np.ones([n_clusters,n_critical], dtype=object)  # dim 1: clusters, dim 2: criticality
                                                # cluster_models[2,1] 
                                                #        is model for cluster 3 (0 would be 1), 
                                                #        criticality high (1)
    for cluster_idx in range(0, n_clusters):
        log("training model for cluster #{}".format(cluster_idx + 1))

        base_filename = DATASET + "_RUL_" + str(FEATURE_SELECT_MANUAL) + "_" + str(RUL_CAP).replace("%", "p") +                        "_" + str(cluster_idx) + "_" + str(RUL_ADD_DERIVED)
        indices = train.cluster == cluster_idx
        X_sub = train.X[np.array(indices)]
        y_sub = train.rul[indices]

        if RUL_ADD_DERIVED > 0:
            log("adding derivatives to dataset")
            X_sub = add_derivatives(X_sub)

        # FEATURE FILTER
        # in
        #    X, y
        #    n_features: 0=Auto, using random, any other value: discrete target value
        # out
        #    n_features_used: Features acutally used, will match n_features except for n_features = 0
        #    X
        if isinstance(X_sub, pd.DataFrame):
            X_sub = X_sub.as_matrix()

        if FEATURE_SELECT_MANUAL > 0:  # -1: like wang, 0: no FS, >0: use poly
                
            pf = PolyFitter()
            pf.fit(X_sub, y_sub, num = FEATURE_SELECT_MANUAL, show_plots=False)
            weights = pf._weights
            selector_models[cluster_idx] = pf
            X_sub = pf.select(X_sub) 

            if False:
                log("feature weights:")
                for i, w in enumerate(weights):
                    print("\t\t\tfeature{0}: {1:.3f}".format(i+1, w))
        else:
            selector_models = None
            
        if ADD_HISTORY > 0:
            log("adding history to dataset")
            X_sub = add_history(X_sub, ADD_HISTORY, mode = ADD_HISTORY_MODE)
                
        if is_percentage(CRIT_SEPARATE_RUL):
            rul_to_use = percent_to_num(CRIT_SEPARATE_RUL) * np.max(y_sub)
            y_sub_crit = (y_sub < rul_to_use)+0 # binarize sub to criticality
            
            log("training criticality for cluster={}, criticality={}".format(cluster_idx, 0))
            mdl = classification_fit(X=X_sub, y=y_sub_crit, 
                                     model_filename=base_filename + "_CRIT_MDL", 
                                     reload_if_existing=RUL_RELOAD_IF_EXISTING,
                                     modeltype="RF", 
                                     cv_measure="recall")
            critical_models[cluster_idx, 0] = mdl
            
            # DEBUG HERE 
            if False:
                criticality = mdl.predict_proba(X_sub)[:,1] # probability of class 1, see crit_models[0].best_estimator_.classes_
                prec, rec, thresholds = precision_recall_curve(y_sub_crit, criticality)
                f1 = [2*(precision*recall)/(precision+recall) for (precision, recall) in zip(prec, rec)]
                optimum_threshold = thresholds[np.argmax(f1)]
                criticality = (criticality > optimum_threshold) + 0
            criticality = mdl.predict(X_sub)
            
            xs = list()
            for critical_sub in range(0,n_critical):
                indices = criticality == critical_sub
                xs.append(y_sub[indices])
            if False:
                plt.figure()
                plt.hist(xs, bins=50, stacked=True, normed=True)
                plt.title("Distincition of critical groups (train)")
                plt.show(block=False)

            # train a RUL model for every critical group
            for critical in range(0, n_critical):
                # get the indices matching the criticality
                indices = criticality == critical
                
                # select samples matching the current criticality
                X_sub_crit = X_sub[indices]
                y_sub_crit = y_sub[indices]
                
                log("training RUL for cluster={}, criticality={}".format(cluster_idx, critical))
                log("samples={}, features={}, y={}".format(X_sub.shape[0], X_sub.shape[1], len(y_sub)))
                mdl = regression_fit(X=X_sub_crit, y=y_sub_crit, 
                                     model_filename=base_filename + "_" + str(critical) + "_MDL", 
                                     reload_if_existing=RUL_RELOAD_IF_EXISTING,
                                     modeltype=RUL_MODELTYPE,
                                     scoring=RUL_SCORING)
                regression_models[cluster_idx, critical] = mdl
        else:
            log("samples={}, features={}, y={}".format(X_sub.shape[0], X_sub.shape[1], len(y_sub)))
            mdl = regression_fit(X=X_sub, y=y_sub, 
                                 model_filename=base_filename + "_MDL", 
                                 reload_if_existing=RUL_RELOAD_IF_EXISTING,
                                 modeltype=RUL_MODELTYPE,
                                 scoring=RUL_SCORING)
            regression_models[cluster_idx, 0] = mdl
            critical_models = None

    log("done training.")
    
    return regression_models, selector_models, critical_models


# In[ ]:

if False:
    DATASET = "nasa-turbofan"  # "nasa-phm" # "nasa-turbofan"
    RUL_CAP = 0 # "50%"
    RUL_CAP_DROP_ABOVE = True  # True
    FEATURE_SELECT_MANUAL = 0
    USE_CLUSTERS_FROM_DATASET = 1  #1 (enabled) or 0 (disabled)
    RUL_MODELTYPE = "RF"  # RF"
    RUL_SCORING = "default" # "default", "phm"
    RUL_ADD_DERIVED = 0 # this doesnt work... 1(enabled), 0(disabled)
    RUL_RELOAD_IF_EXISTING = False  # False # True (enabled), False (disabled)
    CRIT_SEPARATE_RUL = 0 # "25%"
    ADD_HISTORY = 5 # if 0, no history is added, if > 0, the last N steps will be added
    ADD_HISTORY_MODE = "copy" # "add_zero" or "copy"
    regression_models, selector_models, critical_models = do_all()
    frc, test_phm, test_phm_all, test_rmse = eval(test, regression_models, selector_models, critical_models)
    log("results: test_phm={}, test_phm_all={}, test_rmse={}".format(test_phm, test_phm_all, test_rmse), force=True)


# In[ ]:

columns = [
    "RUL_CAP",
    "RUL_CAP_DROP_ABOVE",
    "FEATURE_SELECT_MANUAL", # -1: manual, 0: no FS, >0: use poly
    "USE_CLUSTERS_FROM_DATASET",
    "RUL_MODELTYPE",
    "RUL_ADD_DERIVED",
    "CRIT_SEPARATE_RUL",
    "ADD_HISTORY",
    "ADD_HISTORY_MODE"]


# In[ ]:

# what are the different levels that should be evaluated?
def get_evalgrid(nasa = False):
    """
    nasa = True will enable the custom, wang et. al feature selection
    """
    i = 0
    evalgrid = []
    for rc in [0, "30%", "50%"]:  # RUL_CAP
        for rcda in [True, False]:  # RUL_CAP_DROP_ABOVE
            for fsm in ([-1, 0, 5, 10] if nasa else [0, 5, 10, 25, 50]):  # FEATURE_SELECT_MANUAL
                for ucfd in [1]:  # USE_CLUSTERS_FROM_DATASET
                    for rm in ["RF"]:  # RUL_MODELTYPE
                        for rad in [0, 1]:  # RUL_ADD_DERIVED
                            for csr in [0, "30%", "50%"]:  # CRIT_SEPARATE_RUL (0: disabled, percentage else)
                                for ah in ([0, 2, 5, 15] if nasa else [0, 1, 5]):  # ADD_HISTORY
                                    for ahm in ["add_zero", "copy"]:  # ADD_HISTORY_MODE

                                        # check if combination makes sense
                                        if rc == 0 and rcda == True:
                                            # only evaluate one value of RUL_CAP_DROP_ABOVE when RUL_CAP is disabled.
                                            # Which RUL_CAP_DROP_ABOVE that really is does not matter in this case.
                                            # in this case, will skipp all RUL_CAP_DROP_ABOVE = True.
                                            continue

                                        if percent_to_num(csr) > percent_to_num(rc):
                                            continue # doesn't make sense, since RUL values were cut off
                                            
                                        if ah == 0 and ahm == "copy":
                                            # only evaluate one value of ADD_HISTORY_MODE if ADD_HISTORY is disabled.
                                            # Which ADD_HISTORY_MODE that really is does not matter in this case.
                                            # in this case, will skipp all ADD_HISTORY_MODE = copy.
                                            continue


                                        """
                                        row = pd.DataFrame()

                                        row.set_value(0, "FEATURE_SELECT_MANUAL", sm)
                                        row.set_value(0, "RUL_CAP", rc)
                                        row.set_value(0, "RUL_CAP_DROP_ABOVE", rcda)
                                        row.set_value(0, "FEATURE_SELECT_MANUAL", fsm)
                                        row.set_value(0, "USE_CLUSTERS_FROM_DATASET", ucfd)
                                        row.set_value(0, "RUL_MODELTYPE", rm)
                                        row.set_value(0, "RUL_ADD_DERIVED", rad)
                                        row.set_value(0, "CRIT_SEPARATE_RUL", csr)
                                        row.set_value(0, "ADD_HISTORY", ah)
                                        row.set_value(0, "ADD_HISTORY_MODE", ahm)

                                        evalgrid[i] = row.iloc[0]
                                        """

                                        row = []
                                        row.append(rc)
                                        row.append(rcda)
                                        row.append(fsm)
                                        row.append(ucfd)
                                        row.append(rm)
                                        row.append(rad)
                                        row.append(csr)
                                        row.append(ah)
                                        row.append(ahm)

                                        evalgrid.append(row)

                                        #log("done {}".format(i))
                                        #i += 1

    evalgrid = pd.DataFrame(evalgrid, columns=columns)
    return evalgrid


# In[ ]:

# what are the different levels that should be evaluated?
def get_evalgrid_naive(nasa = False):
    """
    nasa = True will enable the custom, wang et. al feature selection
    """
    i = 0
    evalgrid = []
    for rc in [0]:  # RUL_CAP
        for rcda in [False]:  # RUL_CAP_DROP_ABOVE
            for fsm in ([-1] if nasa else [0]):  # FEATURE_SELECT_MANUAL
                for ucfd in [1]:  # USE_CLUSTERS_FROM_DATASET
                    for rm in ["RF"]:  # RUL_MODELTYPE
                        for rad in [0]:  # RUL_ADD_DERIVED
                            for csr in [0]:  # CRIT_SEPARATE_RUL (0: disabled, percentage else)
                                for ah in [0]:  # ADD_HISTORY
                                    for ahm in ["add_zero"]:  # ADD_HISTORY_MODE

                                        # check if combination makes sense
                                        if rc == 0 and rcda == True:
                                            # only evaluate one value of RUL_CAP_DROP_ABOVE when RUL_CAP is disabled.
                                            # Which RUL_CAP_DROP_ABOVE that really is does not matter in this case.
                                            # in this case, will skipp all RUL_CAP_DROP_ABOVE = True.
                                            continue

                                        if percent_to_num(csr) > percent_to_num(rc):
                                            continue # doesn't make sense, since RUL values were cut off

                                        if ah == 0 and ahm == "copy":
                                            # only evaluate one value of ADD_HISTORY_MODE if ADD_HISTORY is disabled.
                                            # Which ADD_HISTORY_MODE that really is does not matter in this case.
                                            # in this case, will skipp all ADD_HISTORY_MODE = copy.
                                            continue


                                        """
                                        row = pd.DataFrame()

                                        row.set_value(0, "FEATURE_SELECT_MANUAL", sm)
                                        row.set_value(0, "RUL_CAP", rc)
                                        row.set_value(0, "RUL_CAP_DROP_ABOVE", rcda)
                                        row.set_value(0, "FEATURE_SELECT_MANUAL", fsm)
                                        row.set_value(0, "USE_CLUSTERS_FROM_DATASET", ucfd)
                                        row.set_value(0, "RUL_MODELTYPE", rm)
                                        row.set_value(0, "RUL_ADD_DERIVED", rad)
                                        row.set_value(0, "CRIT_SEPARATE_RUL", csr)
                                        row.set_value(0, "ADD_HISTORY", ah)
                                        row.set_value(0, "ADD_HISTORY_MODE", ahm)

                                        evalgrid[i] = row.iloc[0]
                                        """

                                        row = []
                                        row.append(rc)
                                        row.append(rcda)
                                        row.append(fsm)
                                        row.append(ucfd)
                                        row.append(rm)
                                        row.append(rad)
                                        row.append(csr)
                                        row.append(ah)
                                        row.append(ahm)

                                        evalgrid.append(row)

                                        #log("done {}".format(i))
                                        #i += 1

    evalgrid = pd.DataFrame(evalgrid, columns=columns)
    return evalgrid


# In[ ]:

# what are the different levels that should be evaluated?
def get_evalgrid_old(nasa = False):
    """
    nasa = True will enable the custom, wang et. al feature selection
    """
    i = 0
    evalgrid = []
    rc_disabled_done_once = False
    for rc in [0, "30%", "50%"]:  # RUL_CAP
        for rcda in [True, False]:  # RUL_CAP_DROP_ABOVE
            for fsm in ([-1, 0, 5, 10] if nasa else [0, 5, 10, 25, 50]):  # FEATURE_SELECT_MANUAL
                for ucfd in [1]:  # USE_CLUSTERS_FROM_DATASET
                    for rm in ["RF", "SVR"]:  # RUL_MODELTYPE
                        for rad in [0, 1]:  # RUL_ADD_DERIVED
                            for csr in [0, "30%", "50%"]:  # CRIT_SEPARATE_RUL (0: disabled, percentage else)
                                no_history_done_once = False
                                for ah in ([0, 2, 5, 15] if nasa else [0, 1, 5]):  # ADD_HISTORY
                                    for ahm in ["add_zero", "copy"]:  # ADD_HISTORY_MODE

                                        # check if combination makes sense
                                        if rc_disabled_done_once and rc == 0:
                                            # evaluating different RUL_CAP_DROP_ABOVE doesn't make sense if rc is zero
                                            continue
                                        if rc == 0:
                                            rc_disabled_done_once = True

                                        if percent_to_num(csr) > percent_to_num(rc):
                                            continue # doesn't make sense, since RUL values were cut off
                                            
                                        if no_history_done_once and ah == 0:
                                            continue
                                        else:
                                            no_history_done_once = True


                                        """
                                        row = pd.DataFrame()

                                        row.set_value(0, "FEATURE_SELECT_MANUAL", sm)
                                        row.set_value(0, "RUL_CAP", rc)
                                        row.set_value(0, "RUL_CAP_DROP_ABOVE", rcda)
                                        row.set_value(0, "FEATURE_SELECT_MANUAL", fsm)
                                        row.set_value(0, "USE_CLUSTERS_FROM_DATASET", ucfd)
                                        row.set_value(0, "RUL_MODELTYPE", rm)
                                        row.set_value(0, "RUL_ADD_DERIVED", rad)
                                        row.set_value(0, "CRIT_SEPARATE_RUL", csr)
                                        row.set_value(0, "ADD_HISTORY", ah)
                                        row.set_value(0, "ADD_HISTORY_MODE", ahm)

                                        evalgrid[i] = row.iloc[0]
                                        """

                                        row = []
                                        row.append(rc)
                                        row.append(rcda)
                                        row.append(fsm)
                                        row.append(ucfd)
                                        row.append(rm)
                                        row.append(rad)
                                        row.append(csr)
                                        row.append(ah)
                                        row.append(ahm)

                                        evalgrid.append(row)

                                        #log("done {}".format(i))
                                        #i += 1

    evalgrid = pd.DataFrame(evalgrid, columns=columns)
    return evalgrid


# In[ ]:


def get_evalgrid_optimized(nasa=False):
    """
    nasa = True will enable the custom, wang et. al feature selection
    """
    i = 0
    evalgrid = []
    for rc in ["50%"]:  # RUL_CAP
        for rcda in [False]:  # RUL_CAP_DROP_ABOVE
            for fsm in ([-1, 0, 5, 10] if nasa else [0, 5, 10, 15]):  # FEATURE_SELECT_MANUAL
                for ucfd in [1]:  # USE_CLUSTERS_FROM_DATASET
                    for rm in ["RF"]:  # RUL_MODELTYPE
                        for rad in [0, 1]:  # RUL_ADD_DERIVED
                            for csr in [0]:  # CRIT_SEPARATE_RUL (0: disabled, percentage else)
                                for ah in ([1, 2] if nasa else [1, 2]):  # ADD_HISTORY
                                    for ahm in ["add_zero", "copy"]:  # ADD_HISTORY_MODE

                                        # check if combination makes sense
                                        if rc == 0 and rcda == True:
                                            # only evaluate one value of RUL_CAP_DROP_ABOVE when RUL_CAP is disabled.
                                            # Which RUL_CAP_DROP_ABOVE that really is does not matter in this case.
                                            # in this case, will skipp all RUL_CAP_DROP_ABOVE = True.
                                            continue

                                        if percent_to_num(csr) > percent_to_num(rc):
                                            continue  # doesn't make sense, since RUL values were cut off

                                        if ah == 0 and ahm == "copy":
                                            # only evaluate one value of ADD_HISTORY_MODE if ADD_HISTORY is disabled.
                                            # Which ADD_HISTORY_MODE that really is does not matter in this case.
                                            # in this case, will skipp all ADD_HISTORY_MODE = copy.
                                            continue

                                        """
                                        row = pd.DataFrame()

                                        row.set_value(0, "FEATURE_SELECT_MANUAL", sm)
                                        row.set_value(0, "RUL_CAP", rc)
                                        row.set_value(0, "RUL_CAP_DROP_ABOVE", rcda)
                                        row.set_value(0, "FEATURE_SELECT_MANUAL", fsm)
                                        row.set_value(0, "USE_CLUSTERS_FROM_DATASET", ucfd)
                                        row.set_value(0, "RUL_MODELTYPE", rm)
                                        row.set_value(0, "RUL_ADD_DERIVED", rad)
                                        row.set_value(0, "CRIT_SEPARATE_RUL", csr)
                                        row.set_value(0, "ADD_HISTORY", ah)
                                        row.set_value(0, "ADD_HISTORY_MODE", ahm)

                                        evalgrid[i] = row.iloc[0]
                                        """

                                        row = []
                                        row.append(rc)
                                        row.append(rcda)
                                        row.append(fsm)
                                        row.append(ucfd)
                                        row.append(rm)
                                        row.append(rad)
                                        row.append(csr)
                                        row.append(ah)
                                        row.append(ahm)

                                        evalgrid.append(row)

                                        # log("done {}".format(i))
                                        # i += 1

    evalgrid = pd.DataFrame(evalgrid, columns=columns)
    return evalgrid


# In[ ]:

def get_evalgrid_sub(nasa = False):
    eg = get_evalgrid(nasa = nasa)
    return eg[eg["RUL_CAP"] == 0]


# In[ ]:

import pandas as pd
get_evalgrid().shape


# In[ ]:

def get_indexes_to_redo(path="simulation_results/bernhard_results.csv"):
    df_missing = pd.read_csv(path)
    df_missing.insert(df_missing.shape[1], 'INCREMENT', range(0, len(df_missing)))
    df_missing.insert(df_missing.shape[1], 'DELTA', df_missing["META_BERNHARD_TRAIN_POINT"]-df_missing["INCREMENT"])
    df_missing.insert(df_missing.shape[1], 'DELTA_CHANGE', df_missing["DELTA"]-df_missing.shift(1)["DELTA"])
    df_missing.insert(df_missing.shape[1], 'DELTA_CHANGE_SHIFTED', df_missing.shift(-1)["DELTA_CHANGE"])
    df_missing = df_missing[df_missing["DELTA_CHANGE_SHIFTED"] > 0]

    indexes_to_redo = []
    for last_sucess_index, n_errors in zip(df_missing["META_BERNHARD_TRAIN_POINT"], df_missing["DELTA_CHANGE_SHIFTED"]):
        print("{} was the last successfull, after that, {} points failed.".format(last_sucess_index, n_errors))

        n_failed_left = n_errors
        last_err_index = last_sucess_index + 1
        while n_failed_left > 0:
            indexes_to_redo.append(last_err_index)
            last_err_index = last_err_index + 1
            n_failed_left = n_failed_left - 1
            
    return indexes_to_redo

def get_evalgrid_debug():
    eg = get_evalgrid_old(nasa=True)
    return eg.iloc[get_indexes_to_redo()]


# In[ ]:

# seems not to work with includes
import time
import datetime
def log(text, force = False):
    if not SUPRESS_LOG or (SUPRESS_LOG and force):
        print(time.strftime('%Y.%m.%d, %H:%M:%S') + ': ' + text)


# In[ ]:

import uuid
import sys
import traceback

last_DATASET = ""
last_FEATURE_SELECT_MANUAL = ""

train = None
test = None
final_test = None

train_orig = None
test_orig = None
final_test_orig = None

def get_data():
    
    global train_orig
    global test_orig
    global final_test_orig
    global DATASET
    global FEATURE_SELECT_MANUAL
    global last_FEATURE_SELECT_MANUAL
    global last_DATASET
    
    # Input: X, y, OBJECTID, RUL for train_orig
    if DATASET == "nasa-turbofan":
        train_orig, test_orig, final_test_orig = DataSource.load_turbofan()
    elif DATASET == "nasa-phm":
        train_orig, test_orig, final_test_orig = DataSource.load_phm()
    elif DATASET == "bmw-rex":
        train_orig, test_orig = DataSource.load_bmw_rex()
    elif DATASET == "bmw-cell":
        train_orig, test_orig = DataSource.load_bmw_cells()
    elif DATASET == "weather":
        train_orig, test_orig = DataSource.load_weather()

    if FEATURE_SELECT_MANUAL == -1:
        train_orig.X = feature_selection_wang(train_orig.X)
        test_orig.X = feature_selection_wang(test_orig.X)

    scalers = dict()
    train_orig.X_scaled = train_orig.X
    for cluster in train_orig.cluster.unique():
        idxs = cluster == train_orig.cluster
        X_cluster = train_orig.X[idxs]
        scaler = StandardScaler().fit(X_cluster)
        scalers[cluster] = scaler
        train_orig.X[idxs] = scaler.transform(train_orig.X[idxs])

    test_orig.X_scaled = test_orig.X
    for cluster in test_orig.cluster.unique():
        idxs = cluster == test_orig.cluster
        scaler = scalers[cluster]
        test_orig.X[idxs] = scaler.transform(test_orig.X[idxs])
        
    if FEATURE_SELECT_MANUAL == -1:
        train_orig.X = feature_selection_wang(train_orig.X)
        test_orig.X = feature_selection_wang(test_orig.X)
    
    last_DATASET = DATASET
    last_FEATURE_SELECT_MANUAL = FEATURE_SELECT_MANUAL

import copy


def eval_single(index=None, to_file=False):
    global TRAIN_ID
    global ALGORITHM_NAME

    global last_FEATURE_SELECT_MANUAL
    global last_DATASET
    global DATASET
    global FEATURE_SELECT_MANUAL
    global RUL_SCORING

    global train_orig
    global test_orig
    global final_test_orig
    global train
    global test
    global final_test

    if index is None:
        index = 0

    RUL_SCORING = "default"

    # get new train ID
    TRAIN_ID = str(uuid.uuid4())

    try:
        # get data
        if last_DATASET != DATASET or last_FEATURE_SELECT_MANUAL != FEATURE_SELECT_MANUAL:
            log("getting data...", force=True)
            get_data()
        train = copy.deepcopy(train_orig)
        test = copy.deepcopy(test_orig)
        final_test = copy.deepcopy(final_test_orig)

        # train
        log("starting training...", force=True)
        ts = ping()
        regression_models, selector_models, critical_models = do_all()
        ms_training = pong(ts)
        log("starting training testing...", force=True)
        ts = ping()
        frc, train_phm, train_phm_all, train_rmse = eval(test, regression_models, selector_models, critical_models,
                                                         train_id=TRAIN_ID, to_file=to_file)
        ms_training_test = pong(ts)

        try:
            log("starting testing...", force=True)
            ts = ping()
            frc, test_phm, test_phm_all, test_rmse = eval(test, regression_models, selector_models, critical_models,
                                                          train_id=TRAIN_ID)
            ms_test = pong(ts)
            log("results: test_phm={}, test_phm_all={}, test_rmse={}".format(test_phm, test_phm_all, test_rmse),
                force=True)

            rs.log({
                    
                "SCORE_RMSE_TRAIN": train_rmse,
                "SCORE_PHM_TRAIN": train_phm,
                "TRAINING_TIME_MS_SOLELY_TRAINING": ms_training,
                "TRAINING_TIME_MS_SOLELY_EVAL": ms_training_test,
                "DATASET_NAME": DATASET,
                "DATASET_SUBTYPE": "test",
                "ALGORITHM_NAME": ALGORITHM_NAME,
                "TRAINING_TIME_MS": ms_training + ms_training_test,
                "TESTING_TIME_MS": ms_test,
                "SCORE_RMSE": test_rmse,
                "SCORE_PHM": test_phm,
                "SCORE_PHM_ALL": test_phm_all,
                "USE_CLUSTERS_FROM_DATASET": USE_CLUSTERS_FROM_DATASET,
                # und deine eigenen, Reihenfolge wäre wegen key-value aber wurscht :)
                "HYPER_BERNHARD_RUL_CAP" : RUL_CAP,
                "HYPER_BERNHARD_RUL_CAP_DROP_ABOVE" : RUL_CAP_DROP_ABOVE,
                "HYPER_BERNHARD_FEATURE_SELECT_MANUAL" : FEATURE_SELECT_MANUAL,
                "HYPER_BERNHARD_RUL_MODELTYPE" : RUL_MODELTYPE,
                "HYPER_BERNHARD_RUL_ADD_DERIVED" : RUL_ADD_DERIVED,
                "HYPER_BERNHARD_CRIT_SEPARATE_RUL" : CRIT_SEPARATE_RUL,
                "HYPER_BERNHARD_ADD_HISTORY" : ADD_HISTORY,
                "HYPER_BERNHARD_ADD_HISTORY_MODE" : ADD_HISTORY_MODE,
                "META_BERNHARD_TRAIN_ID": TRAIN_ID,
                "META_BERNHARD_TRAIN_POINT": index
            })

            log("... took {}s (last_point={})".format((ms_training + ms_test)/1000, index), force = True)
        except Exception as e:
            print("Something went terribly wrong while training:\n"+str(sys.exc_info()[0]))    
            print("Printing only the traceback above the current stack frame")
            print("".join(traceback.format_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])))
            ms_test = -1
    except Exception as e:
        print("Something went terribly wrong while testing:\n"+str(sys.exc_info()[0]))       
        print("Printing only the traceback above the current stack frame")
        print("".join(traceback.format_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])))
        ms_test = -1

def get_completed_indexes(dataset_name):
    current_results = pd.read_csv(rs.get_filename())
    current_results = current_results[current_results["DATASET_NAME"] == dataset_name]
    completed_idxs = np.unique(list(current_results["META_BERNHARD_TRAIN_POINT"]))

    return completed_idxs

def eval_evalgrid(evalgrid, dataset_name, skip_until=0, skip_completed = True):
    log("evaluating a total of {} combinations.".format(len(evalgrid)), force = True)
    
    global DATASET
    global RUL_CAP
    global RUL_CAP_DROP_ABOVE
    global FEATURE_SELECT_MANUAL
    global USE_CLUSTERS_FROM_DATASET
    global RUL_MODELTYPE
    global RUL_ADD_DERIVED
    global CRIT_SEPARATE_RUL
    global ADD_HISTORY
    global ADD_HISTORY_MODE
    global RUL_SCORING
    global TRAIN_ID
    
    RUL_SCORING = "default"

    SUPRESS_LOG = True


    for index, point in evalgrid.iterrows():
        # get new train ID
        TRAIN_ID = str(uuid.uuid4())

        if skip_until > index:
            log("skipping point #{}".format(index), force=True)
            continue
        if skip_completed:
            completed_idxs = get_completed_indexes(dataset_name)
            if index in completed_idxs:
                log("skipping point #{} (already completed)".format(index), force=True)
                continue

        DATASET = dataset_name
        RUL_CAP = point["RUL_CAP"]
        RUL_CAP_DROP_ABOVE = point["RUL_CAP_DROP_ABOVE"]
        FEATURE_SELECT_MANUAL = point["FEATURE_SELECT_MANUAL"]
        USE_CLUSTERS_FROM_DATASET = point["USE_CLUSTERS_FROM_DATASET"]
        RUL_MODELTYPE = point["RUL_MODELTYPE"]
        RUL_ADD_DERIVED = point["RUL_ADD_DERIVED"]
        CRIT_SEPARATE_RUL = point["CRIT_SEPARATE_RUL"]
        ADD_HISTORY = point["ADD_HISTORY"]
        ADD_HISTORY_MODE = point["ADD_HISTORY_MODE"]

        log("evaluating: {}/{}\n".format(index,len(evalgrid))+
            "\t\t\tRUL_CAP={}\n".format(RUL_CAP) +
            "\t\t\tRUL_CAP_DROP_ABOVE={}\n".format(RUL_CAP_DROP_ABOVE) +
            "\t\t\tFEATURE_SELECT_MANUAL={}\n".format(FEATURE_SELECT_MANUAL) +
            "\t\t\tUSE_CLUSTERS_FROM_DATASET={}\n".format(USE_CLUSTERS_FROM_DATASET) +
            "\t\t\tRUL_MODELTYPE={}\n".format(RUL_MODELTYPE) +
            "\t\t\tRUL_ADD_DERIVED={}\n".format(RUL_ADD_DERIVED) +
            "\t\t\tCRIT_SEPARATE_RUL={}\n".format(CRIT_SEPARATE_RUL) +
            "\t\t\tADD_HISTORY={}\n".format(ADD_HISTORY) +
            "\t\t\tADD_HISTORY_MODE={}".format(ADD_HISTORY_MODE), force = True)


        eval_single(index=index)


# In[ ]:

RUL_RELOAD_IF_EXISTING = False  # False # True (enabled), False (disabled)


ALGORITHM_NAME = "BRR"

base_file_name = "brr_phd"
rs = ResultLogger(approach_name = base_file_name, additional_header_fields = custom_fields)


if __name__ == '__main__':
    eval_evalgrid(get_evalgrid_optimized(nasa=False), "bmw-rex")
if __name__ == '__main__':
    eval_evalgrid(get_evalgrid_optimized(nasa=False), "bmw-cell")
if __name__ == '__main__':
    eval_evalgrid(get_evalgrid(nasa=False), "weather")
if __name__ == '__main__':
    eval_evalgrid(get_evalgrid(nasa=True), "nasa-turbofan")


ALGORITHM_NAME = "naive"
rs = ResultLogger(approach_name = "naive_phd", additional_header_fields = custom_fields)
if __name__ == '__main__':
    eval_evalgrid(get_evalgrid_naive(nasa=False), "bmw-rex")
if __name__ == '__main__':
    eval_evalgrid(get_evalgrid_naive(nasa=False), "bmw-cell")
if __name__ == '__main__':
    eval_evalgrid(get_evalgrid_naive(nasa=True), "nasa-turbofan")
if __name__ == '__main__':
    eval_evalgrid(get_evalgrid_naive(nasa=False), "weather")

