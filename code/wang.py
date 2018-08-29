## header ##
import numpy as np

# common / individual libraries
from common.data import DataSource
from common.log import log, ping, pong, ResultLogger
from peter.additional_funs import Scoring
from peter.wang_train import WangTrain
from peter.wang_test import WangTest

# data set selection
dataset_list = ["nasa-turbofan", "weather", "bmw-rex", "bmw-cells", "nasa-phm"]
dataset_list = ["bmw-cells"]

hy_param_perc_score_min_phm = 99.705997648    # percentile score based on C_min = -300 of wang et al. (2008) paper
hy_param_perc_score_max_phm = 2.61226534257   # percentile score based on C_max = -5 of wang et al. (2008) paper
hy_param_max_cycles_score_phm = 1.06741573034 # percentage/100 based on own testing
hy_param_min_num_ruls_candi = 15 #30
hy_param_min_num_ruls_out_1 = 10 #25
hy_param_scaling_curve_fit = 1.1207865168539326

## main ##
for dataset in dataset_list:
    log("#################################### loading data ##################################", force=True)
    if dataset == "nasa-turbofan":
        dataset_subtype = "all"
        train, test, final_test = DataSource.load_turbofan()
    elif dataset == "nasa-phm":
        dataset_subtype = "all"
        train, test, final_test = DataSource.load_phm()
    elif dataset == "bmw-rex":
        dataset_subtype = "test"
        train, test = DataSource.load_bmw_rex()
    elif dataset == "bmw-cells":
        dataset_subtype = "test"
        train, test = DataSource.load_bmw_cells()
    elif dataset == "weather":
        dataset_subtype = "test"
        train, test = DataSource.load_weather()

    rs = ResultLogger(approach_name="peter", additional_header_fields=["HYPER_PETER_PARAM_MAX_CYCLES_SCORE_PHM",
                                                                       "HYPER_PETER_PARAM_MIN_NUM_RULS_CANDI",
                                                                       "HYPER_PETER_PARAM_MIN_NUM_RULS_OUT1",
                                                                       "HYPER_PETER_PARAM_PERC_SCORE_MIN_PHM",
                                                                       "HYPER_PETER_PARAM_PERC_SCORE_MAX_PHM",
                                                                       "HYPER_PETER_PARAM_SCALING_CURVE_FIT"])
    ## training stage ##
    log("################################### training stage #################################", force=True)

    if dataset == "nasa-turbofan" or dataset == "nasa-phm":
        C_max = -5
        C_min = -300
        max_cycles = 380
    elif dataset == "weather":
        C_max = -1  # np.percentile(train.rul, hy_param_perc_score_max_phm)
        C_min = -22.75  # based on own testing
        max_cycles = int(round(hy_param_max_cycles_score_phm * max(train.rul)))
    elif dataset == "bmw-rex":
        C_max = -450  # np.percentile(train.rul, hy_param_perc_score_max_phm)
        C_min = -4119  # np.percentile(train.rul, hy_param_perc_score_min_phm)
        max_cycles = int(round(hy_param_max_cycles_score_phm * max(train.rul)))
    else:
        C_max = -16767911.0#np.percentile(train.rul, hy_param_perc_score_max_phm)
        C_min = -np.percentile(train.rul, hy_param_perc_score_min_phm)
        max_cycles = int(round(hy_param_max_cycles_score_phm * max(train.rul)))

    train.start_time = ping()
    train = WangTrain.train(train, dataset, C_max, C_min, hy_param_scaling_curve_fit)
    train.time = pong(train.start_time)

    log("training time: {}s".format(train.time / 1000), force=True)
    ## end training stage ##

    ## ///////////////////////////////////////////////////////////////////////////////////////// ##

    ## testing stage ##
    log("#################################### testing stage #################################", force=True)
    test.start_time = ping()

    test = WangTest.predict(test,
                            train.Pa_pool,
                            train.Model_pool,
                            train.variance_models,
                            dataset_subtype,
                            dataset,
                            max_cycles,
                            hy_param_min_num_ruls_candi,
                            hy_param_min_num_ruls_out_1)

    test.time = pong(test.start_time)
    log("testing time: {}s".format(test.time / 1000), force=True)

    # scoring #
    log("#################################### scoring stage #################################", force=True)
    test.rmse, test.phm_score_final = Scoring.get_scores(test.rul, test.rul_pred, test.objectid)

    # logging #
    phm_score_to_use = test.phm_score_final
    if isinstance(test.phm_score_final, (np.ndarray)):
        phm_score_to_use = test.phm_score_final[0]

    rs.log({
        "DATASET_NAME": dataset,
        "DATASET_SUBTYPE": "test",
        "ALGORITHM_NAME": "wang_et_al",
        "TRAINING_TIME_MS": train.time,
        "TESTING_TIME_MS": test.time,
        "SCORE_RMSE": test.rmse,
        "SCORE_PHM": phm_score_to_use,
        "SCORE_PHM_ALL": "not possible in wang_et_al algorithm",
        "USE_CLUSTERS_FROM_DATASET": 1,
        # peters hyperparameters
        "HYPER_PETER_PARAM_MAX_CYCLES_SCORE_PHM": hy_param_max_cycles_score_phm,
        "HYPER_PETER_PARAM_MIN_NUM_RULS_CANDI": hy_param_min_num_ruls_candi,
        "HYPER_PETER_PARAM_MIN_NUM_RULS_OUT1": hy_param_min_num_ruls_out_1,
        "HYPER_PETER_PARAM_PERC_SCORE_MIN_PHM": hy_param_perc_score_min_phm,
        "HYPER_PETER_PARAM_PERC_SCORE_MAX_PHM": hy_param_perc_score_max_phm,
        "HYPER_PETER_PARAM_SCALING_CURVE_FIT": hy_param_scaling_curve_fit
    })

    if dataset == "nasa-turbofan" or dataset == "nasa-phm":
        log("################################# final testing stage ##############################", force=True)

        final_test.start_time = ping()
        final_test = WangTest.predict(final_test,
                                      train.Pa_pool,
                                      train.Model_pool,
                                      train.variance_models,
                                      dataset_subtype,
                                      dataset,
                                      max_cycles,
                                      hy_param_min_num_ruls_candi,
                                      hy_param_min_num_ruls_out_1)

        final_test.time = pong(final_test.start_time)

        log("testing time (final test): {}s".format(final_test.time / 1000), force=True)

        # scoring #
        log("#################################### scoring stage #################################", force=True)
        final_test.rmse, final_test.phm_score_final = Scoring.get_scores(final_test.rul,
                                                                         final_test.rul_pred,
                                                                         final_test.objectid)

        # logging #
        rs.log({
            "DATASET_NAME": dataset,
            "DATASET_SUBTYPE": "final_test",
            "ALGORITHM_NAME": "wang_et_al",
            "TRAINING_TIME_MS": train.time,
            "TESTING_TIME_MS": final_test.time,
            "SCORE_RMSE": final_test.rmse,
            "SCORE_PHM": final_test.phm_score_final,
            "SCORE_PHM_ALL": "not possible in wang_et_al algorithm",
            "USE_CLUSTERS_FROM_DATASET": 1,
            # peters hyperparameters
            "HYPER_PETER_PARAM_MAX_CYCLES_SCORE_PHM": hy_param_max_cycles_score_phm,
            "HYPER_PETER_PARAM_MIN_NUM_RULS_CANDI": hy_param_min_num_ruls_candi,
            "HYPER_PETER_PARAM_MIN_NUM_RULS_OUT1": hy_param_min_num_ruls_out_1,
            "HYPER_PETER_PARAM_PERC_SCORE_MIN_PHM": hy_param_perc_score_min_phm,
            "HYPER_PETER_PARAM_PERC_SCORE_MAX_PHM": hy_param_perc_score_max_phm,
            "HYPER_PETER_PARAM_SCALING_CURVE_FIT": hy_param_scaling_curve_fit
        })
