import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from .log import log, ping, pong

"""
import like from common.score import score_rmse, output_phm_file, score_phm
"""


def plot_heatmap(predictions, real_values, filename=None):
    """
    Helper function to plot a heatmap of the given predictions over the 
    real values
    :param predictions: 
    :param real_values: 
    :return: 
    """
    heatmap, xedges, yedges = np.histogram2d(np.array(predictions), real_values, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.figure()
    plt.xlabel("predictions")
    plt.ylabel("real value")
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    if filename is not None:
        plt.savefig(filename)
    plt.show(block=False)
    if filename is not None:
        plt.close("all")

def score_rmse(y_true, y_pred):
    """
    Calculates the RMSE score
    :param y_true: true labels
    :param y_pred: predictions
    :return: the rmse
    """
    return mean_squared_error(y_true, y_pred) ** 0.5


def output_phm_file(df, filename = "phm_out.csv"):
    """
    Outputs the predictions of the given dataframe to a csv. This csv can be 
    submitted to the online evaluation system. Use dataset "phm test" to do 
    this.
    :param df: a dataframe holding at least OBJECTID, TS, RUL_REAL, and RUL_PRED columns
    """

    df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True]).groupby \
        ('OBJECTID').last().reset_index().sort_values(by="OBJECTID")

    df["RUL_PRED"].to_csv(filename, index=False)


def score_phm(df: pd.DataFrame, plot_heatmap_enabled=True, all=False, train_id=None):
    """
    Caculates the phm score for the given dataframe holding the predictions.

    :param df: a dataframe holding at least OBJECTID, TS, RUL_REAL, and RUL_PRED columns
    :param plot_heatmap_enabled: If true, a heatmap of prediction vs. true value will be plotted
    :return: the phm score
    """

    if not all:
        # groupby preserves the order of rows within each group,
        # see http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
        df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True]).groupby \
            ('OBJECTID').last().reset_index().sort_values(by="OBJECTID")

    if train_id is not None:
        df.to_csv("simulation_results/details/" + train_id + "_predictions" + ("_all" if all else "") + ".csv", index=False)

    scores = []
    cnt = 0  # to count the number of objects
    for pred, actual, objectid in zip(df["RUL_PRED"], df["RUL_REAL"], df["OBJECTID"]):

        log("comparing pred={}/rul={} for object {}".format(pred, actual, objectid))

        a1 = 13
        a2 = 10
        d = pred - actual

        if d < 0:
            scores.append(np.exp(-d / a1) - 1)
        elif d >= 0:
            scores.append(np.exp(d / a2) - 1)

        cnt += 1

    if plot_heatmap_enabled:  # lambda
        if train_id is not None:
            filename = "simulation_results/details/" + train_id + "_predictions" + ("_all" if all else "") + "_heatmap.png"
            plot_heatmap(df["RUL_REAL"], [round(x) for x in df["RUL_PRED"]],
                         filename=filename)
        else:
            plot_heatmap(df["RUL_REAL"], [round(x) for x in df["RUL_PRED"]])

    log("scored {} elements".format(cnt))
    return sum(scores)


def score_phm_all(df, plot_heatmap_enabled=True, train_id=None):
    """
    Caculates the phm score for the given dataframe holding the predictions for ALL samples (not 
    just the last sample from each object).

    :param df: a dataframe holding at least OBJECTID, TS, RUL_REAL, and RUL_PRED columns
    :param plot_heatmap_enabled: If true, a heatmap of prediction vs. true value will be plotted
    :return: the phm score
    """

    return score_phm(df, plot_heatmap_enabled=plot_heatmap_enabled, all=True, train_id=train_id)