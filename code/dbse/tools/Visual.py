from dbse.tools.Singleton import Singleton
import matplotlib.pyplot as plt
import numpy as np

class Visual(Singleton):

    def __init__(self):
        pass

    @staticmethod
    def plot_hist(x, max_x=None):

        plt.hist(x, normed=True, bins=30)
        plt.ylabel('Probability')
        if max_x is not None:
            axes = plt.gca()
            axes.set_xlim([0, max_x])
            #axes.set_ylim([ymin, ymax])
        plt.show(block=False)

    @staticmethod
    def plot_scatter(x, y, label_x = "x", label_y="y"):
        plt.scatter(x, y)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.show(block=False)

    @staticmethod
    def plot_normal(plot_args, x_label, y_label):
        plt.plot(plot_args[0])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show(block=False)

