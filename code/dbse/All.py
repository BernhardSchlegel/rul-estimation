from dbse.pipeline.BaseScorer import BaseScorer
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

class All(BaseScorer):
    """
    Calculates the RMSE scores, e.g. for RUL
    """

    def __init__(self):
        """
        Initializes the BaseScorer using the given parameters
        """

    @staticmethod
    def score(labels: np.ndarray, predictions: np.ndarray):
        """
        does the scoring
        :param labels: actual label
        :param predictions: the model based preductions
        :return: the score as float
        """
        return sqrt(mean_squared_error(labels, predictions))

    @staticmethod
    def score_all(y_true, y_pred, bound_phm=True):

        rmse = []
        mape = []
        PHM_score_final = None
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) > 1:
            # RMSE
            rmse = All.score_rmse(y_pred, y_true)

            # MAPE, TODO: Replace
            y_true_np, y_pred = np.array(y_true), np.array(y_pred)
            mape = 0 # np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            # phm2008 scoring
            PHM_score_final = All.score_phm(bound_phm, y_pred, y_true)

        return [rmse, mape, PHM_score_final]

    @staticmethod
    def score_rmse(y_pred, y_true):
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        return rmse

    
    @staticmethod
    def score_phm(df, bound_phm = False, plot_heatmap_enabled=True):
    
        # groupby preserves the order of rows within each group, 
        # see http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
        #df = df.sort_values(["object_id", "RUL"], ascending=[True, False]).groupby('object_id').last().reset_index().sort_values(by="object_id")
        # THIS MUST BE ONE LINE - BUG?
        
        scores = []
        cnt = 0
        
        last = 0
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
                da_score = np.exp(-d / a1) - 1
                scores.append(np.exp(-d / a1) - 1)
                
            elif d >= 0:
                da_score = np.exp(d / a2) - 1
                scores.append(np.exp(d / a2) - 1)
            last = da_score
            cnt += 1
            print("PHM - " + str(da_score))
         
        print("scored {} elements".format(cnt))
        print("\nPhm Scores: "+str(sorted(scores)))
        print("\nDavon ueber 100: "+str(len([s for s in scores if s>100])))
        return sum(scores), last
'''    
    @staticmethod
    def eval(evalobject: DataStructure):
        final_result_combined = {'RUL': evalobject.rul, 
                                 'predicted_RUL': evalobject.y_pred, 
                                 'object_id': evalobject.objectid, 
                                 'TS': evalobject.ts}
        final_result_combined = pd.DataFrame(data=final_result_combined)
        test_phm = score_phm(final_result_combined)
        test_rmse = score_rmse(final_result_combined["RUL_REAL"], final_result_combined["RUL_PRED"])
        print("test: rmse={}, phm={}".format(test_rmse, test_phm))
        
        return final_result_combined, test_phm, test_rmse
'''
