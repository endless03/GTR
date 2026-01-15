import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from datetime import timedelta


def evaluating_change_point(true, prediction, metric='nab', numenta_time=None):
    """
    Evaluate change point detection performance

    Parameters:
    -----------
    true : list or pd.Series
        Ground truth labels
    prediction : list or pd.Series
        Predicted labels
    metric : str
        Evaluation metric: 'nab', 'binary', 'average_delay'
    numenta_time : str
        Time window for NAB evaluation
    """

    def binary(true, prediction):
        """Binary classification metrics"""

        def single_binary(true, prediction):
            true_ = true == 1
            prediction_ = prediction == 1
            TP = (true_ & prediction_).sum()
            TN = (~true_ & ~prediction_).sum()
            FP = (~true_ & prediction_).sum()
            FN = (true_ & ~prediction_).sum()
            return TP, TN, FP, FN

        if not isinstance(true, list):
            TP, TN, FP, FN = single_binary(true, prediction)
        else:
            TP, TN, FP, FN = 0, 0, 0, 0
            for i in range(len(true)):
                TP_, TN_, FP_, FN_ = single_binary(true[i], prediction[i])
                TP += TP_;
                TN += TN_;
                FP += FP_;
                FN += FN_

        f1 = round(TP / (TP + (FN + FP) / 2), 4) if (TP + (FN + FP) / 2) > 0 else 0
        print(f'False Alarm Rate {round(FP / (FP + TN) * 100, 2) if FP + TN > 0 else 0}%')
        print(f'Missing Alarm Rate {round(FN / (FN + TP) * 100, 2) if FN + TP > 0 else 0}%')
        print(f'F1 metric {f1}')
        return f1

    def average_delay(detecting_boundaries, prediction):
        """Calculate detection delay"""

        def single_average_delay(detecting_boundaries, prediction):
            missing = 0
            detect_history = []

            for couple in detecting_boundaries:
                t1, t2 = couple[0], couple[1]
                index = prediction[t1:t2][prediction[t1:t2] == 1].index
                if len(index) == 0:
                    missing += 1
                else:
                    detect_history.append(index[0] - t1)

            return missing, detect_history

        if not isinstance(prediction, list):
            missing, detect_history = single_average_delay(detecting_boundaries, prediction)
        else:
            missing, detect_history = 0, []
            for i in range(len(prediction)):
                missing_, detect_history_ = single_average_delay(
                    detecting_boundaries[i], prediction[i])
                missing += missing_
                detect_history += detect_history_

        add = np.mean(detect_history) if detect_history else 0
        print('Average delay', add)
        print('Number of missed CPs =', missing)
        return add

    def evaluate_nab(detecting_boundaries, prediction, table_of_coef=None):
        """NAB scoring"""

        def single_evaluate_nab(detecting_boundaries, prediction,
                                table_of_coef=None, name_of_dataset=None):
            if table_of_coef is None:
                table_of_coef = pd.DataFrame([
                    [1.0, -0.11, 1.0, -1.0],
                    [1.0, -0.22, 1.0, -1.0],
                    [1.0, -0.11, 1.0, -2.0]
                ])
                table_of_coef.index = ['Standard', 'LowFP', 'LowFN']
                table_of_coef.index.name = "Metric"
                table_of_coef.columns = ['A_tp', 'A_fp', 'A_tn', 'A_fn']

            alist = detecting_boundaries.copy()
            prediction = prediction.copy()
            Scores, Scores_perfect, Scores_null = [], [], []

            for profile in ['Standard', 'LowFP', 'LowFN']:
                A_tp = table_of_coef['A_tp'][profile]
                A_fp = table_of_coef['A_fp'][profile]
                A_fn = table_of_coef['A_fn'][profile]

                def sigm_scale(y, A_tp, A_fp, window=1):
                    return (A_tp - A_fp) * (1 / (1 + np.exp(5 * y / window))) + A_fp

                score = 0
                if len(alist) > 0:
                    score += prediction[:alist[0][0]].sum() * A_fp
                else:
                    score += prediction.sum() * A_fp

                for i in range(len(alist)):
                    if i <= len(alist) - 2:
                        win_space = prediction[alist[i][0]:alist[i + 1][0]].copy()
                    else:
                        win_space = prediction[alist[i][0]:].copy()

                    win_fault = prediction[alist[i][0]:alist[i][1]]
                    slow_width = int(len(win_fault) / 4)

                    if len(win_fault) + slow_width >= len(win_space):
                        print(f'Intersection of windows too wide for dataset {name_of_dataset}')
                        win_fault_slow = win_fault.copy()
                    else:
                        win_fault_slow = win_space[:len(win_fault) + slow_width]

                    win_fp = win_space[-len(win_fault_slow):]

                    if win_fault_slow.sum() == 0:
                        score += A_fn
                    else:
                        tr = pd.Series(win_fault_slow.values,
                                       index=range(-len(win_fault), len(win_fault_slow) - len(win_fault)))
                        tr_values = tr[tr == 1].index[0]
                        tr_score = sigm_scale(tr_values, A_tp, A_fp, slow_width)
                        score += tr_score

                    score += win_fp.sum() * A_fp

                Scores.append(score)
                Scores_perfect.append(len(alist) * A_tp)
                Scores_null.append(len(alist) * A_fn)

            return np.array([np.array(Scores), np.array(Scores_null),
                             np.array(Scores_perfect)])

        if not isinstance(prediction, list):
            matrix = single_evaluate_nab(detecting_boundaries, prediction,
                                         table_of_coef=table_of_coef)
        else:
            matrix = np.zeros((3, 3))
            for i in range(len(prediction)):
                matrix_ = single_evaluate_nab(detecting_boundaries[i],
                                              prediction[i],
                                              table_of_coef=table_of_coef,
                                              name_of_dataset=i)
                matrix += matrix_

        results = {}
        desc = ['Standard', 'LowFP', 'LowFN']
        for t, profile_name in enumerate(desc):
            numerator = matrix[0, t] - matrix[1, t]
            denominator = matrix[2, t] - matrix[1, t]
            if denominator <= 0:
                score_ratio = 0.0
            else:
                score_ratio = max(0, numerator / denominator)
            results[profile_name] = round(100 * score_ratio, 2)
            print(f'{profile_name}: {results[profile_name]}')

        return results

    # Prepare ground truth labels
    if not isinstance(true, list):
        true_items = [true[true == 1].index]
    else:
        true_items = [true[i][true[i] == 1].index for i in range(len(true))]

    # Create detection boundaries
    if metric != 'binary':
        def create_detecting_boundaries(true, numenta_time, true_items):
            detecting_boundaries = []
            td = pd.Timedelta(numenta_time) if numenta_time is not None else \
                pd.Timedelta((true.index[-1] - true.index[0]) / len(true_items[0]))

            for val in true_items[0]:
                detecting_boundaries.append([val, val + td])
            return detecting_boundaries

        if not isinstance(true, list):
            detecting_boundaries = create_detecting_boundaries(true, numenta_time, true_items)
        else:
            detecting_boundaries = []
            for i in range(len(true)):
                boundaries = create_detecting_boundaries(true[i], numenta_time,
                                                         [true_items[i]])
                detecting_boundaries.append(boundaries)

    # Select evaluation metric
    if metric == 'nab':
        return evaluate_nab(detecting_boundaries, prediction)
    elif metric == 'average_delay':
        return average_delay(detecting_boundaries, prediction)
    elif metric == 'binary':
        return binary(true, prediction)