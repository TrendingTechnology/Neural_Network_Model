# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


"""

Created on 20.06.2021

@author: Nikita

    The module contains Metrics for evaluating the effectiveness of classification of neural network models.

"""


class Metrics:

    """

    Parameters
    ----------

    y_predict : List of values with dimension 'n'.
        True targets of binary classification.

    y_true : List of values with dimension 'n'.
        Estimated probabilities or output of a decision function.

    """

    def __init__(self, y_predict, y_true):
        self.y_predict = y_predict
        self.y_true = y_true

    def input_date_threshold(self, rounding_threshold=0.5):

        """

        The function is designed to assign 'y_predict' to a class (1 or 0), depending on the rounding_threshold value.

        Parameters
        ----------

        rounding_threshold : Threshold value.

        Returns
        -------

        y_predict : tf.tensor, int64
        "y_predict" which was predicted by the neural network model.

        y_predict_reverse : tf.tensor, int64
        Inverse tensor to the "y_predict".

        y_true : tf.tensor, int64
        Correct values.

        y_true_reverse : tf.tensor, int64
        Inverse tensor to the "y_true".

        """

        y_predict_new = []

        for item in self.y_predict:
            if item < rounding_threshold:
                y_predict_new.append(0)
            else:
                y_predict_new.append(1)

        y_predict = tf.convert_to_tensor(y_predict_new, dtype='int64')
        y_predict_reverse = 1 - y_predict
        y_true = tf.convert_to_tensor(self.y_true, dtype='int64')
        y_true_reverse = 1 - y_true

        return y_predict, y_predict_reverse, y_true, y_true_reverse

    def precision_metric(self, precision_threshold=0.5):

        """

        The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of true positives and ``fp``
        the number of false positives. The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative.

        The best value is 1 and the worst value is 0.

        Parameters
        ----------

        precision_threshold : Threshold value.

        """

        y_predict, y_predict_reverse, y_true, y_true_reverse = Metrics.input_date_threshold(self, precision_threshold)
        tp = sum(y_true * y_predict)
        fp = sum(y_true_reverse * y_predict)
        precision = tp / (tp + fp)

        return precision

    def recall_metric(self, recall_threshold=0.5):

        """

        The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of true positives and ``fn`` the number
        of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

        The best value is 1 and the worst value is 0.

        Parameters
        ----------

        recall_threshold : Threshold value.

        """

        y_predict, y_predict_reverse, y_true, y_true_reverse = Metrics.input_date_threshold(self, recall_threshold)
        tp = sum(y_true * y_predict)
        fn = sum(y_true * y_predict_reverse)
        recall = tp / (tp + fn)

        return recall

    def f1_score(self, recall_threshold=0.5, precision_threshold=0.5, beta=None):

        """

        The F score is the weighted harmonic mean of precision and recall.
        Here it is only computed as a batch-wise average, not globally. This is useful for multi-label classification,
        where input samples can be classified as sets of labels. By only using accuracy (precision) a model would
        achieve a perfect score by simply assigning every class to every input. In order to avoid this, a metric should
        penalize incorrect class assignments as well (recall).The F-beta score (ranged from 0.0 to 1.0) computes this,
        as a weighted mean of the proportion of correct class assignments vs. the proportion of incorrect
        class assignments.

        Parameters
        ----------

        recall_threshold : Threshold value.

        precision_threshold : Threshold value.

        beta : With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning correct classes becomes more
        important, and with beta > 1 the metric is instead weighted towards penalizing incorrect class assignments.

        """

        # If there are no true positives, fix the F score at 0 like sklearn.
        if sum(self.y_predict) == 0:
            return 0

        precision = Metrics.precision_metric(self, precision_threshold)
        recall = Metrics.recall_metric(self, recall_threshold)
        f1_score = 2 * (precision * recall) / (precision + recall)

        if beta is not None:

            if beta < 0:

                raise ValueError('The lowest choosable beta is zero (only precision).')

            else:

                bb = beta ** 2
                f1_score = (1 + bb) * (precision * recall) / ((bb * precision + recall) + recall)

        if f1_score is None:
            return 0

        return f1_score

    def curve_values(self, step: int and float):

        """

        The function is intended for calculating the following values:
            tp : true positives;
            fp : false positives;
            fn : false negatives;
            tn : true negatives.

        The function also calculates the True Positive Velocity (TPR) and False Positive Velocity (FPR) values for each
        threshold value with a threshold 'step' for the AUC-ROC curve.

        Parameters
        ----------

        step : Step of threshold values.

        """

        fpr = []
        tpr = []
        thresholds = []

        for threshold in np.arange(0, 1, step):

            tp = 0
            fp = 0
            fn = 0
            tn = 0

            for y_predict_item, y_true_item in zip(self.y_predict, self.y_true):

                if y_predict_item >= threshold:  # predicted == 1

                    if y_true_item == 1:
                        tp += 1
                    else:
                        fp += 1

                else:  # predicted == 0

                    if y_true_item == 1:
                        fn += 1
                    else:
                        tn += 1

            fpr.append(fp / (fp + tn))
            tpr.append(tp / (tp + fn))

            thresholds.append(threshold)

        return tpr, fpr, thresholds

    @staticmethod
    def auc_roc(tpr, fpr):

        """

        The function is used to calculate the area under the ROC curve.

        """

        area = np.trapz(tpr, fpr)

        const = 1
        if area < 0:
            const = -1

        square = const * area

        return square

    @staticmethod
    def get_threshold(fpr, tpr, threshold_values, threshold_fpr=1, threshold_tpr=0):

        """

        The function is used to get the threshold value. The threshold is calculated using the following formula:

                                       optimal = |TPR - (1-FPR)|, optimal -> min

        By default, the thresholds for FPR and TPR are not set, if necessary, you can set them by passing them to
        the arguments.

        Parameters
        ----------

        fpr : False Positive Rate values for each threshold value.

        tpr : True Positive Rate values for each threshold value.

        threshold_fpr : The threshold value for FPR.

        threshold_tpr : The threshold value for TPR.

        threshold_values : Threshold values with a certain step.

        """

        dict_threshold = {}

        for values_tpr, values_fpr, threshold in zip(tpr, fpr, threshold_values):

            optimal = abs(np.array(values_tpr) - (1 - np.array(values_fpr)))

            if threshold_tpr <= values_tpr < 1 and values_fpr < threshold_fpr:
                dict_threshold[optimal] = threshold

        try:

            return dict_threshold[min(dict_threshold.keys())]

        except ValueError:

            print('For these values TPR and FPR, there is no suitable threshold value, try changing them.')
