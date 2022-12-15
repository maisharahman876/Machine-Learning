"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""
import numpy as np
def classification(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
 
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
 
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
 
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return TP,TN,FP,FN

def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    TP,TN,FP,FN = classification(y_true, y_pred)
    return (TP+TN)/(TP+TN+FP+FN)
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    TP,TN,FP,FN = classification(y_true, y_pred)
    if TP+FP == 0:
        return 0
    return TP/(TP+FP)


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    TP,TN,FP,FN = classification(y_true, y_pred)
    if TP+FN == 0:
        return 0
    return TP/(TP+FN)


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision+recall == 0:
        return 0
    return 2*precision*recall/(precision+recall)
