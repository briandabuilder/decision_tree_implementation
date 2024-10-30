import numpy as np

def compute_confusion_matrix(actual, predictions):
    assert(predictions.shape[0] == actual.shape[0])
    return [[np.sum((actual == 0) & (predictions == 0)), np.sum((actual == 0) & (predictions == 1))],
            [np.sum((actual == 1) & (predictions == 0)), np.sum((actual == 1) & (predictions == 1))]]

def compute_accuracy(actual, predictions):
    assert(predictions.shape[0] == actual.shape[0])
    conf_matrix = compute_confusion_matrix(actual, predictions)
    return (conf_matrix[1][1] + conf_matrix[0][0]) / (conf_matrix[1][1] + conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0])


def compute_precision_and_recall(actual, predictions):
    assert(predictions.shape[0] == actual.shape[0])
    conf_matrix = compute_confusion_matrix(actual, predictions)
    precision = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1]) if np.nonzero(conf_matrix[1][1] + conf_matrix[0][1]) else np.nan
    recall = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]) if np.nonzero(conf_matrix[1][1] + conf_matrix[1][0]) else np.nan
    return precision, recall

def compute_f1_measure(actual, predictions):
    assert(predictions.shape[0] == actual.shape[0])
    precision, recall = compute_precision_and_recall(actual, predictions)
    return 2 * (precision * recall) / (precision + recall) if precision != np.nan and recall != np.nan else np.nan

    
