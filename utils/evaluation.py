from sklearn.metrics import precision_score, recall_score

def weighted_f1_score(y_true, y_pred):
    """
    Calculate the weighted F1 score using the formula:
    :param y_true:
    :param y_pred:
    :return:

    The weighted F1 score is calculated as:
    F1 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    """
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_score = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f1_score
