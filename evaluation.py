import numpy as np

from sklearn.metrics import confusion_matrix, recall_score, f1_score, \
                            precision_score, accuracy_score, matthews_corrcoef
                            
import DutchDraw as dutchdraw

BINARY_MEASURE_NAMES = ["tp", "tn", "fp", "fn", "recall", "prec", 
                        "acc", "f1", "mcc"]

def binary_evaluation_measures(y_true, y_pred, labels = [0,1]):
    '''
    We assume here that `1' represents a malicious session, while `0' is benign.
    
    Example:
    y_true = np.random.choice([0,1], size = 100)
    y_pred = np.random.choice([0,1], size = 100)
    binary_evaluation_measures(y_true, y_pred)
    '''
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()

    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()
        
    if len(y_true) != len(y_pred):
        raise ValueError("y_pred and y_true should have equal length")
    
    if np.unique(np.array(y_true)) not in np.array([0, 1]):
        raise ValueError("y_true should only contain zeros and ones.")

    if np.unique(np.array(y_pred)) not in np.array([0, 1]):
        raise ValueError("y_pred should only contain zeros and ones.")
        
    # Compute base measures
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = labels).ravel()
    
    # Compute performance metrics
    recall = recall_score(y_true, y_pred, zero_division = 0, labels = labels)
    precision = precision_score(y_true, y_pred, zero_division = 0, labels = labels)
    f1 = f1_score(y_true, y_pred, zero_division = 0, labels = labels)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return [tp, tn, fp, fn, recall, precision, accuracy, f1, mcc]


def print_binary_results(y_true, y_pred, print_only_mcc = False):
    scores = binary_evaluation_measures(y_true, y_pred)
    scores_dict = dict(zip(BINARY_MEASURE_NAMES, scores))
    
    scores_dict["acc_baseline"] = dutchdraw.optimized_baseline_statistics(y_true, "ACC")["Max Expected Value"]
    scores_dict["f1_baseline"] = dutchdraw.optimized_baseline_statistics(y_true, "FBETA", beta = 1)["Max Expected Value"]
    
    if print_only_mcc:
        print(scores_dict["mcc"])
    else:
        print(scores_dict)
    