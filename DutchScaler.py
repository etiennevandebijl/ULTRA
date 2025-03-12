import numpy as np
import DutchDraw as DutchDraw

# %%

measure_dictionary = DutchDraw.measure_dictionary

def indicator_score(y_true, measure, alpha, thetaopts, rho = 0, beta = 1):
    """
    This function determines the Dutch Scaler Performance Indicator score
    if you want to manually select all parameters. It derives the base measures
    and afterwards computes, using those base measures, the performance metric 
    indicator score. 

    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.

        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.
        
        alpha (float): Parameter for the Dutch Scaler.
        
        thetaopts (list): List of all `theta_star` values that maximize the expected value of the DDB.
        
        rho (float): Default is 0. Parameter for the Dutch Oracle.

        beta (float): Default is 1. Parameter for the F-beta score.

    Returns:
    --------
        float: The indicator score of the given measure evaluated with the Dutch Scaler

    Raises:
    --------
        ValueError
            If `measure` is not in `select_all_names_except([''])`.
        ValueError
            If 'measure' is not in supported metric list.
        ValueError
            If `y_true` does not only contain zeros and ones.
        ValueErrpr
            If 'rho' is above the accepted barrier.

    See also:
    --------
        select_all_names_except

    Example:
    --------
        Open ticket: TO DO
    """
    measure = measure.upper()
    
    if measure not in DutchDraw.select_all_names_except(['']):
        raise ValueError("This measure name is not recognized.")
    
    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "G2", "TS"]:
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("The DSPI is not supported for this measure")
        
    # convert np.array to list
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()    
    
    if np.unique(np.array(y_true)) not in np.array([0, 1]):
        raise ValueError("y_true should only contain zeros and ones.")
    
    if rho > valid_rho_values(y_true, measure, beta):
        raise ValueError("Rho is selected to high.")
        
    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
    
    t = thetaopts[0]
    TP = alpha * P * (1 - rho - t) + P * t 
    TN = alpha * N * (t - rho) + N * (1 - t) 
    FN = alpha * P * (rho - 1 + t) + P * (1 - t)
    FP = alpha * N * (rho - t) + N * t

    if measure in measure_dictionary['PPV']:
        return TP / (TP + FP)

    if measure in measure_dictionary['NPV']:
        return TN / (TN + FN)

    if measure in measure_dictionary['ACC']:
        return (TP + TN) / M

    if measure in measure_dictionary['BACC']:
        TPR = TP / P
        TNR = TN / N
        return (TPR + TNR) / 2

    if measure in measure_dictionary['FBETA']:
        beta_squared = beta ** 2
        return (1 + beta_squared) * TP / (((1 + beta_squared) * TP) + (beta_squared * FN) + FP)

    if measure in measure_dictionary['MCC']:
        return (TP * TN - FP * FN)/(np.sqrt((TP + FP) * (TN + FN) * P * N))

    if measure in measure_dictionary['J']:
        TPR = TP / P
        TNR = TN / N
        return TPR + TNR - 1

    if measure in measure_dictionary['MK']:
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        return PPV + NPV - 1

    if measure in measure_dictionary['KAPPA']:
        P_o = (TP + TN) / M
        P_yes = ((TP + FP) / M) * (P / M)
        P_no = ((TN + FN) / M) * (N / M)
        P_e = P_yes + P_no
        return (P_o - P_e) / (1 - P_e)

    if measure in measure_dictionary['FM']:
        TPR = TP / P
        PPV = TP / (TP + FP)
        return np.sqrt(TPR * PPV)

    if measure in measure_dictionary['G2']:
        TPR = TP / P
        TNR = TN / N
        return np.sqrt(TPR * TNR)

    if measure in measure_dictionary['TS']:
        return TP / (TP + FN + FP)

def optimized_indicator(y_true, measure, alpha, rho = 0, beta = 1):
    """
    This function determines the Dutch Scaler Performance Indicator score
    without the need to search for the optimal thetaopts. It derives the base measures
    and afterwards computes, using those base measures, the performance metric 
    indicator score. 

    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.

        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.
        
        alpha (float): Parameter for the Dutch Scaler.
        
        rho (float): Default is 0. Parameter for the Dutch Oracle.

        beta (float): Default is 1. Parameter for the F-beta score.

    Returns:
    --------
        float: The indicator score of the given measure evaluated with the Dutch Scaler

    Raises:
    --------
        ValueError
            If `measure` is not in `select_all_names_except([''])`.
        ValueError
            If 'measure' is not in supported metric list.
        ValueError
            If `y_true` does not only contain zeros and ones.
        ValueError
            If 'rho' is above the accepted barrier.

    See also:
    --------
        select_all_names_except

    Example:
    --------
        TO DO
        
    Open tickets: Derive optimal thetaopts for G2, MCC, MK. 
    """
    measure = measure.upper()

    if measure not in DutchDraw.select_all_names_except(['']):
        raise ValueError("This measure name is not recognized.")
    
    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "J", "KAPPA", "FM", "TS"]:
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("No closed-form expression without thetaopts is supported for the given measure")
        
    # convert np.array to list
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()    
    
    if np.unique(np.array(y_true)) not in np.array([0, 1]):
        raise ValueError("y_true should only contain zeros and ones.")
    
    if rho > valid_rho_values(y_true, measure, beta):
        raise ValueError("Rho is selected to high.")
    
    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
 
    if measure in measure_dictionary['PPV']:
        up = alpha * P * (M - 1 - M * rho) + P
        down = alpha * M * (rho * (N - P) + P - 1) + M
        return up/down

    if measure in measure_dictionary['NPV']:
        up = alpha * N * (M - 1 - M * rho) + N
        down = alpha * M * (rho * (P - N) + N - 1) + M
        return up/down

    if measure in measure_dictionary['FBETA']:
        up = (1 + beta * beta) * P * (1 - alpha * rho)
        down = alpha * (rho * (N - P) -N) + M + P * beta * beta
        return up / down

    if measure in measure_dictionary['ACC']:
        return  (alpha * (min(P , N) - M * rho) + max(P, N)) * (1 / M)

    if measure in measure_dictionary['FM']:
        up = np.sqrt(P) * (1 - alpha * rho)
        down = np.sqrt(alpha * (rho * (N - P) -N) + M)
        return up / down

    if measure in measure_dictionary['J']:
        return alpha * (1 - 2 * rho)

    if measure in measure_dictionary['BACC']:
        return alpha * (1 - 2 * rho) * 0.5  + 0.5

    if measure in measure_dictionary['KAPPA']:
        up = 2 * P * N * alpha * (1 - 2 * rho)
        down = M * min(P,N) + alpha * ( rho * (N - P)**2 - (min(N,P)**2) + P * N)
        return up / down
    
    if measure in measure_dictionary['TS']:
        if (P > 1) or (alpha * N * (1 - 2 * rho) <= 1):
            up = P * (1 - alpha * rho)
            down = M - N * alpha * (1 - rho)
            return up/down
        else:
            up = alpha * (N - M * rho) + 1
            down = M + N - alpha * N * (1 - M * rho)
            return up/down

def valid_rho_values(y_true, measure, beta = 1):
    """
    This function returns the rho bound to fulfill the requirement that the 
    performance metric under the Dutch Scaler is strictly increasing in alpha.

    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.

        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.
        
        beta (float): Default is 1. Parameter for the F-beta score.

    Returns:
    --------
        float: The rho value that should NOT be crossed or touched. 

    Raises:
    --------
        ValueError
            If `measure` is not in `select_all_names_except([''])`.
        ValueError
            If 'measure' is not in supported metric list.
        ValueError
            If `y_true` does not only contain zeros and ones.

    See also:
    --------
        select_all_names_except

    Example:
    --------
        TO DO
        
    Open tickets: Need to check if TS is correct. 
    """
   
    measure = measure.upper()
    
    
    if measure not in DutchDraw.select_all_names_except(['']):
        raise ValueError("This measure name is not recognized.")
    
    # convert np.array to list
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()    
    
    if np.unique(np.array(y_true)) not in np.array([0, 1]):
        raise ValueError("y_true should only contain zeros and ones.")
        
    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "G2", "TS"]:
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("No valid rho could be determined for the requested measure")
        
    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
    
    if measure in measure_dictionary['PPV']:
        return 0.5
    
    if measure in measure_dictionary['NPV']:
        return 0.5
    
    if measure in measure_dictionary['ACC']:
        return min(N,P)/M

    if measure in measure_dictionary['BACC']:
        return 0.5
    
    if measure in measure_dictionary['FBETA']:
        return N / (2 * N + P * beta * beta)
        
    if measure in measure_dictionary['MCC']:
        return 0.5
    
    if measure in measure_dictionary['J']:
        return 0.5
    
    if measure in measure_dictionary['MK']:
        return 0.5
    
    if measure in measure_dictionary['KAPPA']:
        return 0.5
    
    if measure in measure_dictionary['FM']:
        return N / (3 * N + P)

    if measure in measure_dictionary['G2']:
        return 0.5
    
    if measure in measure_dictionary['TS']:
        return N / (N + M)
    
def lower_bound(y_true, measure, beta = 1):
    """
    This function returns the lower value of the Dutch Scaler Performance Indicator
    when the theta is optimized. Selecting other theta values might lead to other values.
    An identical value should be obtained with optimized_indicator(y_true, measure, 0.0).
    
    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.
        
        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.
        
        beta (float): Default is 1. Parameter for the F-beta score.
    
    Returns:
    --------
        float: The lower bound of the performance indicator score.
        
    Raises:
    --------
        ValueError
            If `measure` is not in `select_all_names_except([''])`.
        ValueError
            If 'measure' is not in supported metric list.
        ValueError
            If `y_true` does not only contain zeros and ones.
        
    See also:
    --------
        select_all_names_except
    
    Example:
    --------
        TO DO
        
        Open Ticket: The G2 is problematic.
    """
    measure = measure.upper()
    
    if measure not in DutchDraw.select_all_names_except(['']):
        raise ValueError("This measure name is not recognized.")
    
    # convert np.array to list
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()    
    
    if np.unique(np.array(y_true)) not in np.array([0, 1]):
        raise ValueError("y_true should only contain zeros and ones.")
        
    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "TS"]:
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("The DSPI is not supported for this measure")
    
    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
     
    if measure in measure_dictionary['PPV']:
        return P / M
    
    if measure in measure_dictionary['NPV']:
        return N / M
    
    if measure in measure_dictionary['ACC']:
        return max(P,N)/M
    
    if measure in measure_dictionary['BACC']:
        return 0.5
    
    if measure in measure_dictionary['FBETA']:
        return (1 + beta * beta) * P / (M + P * beta * beta)
        
    if measure in measure_dictionary['MCC']:
        return 0
    
    if measure in measure_dictionary['J']:
        return 0
    
    if measure in measure_dictionary['MK']:
        return 0
    
    if measure in measure_dictionary['KAPPA']:
        return 0
    
    if measure in measure_dictionary['FM']:
        return np.sqrt(P / M)
    
    if measure in measure_dictionary['TS']:
        return P / M
    
def upper_bound(y_true, measure, rho = 0, beta = 1):
    """
    This function returns the upper bound of the Dutch Scaler Performance Indicator. 
    An identical value should be obtained with optimized_indicator(y_true, measure, 1.0).
    
    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.
        
        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.
        
        rho (float): Default is 0. Parameter for the Dutch Oracle.
        
        beta (float): Default is 1. Parameter for the F-beta score.
    
    Returns:
    --------
        float: The upper bound of the performance indicator score.
    
    Raises:
    --------
        ValueError
            If `measure` is not in `select_all_names_except([''])`.
        ValueError
            If 'measure' is not in supported metric list.
        ValueError
            If `y_true` does not only contain zeros and ones.
        ValueError
            If 'rho' is above the accepted barrier.

    See also:
    --------
        select_all_names_except
    
    Example:
    --------
    """
    measure = measure.upper()
    
    if measure not in DutchDraw.select_all_names_except(['']):
        raise ValueError("This measure name is not recognized.")
    
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()    
    
    if np.unique(np.array(y_true)) not in np.array([0, 1]):
        raise ValueError("y_true should only contain zeros and ones.")
        
    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "TS", "G2"]:
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("The DSPI is not supported for this measure")
    
    if rho > valid_rho_values(y_true, measure, beta):
        raise ValueError("Rho is selected to high.")
        
    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
    
    if measure in measure_dictionary['PPV']:
        return P * (1 - rho) / ( rho * (N - P) + P)
    
    if measure in measure_dictionary['NPV']:
        return N * (1 - rho) / (rho * (P - N) + N)
    
    if measure in measure_dictionary['ACC']:
        return (1 - rho)
    
    if measure in measure_dictionary['BACC']:
        return (1 - rho)
    
    if measure in measure_dictionary['FBETA']:
        up = (1 + beta * beta) * P * (1 - rho)
        down = rho * (N - P) + P * (1 + beta * beta)
        return up/down
        
    if measure in measure_dictionary['MCC']:
        up = np.sqrt(P * N) * (1 - 2 * rho)
        down = np.sqrt( (rho * (N - P) + P)*( rho * (P - N) + N))
        return up/down
    
    if measure in measure_dictionary['J']:
        return 1 - 2 * rho
    
    if measure in measure_dictionary['MK']:
        up = N * P * (1 - 2 * rho)
        down = N * P - rho**2 * (P - N)**2 + rho * (P - N)**2 
        return up / down

    if measure in measure_dictionary['KAPPA']:
        up = 2 * P * N * (1 - 2 * rho)
        down = rho * (N - P)**2 + 2 * P * N
        return up/down
    
    if measure in measure_dictionary['FM']:
        up = np.sqrt(P) * (1 - rho)
        down = np.sqrt(P * (1 - rho) + N * rho)
        return up/down

    if measure in measure_dictionary['G2']:
        return 1 - rho
    
    if measure in measure_dictionary['TS']:
        up = P * (1 - rho)
        down = rho * N + P
        return up/down

def select_rho(y_true, measure, max_score, beta = 1):
    """
    This function returns the rho to satisfy the requirement that alpha
    cannot be bigger than 1. 
    
    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.
        
        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.
        
        max_score (float): This is the maximal obtainable performance metric score.
        
        beta (float): Default is 1. Parameter for the F-beta score.
    
    Returns:
    --------
        float: The rho satisfying the requirement. 
    
    Raises:
    --------
        ValueError
            If `measure` is not in `select_all_names_except([''])`.
        ValueError
            If 'measure' is not in supported metric list.
        ValueError
            If `y_true` does not only contain zeros and ones.

    See also:
    --------
        select_all_names_except
    
    Example:
    --------
        TO DO
    
        Open Ticket: The MCC is problamatic.
    """
    
    measure = measure.upper()
    
    if measure not in DutchDraw.select_all_names_except(['']):
        raise ValueError("This measure name is not recognized.")

    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()    
    
    if np.unique(np.array(y_true)) not in np.array([0, 1]):
        raise ValueError("y_true should only contain zeros and ones.")
    
    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "G2", "TS"]:
        if measure == "MCC":
            raise ValueError("The DSPI is not YET supported for this measure")
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("The DSPI is not supported for this measure")
        
    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
    
    if measure in measure_dictionary['ACC']:
        return (1 - max_score)

    if measure in measure_dictionary['BACC']:
        return (1 - max_score)
    
    if measure in measure_dictionary['J']:
        return 0.5 - 0.5 * max_score

    if measure in measure_dictionary['G2']:
        return 1 - max_score

    if measure in measure_dictionary['PPV']:
        return P * (1 - max_score) / ( max_score * (N - P) + P)
    
    if measure in measure_dictionary['NPV']:
        return N * (1 - max_score) / (max_score * (P - N) + N)
    
    if measure in measure_dictionary['FBETA']:
        up = (1 + beta * beta) * P * (1 - max_score)
        down = max_score * (N - P) + P * (1 + beta * beta)
        return up/down
        
    if measure in measure_dictionary['TS']:
        up = P * (1 - max_score)
        down = max_score * N + P
        return up/down
    
    if measure in measure_dictionary['KAPPA']:
        up = 2 * P * N * (1 - max_score)
        down = max_score * (N - P)**2 + 4 * P * N
        return up/down    
    
    if measure in measure_dictionary['FM']:
        term = (P - N) / P
        up = term * max_score * max_score + max_score * np.sqrt(term * term * max_score * max_score + 4 * (N / P))
        return 1 - up/2

    if measure in measure_dictionary['MK']:
        D = max_score * max_score * (P - N)**4 + 4 * N * P * max_score**2 * (P-N)**2 + 4 * N**2 * P**2
        up = 2 * N * P - np.sqrt(D)
        down = 2 * max_score * (P - N)**2
        return 0.5 + up / down
    
    if measure in measure_dictionary['MCC']:
        return 0
    
def optimized_indicator_inverted(y_true, measure, score, rho = 0, beta = 1):
    """
    This function gives the Dutch Scaler indicator alpha for a corresponding measure
    score. 
    
    Args:
    --------
        y_true (list or numpy.ndarray): 1-dimensional boolean list/numpy.ndarray containing the true labels.
        
        measure (string): Measure name, see `select_all_names_except([''])` for possible measure names.

        score (float): Realized performance metric score. 
        
        rho (float): Default is 0. Parameter for the Dutch Oracle.
        
        beta (float): Default is 1. Parameter for the F-beta score.
    
    Returns:
    --------
        float: The obtained alpha.

        float: The corresponding thetaopts leading to the alpha.
    
    Raises:
    --------
        ValueError
            If `measure` is not in `select_all_names_except([''])`.
        ValueError
            If 'measure' is not in supported metric list.
        ValueError
            If `y_true` does not only contain zeros and ones.
        ValueError
            If 'rho' is above the accepted barrier.
        ValueError
            If 'score' is above the accepted upper limit.

    See also:
    --------
        select_all_names_except
    
    Example:
    --------
        TO DO
    
    """    
    measure = measure.upper()

    if measure not in DutchDraw.select_all_names_except(['']):
        raise ValueError("This measure name is not recognized.")
    

    # Check if DSPI is supported for the corresponding metric
    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "G2", "TS"]:
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("The DSPI is not supported for this measure")

    # Check if the realized score outperforms the DDB
    baseline = DutchDraw.optimized_baseline_statistics(y_true, measure)
    if baseline['Max Expected Value'] == score:
        if measure != "G2":
            if measure != "KAPPA":
                return 0, baseline['Argmax Expected Value']
    if baseline['Max Expected Value'] > score:
        raise ValueError("De score must outperform the Dutch Draw.")

    if rho >= valid_rho_values(y_true, measure, beta):
        raise ValueError("Rho is selected to high to have increasing DSPI.")

    if score > upper_bound(y_true, measure, rho, beta):
        raise ValueError("The score is above the upper limit.")

    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)

    if measure in measure_dictionary['J']:
        alpha = score / (1 - 2 * rho)
        thetaopts = [i/M for i in range(0, M + 1)]
        return  alpha, thetaopts

    if measure in measure_dictionary['BACC']:
        alpha = (2 * score - 1) / (1 - 2 * rho)
        thetaopts = [i/M for i in range(0, M + 1)]
        return alpha, thetaopts

    if measure in measure_dictionary['PPV']:
        alpha = (M * score - P) / (M * score - P + P * M * (1 - score) + rho * M * (score * (P - N) - P))
        thetaopts = [1.0 / M]
        return alpha, thetaopts

    if measure in measure_dictionary['NPV']:
        alpha = ((N - score * M) ) / ((score * M * (N - 1) + N * (1 - M) + rho * M * (score * (P - N) + N)))
        thetaopts = [(M - 1) / M]
        return alpha, thetaopts

    if measure in measure_dictionary['FBETA']:
        alpha = (P * (1 + beta*beta) * (1 - score) - score * N) / (score * ( rho * (N - P) - N) + (1 + beta*beta) * P * rho)
        thetaopts = [1]
        return alpha, thetaopts

    if measure in measure_dictionary['ACC']:
        alpha = (M * score - max(P, N))  / (min(P, N) - M * rho)
        if P < N:
            thetaopts = [0]
        elif P > N:
            thetaopts = [1]
        else:
            thetaopts =  [i/M for i in range(0, M+1)]
        return alpha, thetaopts

    if measure in measure_dictionary['KAPPA']:
        if (P == M or P == 0):
            '''
            Open ticket: Should this not lead to alpha = 0? Below result is from article. Needs check.
            '''
            return 1, [ ( M - 1 ) / M ] 
        else:
            alpha = (score * M * min(P,N)) / (2 * N * P * (1 - 2 * rho) + score * ((min(P,N)**2) - N * P - rho * (P - N)**2)) 
            if P == N:
                thetaopts = [i/M for i in range(0, M + 1)]
            if P > N: 
                thetaopts = [1]
            if N > P:
                thetaopts = [0]   
            return alpha, thetaopts

    if measure in measure_dictionary['FM']:
        thetaopts = [1]
        if rho == 0:
            alpha = (M / N) - (P / (score * score * N))
        else:
            a = rho * rho * P
            b = - 2 * rho * P - score * score * (rho * (N - P) - N)
            c = P - M * score * score 
            '''
            Open ticket: Which solution of the following equations is valid?
            '''
            a_1 = (-b + np.sqrt(b * b - 4 * a * c) ) / (2 * a)
            a_2 = (-b - np.sqrt(b * b - 4 * a * c) ) / (2 * a)
            
            answer = False
            if a_1 >= 0 and a_1 <= 1.0001:
                alpha = a_1
                answer = True
            if a_2 >= 0 and a_2 <= 1.0001:
                alpha = a_1
                answer = True
            if answer == False:
                print(a_1)
                print(a_2)
                raise ValueError("FM PROBLEM")
                
        return alpha, thetaopts

    if measure in measure_dictionary['MK']:
        alpha = np.inf
        thetaopts = []
        for t in baseline['Argmax Expected Value']:

            term = rho * (N - P) + P - M * t
            if np.abs(term)<0.0000001:
                a_1 = (M * M * t * (1 - t) * score) / ((1 - 2 * rho) * P * N)
            else:
                a = term * term * score
                b = -1 * term * M * score * (1 - 2 * t) + N * P * (1 - 2 * rho)
                c = -1 * M * M * t * score * (1 - t)
                a_1 = (-b + np.sqrt(b * b - 4 * a * c) ) / (2 * a) #Checked 
            if a_1 > 1.00000002: 
                raise ValueError("Alpha bigger than 1 MK")
                
            if a_1 == alpha:
                thetaopts.append(t)
            if a_1 < alpha:
                alpha = a_1
                thetaopts = [t]
        return alpha, thetaopts

    if measure in measure_dictionary['TS']:
        if P > 1:
            alpha = (M * score - P) / (N * score * (1 - rho) - P * rho) 
            thetaopts = [1]
        else:
            if N > (1 / score):
                alpha = (score * (M + N) - 1) / (score * N - M * score * N * rho - M * rho + N)
                thetaopts = [1 / M]
            elif score == 1 / N:
                alpha = score * (1 / (1 - 2 * rho))
                thetaopts = [i/M for i in range(1, M + 1)]
            else:
                alpha = (M * score - P) / (N * score * (1 - rho) - P * rho) 
                thetaopts = [1]
        return alpha, thetaopts

    if measure in measure_dictionary['G2']:
        # As G2 is not linear in TP, it can happen that alpha is negative. 
        alpha = np.inf
        thetaopts = []
        for t in baseline['Argmax Expected Value']:
            if (1 - rho - t) * (t - rho) == 0:
                a_1 = (score * score  - t *(1 - t)) / (1 - rho - 2 * t + 2 * t* t)
            else:
                a = (1 - rho - t) * (t - rho)
                b = ((2 * t * t) - (2 * t) + 1 - rho)
                c =  t - ( (t * t) + (score * score))
                ''' 
                Open ticket: Proof dat a_1 is always in [0,1] and a_2 is not
                '''
                a_1 = (-b + np.sqrt(b * b - 4 * a * c) ) / (2 * a) 
                # a_2 = (-b - np.sqrt(b * b - 4 * a * c) ) / (2 * a) 
            if a_1 > 1: 
                raise ValueError("Alpha bigger than 1 G2")
                
            if a_1 == alpha:
                thetaopts.append(t)
            if a_1 < alpha:
                alpha = a_1
                thetaopts = [t]
        return alpha, thetaopts

    if measure in measure_dictionary['MCC']:
        alpha = np.inf
        thetaopts = []
        for t in baseline['Argmax Expected Value']:
            a = score * score * (P - M * t + rho * (N - P) ) * (P - M * t + rho * (N - P)) + (P * N) * (1 - 2 * rho)**2 
            b = -1 * score * score * M * (P - M * t + rho * (N - P)) * (1 - 2 * t)
            c = - M * M * t * (1 - t) * score * score    
            '''
            Open ticket: In article, it needs to be shown that this is smaller than 1.
            '''
            a_1 = (-b + np.sqrt(b * b - 4 * a * c) ) / (2 * a) #Checked in article
            if a_1 > 1.00000002: 
                print(a_1)
                raise ValueError("Alpha bigger than 1 MCC")
            if a_1 == alpha:
                thetaopts.append(t)
            if a_1 < alpha:
                alpha = a_1
                thetaopts = [t]
        return alpha, thetaopts



