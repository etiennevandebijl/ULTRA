import numpy as np

from models import MODEL_DICT

def activelearning(X, y, model_name, rs_clf, strategy, L, U, A, p, q, use_weight = True):

    if strategy == "Random":
        np.random.seed(0)
        selected = np.random.choice(U, size = q, replace = False)
    else:
        
        clf = MODEL_DICT[model_name]

        # Set random state if possible
        if 'random_state' in clf.get_params():
            clf.set_params(random_state=rs_clf)
                
        # Fit learner
        if use_weight:
            clf.fit(X[L] @ A, y[L], sample_weight = p[L])
        else:
            clf.fit(X[L] @ A, y[L])
 
        if hasattr(clf, 'decision_function'):
            y_pred = clf.decision_function(X[U] @ A)

            z_score = np.abs(y_pred)
        else:
            y_pred = clf.predict_proba(X[U] @ A)[:,1]
    
            # Confidence
            z_score = 2 * np.abs(y_pred - 0.5)
            
        confidence = U[np.argsort(z_score)]

        # Sort on certainty or uncertainty
        if strategy == "Uncertainty":
            confidence = confidence[::-1]

        selected = np.array(confidence[:q])

    return selected


