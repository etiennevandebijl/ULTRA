import numpy as np

from models import MODEL_DICT

def activelearning(X, y, model_al, random_state_al, strategy, L, U, A, p, q, weighted_training = True):

    if not strategy in ["Random", "Certainty", "Uncertainty"]:
        raise ValueError("Strategy is only computed for Random and (Un)certainty")

    if q > len(U):
        raise ValueError("Cannot sample more than unknown instances")

    if strategy == "Random":
        np.random.seed(0)
        selected = np.random.choice(U, size = q, replace = False)
    else:

        clf = MODEL_DICT[model_al]

        # Set random state if possible
        if 'random_state' in clf.get_params():
            clf.set_params(random_state=random_state_al)

        # Fit learner
        if weighted_training:
            clf.fit(X[L] @ A, y[L], sample_weight = p[L])
        else:
            clf.fit(X[L] @ A, y[L])

        if hasattr(clf, 'decision_function'):
            y_pred = clf.decision_function(X[U] @ A)

            # 0 is the decision bound here
            # Low values --> Uncertain
            # High values -- > Certain
            z_score = np.abs(y_pred)
        else:
            y_pred = clf.predict_proba(X[U] @ A)[:,1]

            # Confidence (Kan nog ff nadenken over deze 0.5)
            # Low values --> Uncertain
            # High values --> Certain
            z_score = 2 * np.abs(y_pred - 0.5)

        # Default [Uncertain, middle, Certainty]
        U_sorted = U[np.argsort(z_score)]

        # Sort on certainty or uncertainty
        if strategy == "Certainty":
            U_sorted = U_sorted[::-1]

        selected = np.array(U_sorted[:q])

    return selected
