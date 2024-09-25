from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

MODEL_DICT = {"DT": DecisionTreeClassifier(),
              "GNB": GaussianNB(),
              "RF": RandomForestClassifier(n_jobs = -1),
              "NN_BF": KNeighborsClassifier(n_neighbors = 1, algorithm = "brute", n_jobs=-1),
              "SVM": LinearSVC(loss="hinge"),
              "RF_CB": RandomForestClassifier(n_jobs = -1, class_weight = "balanced")
              }

