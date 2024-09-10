from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

MODEL_DICT = {"DT": DecisionTreeClassifier(),
              "GNB": GaussianNB(),
              "RF": RandomForestClassifier(n_jobs = -1),
              "NN_BF": KNeighborsClassifier(n_neighbors = 1, algorithm = "brute", n_jobs=-1)
              }

