import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

def main():
    sklearn_iris = load_iris()
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,3), random_state=1)
    clf.fit(sklearn_iris.data, sklearn_iris.target)
    print(clf.predict([[6.3, 2.8, 5.1, 1.5]]))

if __name__ == "__main__":
    main()