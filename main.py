import pandas as pd
import numpy as np
from mini_batch import *
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

minibatch_size = 30
hidden_layer_sizes = (100, 100, 100)
test_data = []

def main():
    iris_data = load_iris()

    # Sklearn MLP Classifier
    print('--------------------')
    print('Sklearn MLP Classifier')
    print('--------------------')

    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=1)
    classifier.fit(iris_data.data, iris_data.target)

    print('Iteration', classifier.n_iter_)

    count_sklearn = 0
    for i, x in enumerate(classifier.predict(iris_data.data).tolist()):
        if (x == iris_data.target[i]):
            count_sklearn += 1
    print("Sklearn Accuracy: ", count_sklearn/len(iris_data.target) * 100, "%")

    # MyMLP (Backpropagation) with mini-batch gradient descent
    print('--------------------')
    print('MyMLP (Backpropagation) with mini-batch gradient descent')
    print('--------------------')
    model = make_network(hidden_layer_sizes=hidden_layer_sizes)
    model = sgd(model, iris_data.data, iris_data.target, minibatch_size)

    true_count = 0
    for i, x in enumerate(iris_data.data):
        _, prob = forward(x, model)
        y = np.argmax(prob)
        if y == iris_data.target[i]:
            true_count += 1
    print('MyMLP Accuracy', true_count/len(iris_data.target) * 100, "%")

if __name__ == "__main__":
    main()