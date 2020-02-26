import pandas as pd
import numpy as np
from mini_batch import *
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

minibatch_size = 30
test_data = [[6.3, 2.8, 5.1, 1.5]]

def main():
    iris_data = load_iris()

    # Sklearn MLP Classifier
    print('--------------------')
    print('Sklearn MLP Classifier')
    print('--------------------')
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,1,2), random_state=1)
    classifier.fit(iris_data.data, iris_data.target)
    # print('iteration:', classifier.n_iter_)
    print(classifier.predict(test_data))
    for i in range(len(classifier.coefs_)):
        number_neurons_in_layer = classifier.coefs_[i].shape[1]
        for j in range(number_neurons_in_layer):
            weights = classifier.coefs_[i][:,j]
            print(i, j, weights, end=", ")
            print()
        print()

    # MyMLP (Backpropagation) with mini-batch gradient descent
    print('\n\n--------------------')
    print('MyMLP (Backpropagation) with mini-batch gradient descent')
    print('--------------------')
    model = make_network(hidden_layer_sizes=(3,1,2))
    # model = make_network(hidden_layer_sizes=(5,))
    model = sgd(model, iris_data.data, iris_data.target, minibatch_size)

    # for W in model:
    #     print(W)
    #     print(model[W])
    #     print()

    for i, x in enumerate(test_data):
        # print('x', x)
        _, prob = forward(x, model)
        # print('hs', _)
        # print(prob)
        y = np.argmax(prob)
        print(y)

    for i in range(len(model)):
        number_neurons_in_layer = len(model['W'+str(i+1)][0])
        for j in range(number_neurons_in_layer):
            weights = model['W'+str(i+1)][:,j]
            print(i, j, weights, end=", ")
            print()
        print()

if __name__ == "__main__":
    main()