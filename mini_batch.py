import numpy as np

n_feature = 4 # data type (petal length, petal widht, sepal length, sepal widht)
n_class = 3 # target class (iris versicolor, iris setosa, iris virginica)
n_iter = 5 # total iteration
learn_rate = 1e-2 # learning rate
hidden_layer = ()

def f(out, target):
    res = []
    for i in range(len(out)):
        temp = []
        for j in range(len(out[i])):
            temp.append(-(target[i][j]-out[i][j]) * out[i][j] * (1-out[i][j]))
        res.append(temp)
    return res

def h(weight, out, target):
    res = []
    temp = f(out, target)
    for i in range(len(out)):
        sigma = 0
        for j in range(len(weight)):
            sigma += temp[j] * weight[j] * out[j] * (1-out[j])
        res.append(sigma)
    return res

# default set hidden layer size to 100
def make_network(hidden_layer_sizes=(100,)):
    global hidden_layer
    hidden_layer = hidden_layer_sizes
    size = len(hidden_layer_sizes)

    # Initialize weights with Standard Normal random variables
    model = dict()
    model['W1'] = np.random.randn(n_feature, hidden_layer_sizes[0])
    for i in range(size - 1):
        model['W' + str(i+2)] = np.random.randn(hidden_layer_sizes[i], hidden_layer_sizes[i+1])
    model['W' + str(size+1)] = np.random.randn(hidden_layer_sizes[size-1], n_class)

    return model

# sigmoid function
def sigmoid(x):
    # print(x)
    return 1/(1+np.exp(-x))

def forward(x, model):
    global hidden_layer
    size = len(hidden_layer)
    hs = []

    # Input to hidden
    # print('x', x)
    # print('h', h)
    # print('model', model['W1'][0])
    h = sigmoid(x @ model['W1'])
    hs.append(h)
    # print(1, h, end='\n')

    for i in range(size - 1):
        h = sigmoid(h @ model['W' + str(i+2)])
        # print(f'h{i+2}', h)
        hs.append(h)
        # print(i+2, h, end='\n')

    # Hidden to output
    prob = sigmoid(h @ model['W' + str(size+1)])
    # print('out', prob)
    return hs, prob

def backward(model, xs, hs, errs, target, y_pred):
    """
        xs, hs, errs contain all informations
        (input, hidden state, error) of all data
        in the minibatch
    """
    global hidden_layer
    print('xs: ', len(xs))
    print('errs: ', len(errs))
    print('y: ', len(target))
    size = len(hidden_layer)
    dW = [[] for i in range(size+1)]

    # print('dW', size, dW)
    # print('model', model)
    # print('xs : ', len(xs), xs)
    # print('hs[size-1] : ', len(hs[size-1]), np.array(hs[size-1]))
    # print('errs : ', len(errs), errs)

    # errs are the gradients from output layer for minibatch
    # dW[size] = np.array(hs[size - 1]).T @ errs

    # for i in range(len(xs)):
    #     print('hs test: ', hs[len(hs)-1][i])
    test1 = np.array(hs[len(hs)-1]).T
    # print('test1: ', len(test1))
    # print('test1: ', len(test1[0]))
    test2 = np.array(f(y_pred, target))
    # print('test2: ', len(test2))
    # print('test2: ', len(test2[0]))
    dW[size] = test1 @ test2
    print('dW: ', dW[size])

    # Get gradient of hidden layer
    # .T == transpose matrix
    # dh = errs @ model['W' + str(size+1)].T
    for i in reversed(range(size - 1)):
        dW[i+1] = np.array(hs[i]).T @ dh
        dh = dh @ model['W' + str(i+2)].T

    dW[0] = xs.T @ dh

    grad_dict = dict()
    for i in range(size+1):
        grad_dict['W' + str(i+1)] = dW[i]

    return grad_dict

def sgd(model, X_train, y_train, minibatch_size):
    for iter in range(n_iter):
        # print('Iteration {}'.format(iter))

        # Randomize data point
        # X_train, y_train = shuffle(X_train, y_train)

        for i in range(0, X_train.shape[0], minibatch_size):
            # print(i)
            # Get pair of (X, y) of the current minibatch/chunk
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]
            # print(f'\n\nSGD_Step {i}\n----------')

            model = sgd_step(model, X_train_mini, y_train_mini)

    return model

def sgd_step(model, X_train, y_train):
    grad = get_minibatch_grad(model, X_train, y_train)
    model = model.copy()

    # Update every parameters in our networks (W1 and W2) using their gradients
    for layer in grad:
        model[layer] -= learn_rate * grad[layer]

    return model

def get_minibatch_grad(model, X_train, y_train):
    global hidden_layer
    size = len(hidden_layer)
    xs, hs, errs = [], [[] for i in range(size)], []

    pred = []
    target = []
    
    for x, cls_idx in zip(X_train, y_train):    
        h, y_pred = forward(x, model)
        
        pred.append(y_pred)

        # Create probability distribution of true label
        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.
        target.append(y_true)

        # Compute the gradient of output layer
        err = ((y_true - y_pred)**2)/2

        # Accumulate the informations of minibatch
        # x: input
        # h: hidden state
        # err: gradient of output layer
        xs.append(x)

        for i in range(size):
            arr = []
            for j in range(len(h[i])):
                arr.append(h[i][j])
            hs[i].append(np.array(arr))
        errs.append(err)

    print('pred: ', pred)
    print('hs: ', hs[2])
    # print('xs', xs)
    print('hs len: ', len(hs))
    # for i in range(size):
    #     print(len(hs[i]))

    # Backprop using the informations we get from the current minibatch
    return backward(model, np.array(xs), hs, np.array(errs), target, pred)

    # for i in reversed(range(len(xs))):
    #     backprop

