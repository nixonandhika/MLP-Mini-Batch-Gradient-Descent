import numpy as np

n_feature = 4 # data type (petal length, petal widht, sepal length, sepal widht)
n_class = 3 # target class (iris versicolor, iris setosa, iris virginica)
n_iter = 5000 # max iteration
max_divergen = 50
err_threshold = 1e-8 #error threshold
learn_rate = 5e-2 # learning rate
hidden_layer = ()

def f(out, target):
    res = []
    for i in range(len(out)):
        temp = []
        for j in range(len(out[i])):
            temp.append(-(target[i][j]-out[i][j]) * out[i][j] * (1-out[i][j]))
        res.append(temp)
    return res

def h(w_curr, out_target, desired_target, out_curr):
    res = []
    for i in range(len(out_target)):
        sigma_f = 0

        for j in range(len(out_target[i])):
            sigma_f += -(desired_target[i][j]-out_target[i][j]) * out_target[i][j] * (1-out_target[i][j])
        temp = []

        for j in range(len(out_curr[i])):
            temp.append(sigma_f * w_curr[j] * out_curr[i][j])
        res.append(temp)

    return res

# default set hidden layer size to 100
def make_network(hidden_layer_sizes=(100,)):
    global hidden_layer
    hidden_layer = hidden_layer_sizes
    size = len(hidden_layer_sizes)

    # Initialize Random Number between -1 and 1
    model = dict()

    model['W1'] = np.random.uniform(-1, 1, (n_feature, hidden_layer_sizes[0]))
    for i in range(size - 1):
        model['W' + str(i+2)] = np.random.uniform(-1, 1, (hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
    model['W' + str(size+1)] = np.random.uniform(-1, 1, (hidden_layer_sizes[size-1], n_class))

    model['bias'] = np.random.randn(size)

    return model

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward(x, model):
    global hidden_layer
    size = len(hidden_layer)
    hs = []

    # Input to hidden
    h = sigmoid(x @ model['W1'] + model['bias'][0])
    hs.append(h)

    for i in range(size - 1):
        h = sigmoid(h @ model['W' + str(i+2)] + model['bias'][i+1])
        hs.append(h)

    # Hidden to output
    prob = sigmoid(h @ model['W' + str(size+1)])
    return hs, prob

def backward(model, xs, hs, errs, target, y_pred):
    global hidden_layer
    size = len(hidden_layer)
    dW = [[] for i in range(size+1)]

    test1 = np.array(hs[len(hs)-1]).T
    test2 = np.array(f(y_pred, target))
    dW[size] = test1 @ test2

    # Get gradient hidden layer
    for i in reversed(range(size - 1)):
        test1 = np.array(hs[i]).T
        test2 = np.array(h(model['W' + str(i+2)][0], y_pred, target, hs[i+1]))
        dW[i+1] = test1 @ test2
    
    test1 = xs.T
    test2 = np.array(h(model['W' + str(1)][0], y_pred, target, hs[0]))
    dW[0] = test1 @ test2

    grad_dict = dict()
    for i in range(size+1):
        grad_dict['W' + str(i+1)] = dW[i]

    return grad_dict

def sgd(model, X_train, y_train, minibatch_size):
    prev_e_total = 99999
    last_iter = n_iter
    diverge_count = 0
    for iter in range(n_iter):

        curr_e_total = 0

        for i in range(0, X_train.shape[0], minibatch_size):
            
            # Get pair of (X, y) of minibatch
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]
            
            model, e_total = sgd_step(model, X_train_mini, y_train_mini)
            
            curr_e_total += e_total
        
        if (diverge_count > max_divergen and prev_e_total - curr_e_total < err_threshold):
            last_iter = iter+1
            break
        else:
            if (curr_e_total > prev_e_total):
                diverge_count += 1
            prev_e_total = curr_e_total

    print('Iteration {}'.format(last_iter))
    return model

def sgd_step(model, X_train, y_train):
    grad, e_total = get_minibatch_grad(model, X_train, y_train)
    model = model.copy()

    # Update all networks
    for layer in grad:
        model[layer] -= learn_rate * grad[layer]

    return model, e_total

def get_minibatch_grad(model, X_train, y_train):
    global hidden_layer
    size = len(hidden_layer)
    xs, hs, errs = [], [[] for i in range(size)], []

    pred = []
    target = []
    e_total = 0
    
    for x, cls_idx in zip(X_train, y_train):    
        h, y_pred = forward(x, model)
        
        pred.append(y_pred)

        # Probability Distribution
        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.
        target.append(y_true)

        # Gradient Output Layer
        err = ((y_true - y_pred)**2)/2
        e_total += sum(err)

        xs.append(x)

        for i in range(size):
            arr = []
            for j in range(len(h[i])):
                arr.append(h[i][j])
            hs[i].append(np.array(arr))
            
        errs.append(err)

    return backward(model, np.array(xs), hs, np.array(errs), target, pred), e_total