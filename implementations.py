import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param max_iters: number of iterations
    :param gamma: step size
    :return: weights, loss
    """
    w = initial_w
    n = len(y)
    
    for n_iter in range(max_iters):
        gradient = 1/n * tx.T.dot(tx.dot(w) - y)
        w = w - gamma * gradient
    
    loss = 1/(2*n) * np.sum((y - tx.dot(w))**2) # or 1/(2*n) * np.linalg.norm(y - tx.dot(w))**2

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, mini_batch_size=1):
    """
    Linear regression using stochastic gradient descent
    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param max_iters: number of iterations
    :param gamma: step size
    :return: weights, loss
    """
    w = initial_w
    n = len(y)
    
    for n_iter in range(max_iters):
        random_indices = np.random.randint(0, n, mini_batch_size)
        tx_batch = tx[random_indices]
        y_batch = y[random_indices]

        gradient = 1/mini_batch_size * tx_batch.T.dot(tx_batch.dot(w) - y_batch)
        w = w - gamma * gradient
    
    loss = 1/(2*n) * np.sum((y - tx.dot(w))**2)

    return w, loss

def least_squares(y, tx):
    """
    Least squares regression using normal equations
    :param y: labels
    :param tx: features
    :return: weights, loss
    """
    # X^T * X * w = X^T * y
    # w = (X^T * X)^-1 * X^T * y

    # We can solve this equation by isolating w (1) or by using np.linalg.solve (2). (2) is faster and more stable.
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))

    loss = 1/(2*len(y)) * np.sum((y - tx.dot(w))**2)

    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    :param y: labels
    :param tx: features
    :param lambda_: regularization parameter
    :return: weights, loss
    """
    # X^T * X * w + lambda * w = X^T * y
    # w = (X^T * X + lambda * I)^-1 * X^T * y
    # scaled: w = (X^T * X + N * lambda * I)^-1 * X^T * y

    w = np.linalg.solve(tx.T.dot(tx) + 2 * len(y) * lambda_ * np.eye(tx.shape[1]), tx.T.dot(y))

    loss = 1/(2*len(y)) * np.sum((y - tx.dot(w))**2)

    return w, loss

# consider adding eps = 1e-8 to y_hat to avoid log(0)
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent (y ∈ {0,1})
    :param y: labels
    :param tx: features
    :param initial_w: initial weights
    :param max_iters: number of iterations
    :param gamma: step size
    :return: weights, loss
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    w = initial_w
    n = len(y)

    for n_iter in range(max_iters):
        gradient = 1/n * tx.T.dot(sigmoid(tx.dot(w)) - y)
        w = w - gamma * gradient
    
    y_hat = sigmoid(tx.dot(w))
    loss = -1/n * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent (y ∈ {0,1}, with regularization term λ∥w∥2)
    :param y: labels
    :param tx: features
    :param lambda_: regularization parameter
    :param initial_w: initial weights
    :param max_iters: number of iterations
    :param gamma: step size
    :return: weights, loss
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    w = initial_w
    n = len(y)

    for n_iter in range(max_iters):
        gradient = 1/n * tx.T.dot(sigmoid(tx.dot(w)) - y) + 2 * lambda_ * w
        w = w - gamma * gradient
    
    y_hat = sigmoid(tx.dot(w))
    loss = -1/n * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    return w, loss