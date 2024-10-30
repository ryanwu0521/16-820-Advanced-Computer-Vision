import numpy as np
from util import *

# do not include any more libraries here!
# do not put any code outside of functions!


############################## Q 2.1.2 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b.
# X be [Examples, Dimensions]
def initialize_weights(in_size, out_size, params, name=""):
    W, b = None, None

    # Xavier initialization
    limit = np.sqrt(6) / np.sqrt(in_size + out_size)
    W = np.random.uniform(-limit, limit, (in_size, out_size))

    # bias initialization (1D array)
    b = np.zeros(out_size)

    params["W" + name] = W
    params["b" + name] = b


############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None
    X = np.clip(x, -500, 500) # prevent overflow warning

    res = 1 / (1 + np.exp(-x))

    return res


############################## Q 2.2.1 ##############################
def forward(X, params, name="", activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params["W" + name]
    b = params["b" + name]

    # compute the pre-activation
    pre_act = np.dot(X, W) + b

    # compute the post-activation
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params["cache_" + name] = (X, pre_act, post_act)

    return post_act


############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    # subtract the max for numerical stability
    x = x - np.max(x, axis=1, keepdims=True)

    # compute the softmax
    res = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims = True)

    return res


############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    # compute the loss
    loss = -np.sum(y * np.log(probs))

    # compute the accuracy
    acc = np.sum(np.argmax(y, axis=1) == np.argmax(probs, axis=1)) / y.shape[0]

    return loss, acc


############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act * (1.0 - post_act)
    return res


def backwards(delta, params, name="", activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params["W" + name]
    b = params["b" + name]
    X, pre_act, post_act = params["cache_" + name]

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X
    
    # compute the derivative (delta)
    delta = delta * activation_deriv(post_act)

    # compute the gradient of W
    grad_W = np.dot(X.T, delta)

    # compute the gradient of b
    grad_b = np.sum(delta, axis=0)

    # compute the gradient of X
    grad_X = np.dot(delta, W.T)

    # store the gradients
    params["grad_W" + name] = grad_W
    params["grad_b" + name] = grad_b
    return grad_X


############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x, y, batch_size):
    batches = []

    # shuffle the data
    num_data = x.shape[0]
    indices = np.arange(num_data)
    np.random.shuffle(indices)

    # split the data into batches
    for i in range(0, num_data, batch_size):
        batch_indices = indices[i:i+batch_size]
        batches.append((x[batch_indices], y[batch_indices]))

    return batches
