from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    num_train = X.shape[0]
    num_classes = W.shape[1]
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
      scores = np.dot(X[i], W)
      # Shift scores for numerical stability
      shifted_scores = scores - np.max(scores)

      # Calculate sum of the exponentiated row
      exp_sum = np.sum(np.exp(shifted_scores))

      # Calculate this expression, for it will have use in formulas for 
      # gradient
      exp_by_exp_sum = np.exp(shifted_scores) / exp_sum

      for j in range(num_classes):
        if j != y[i]:
          dW[:, j] += X[i] * exp_by_exp_sum[j]
      dW[:, y[i]] += X[i] * (-1 + exp_by_exp_sum[y[i]])

      # Now lets add Li to the entire Loss
      loss += -shifted_scores[y[i]] + np.log(np.sum(np.exp(shifted_scores)))

    loss /= num_train

    loss += reg * np.sum(W*W)
    
    # Gradient regularization
    dW = dW / num_train + 2*reg*W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Get the scores matrix
    scores = np.dot(X, W)
    # Shift scores for the sake of numerical stability
    shift_scores = scores - np.max(scores, axis = 1)[:, np.newaxis]

    # Find softmax scores matrix 
    softmax_scores = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis = 1)[:, np.newaxis]
    
    P = np.choose(y, softmax_scores.T)
    # Find Loss, average it and regularize it 
    loss = -np.sum(np.log(P))
    loss = loss / num_train + reg * np.sum(W*W)

    # Substact one from every position of the right label
    # (because formulas for d(Li)/ d(w_j|w_yi) are identical except for the fact
    #  that in the latter formula we additionaly substract 1)
    softmax_scores[range(num_train), y] -= 1
    dW = np.dot(X.T, softmax_scores)

    # Gradient regularization
    dW = dW / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
