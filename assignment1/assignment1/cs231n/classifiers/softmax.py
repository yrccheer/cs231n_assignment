import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[0]
  num_train = X.shape[1]
  for i in range(num_train):
    scores = W.dot(X[:,i])
    scores -= np.max(scores)
    sum = 0.0
    for j in scores:
        sum += np.exp(j)
    loss = -scores[y[i]]+ np.log(sum)
  for k in range(num_classes):
      p = np.exp(scores[k])/sum
      dW[k, :] += (p-(k == y[i])) * X[:, i]
        
  loss /= num_train
  dW /= num_train
  
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[1]
  num_classes = W.shape[0]
  scores = np.dot(W,X)
  scores -= np.max(scores)
  correct_scores = scores[y,range(num_train)]
  loss = -np.mean(np.log(np.exp(correct_scores)/np.sum(np.exp(scores))))
  p = np.exp(scores)/np.sum(np.exp(scores), axis=0)
  ind = np.zeros(p.shape)
  ind[y, range(num_train)] = 1
  dW = np.dot((p-ind), X.T)
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
