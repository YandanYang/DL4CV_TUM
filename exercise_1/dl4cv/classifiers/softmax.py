import numpy as np
from random import shuffle

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num=X.shape[0]
    P=np.dot(X,W)
    P=P-np.reshape(P.max(1),(num,1))
    exp=np.exp(P)/np.sum(np.exp(P),axis=1,keepdims=True)
    py=np.zeros(num)
    for i in range(num):
        py[i]=exp[i,y[i]]
        loss=loss-np.log(py[i])
    loss=loss/num
    loss+=0.5*reg*np.sum(W*W)
    dw=exp
    for i in range(num):
        dw[i][y[i]]-=1
    dw=dw/num
    dW=np.dot(X.T,dw)+reg*W
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
    num=X.shape[0]
    P=np.dot(X,W)
    P=P-np.reshape(P.max(1),(num,1))
    exp=np.exp(P)/np.sum(np.exp(P),axis=1,keepdims=True)
    py=np.zeros((num,10))
    py[range(num),y]=1.0
    loss=-np.sum(py*np.log(exp))/num+0.5*reg*np.sum(W*W)
    dw=(exp-py)/num
    dW=np.dot(X.T,dw)+reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

