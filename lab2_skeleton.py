from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt

#In this first part, we just prepare our data (mnist)
#for training and testing

#import keras
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255


#We want to have a binary classification: digit 0 is classified 1 and
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new


y_train = y_train.T
y_test = y_test.T


m = X_train.shape[1] #number of examples

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]


# #Display one image and corresponding label
# import matplotlib
# import matplotlib.pyplot as plt
# i = 3
# print('y[{}]={}'.format(i, y_train[:,i]))
# plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()


#Let start our work: creating a neural network
#First, we just use a single neuron.


#####TO COMPLETE
## Auxiliary Functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def initialize_parameters(dim):
    #This function creates a vector of zeros of shape (dim, 1) for weights and initializes bias to 0.
    w = np.random.uniform(-0.005,0.005,size= (dim,1))
    #w = np.zeros((dim,1))
    b = 0
    return w,b


def propagate(w, b, X, Y):
    #This function computes the forward propagation, the cost and the gradients

    m = X.shape[1]

    Z = np.dot(w.T,X)+b
    A = sigmoid(Z)
    cost = -np.sum([Y*np.log(A)+(1-Y)*np.log(1-A)])/m  

    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm

        1) Calculate the cost and the gradient for the current parameters. we used propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs_train = []
    costs_test = []
   
    for i in range(num_iterations):
        grads_train, cost_train = propagate(w, b, X_train, Y_train)
       
        grads_test, cost_test = propagate(w, b, X_test, Y_test)
       
        # Retrieve derivatives from grads
        dw = grads_train["dw"]
        db = grads_train["db"]
       
        # update rule
        w = w-learning_rate*dw
        b = b-learning_rate*db
       
        # Record the costs
        if i % 1 == 0:
            costs_train.append(cost_train)
            costs_test.append(cost_test)
       
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Train cost after iteration %i: %f" %(i, cost_train))
            print ("Test cos after iteration %i: %f" %(i, cost_test))
   
    params = {"w": w,
              "b": b}
   
    grads_train = {"dw": dw,
             "db": db}
   
    return params, grads_train, costs_train, costs_test

def predict(w, b, X):
    #Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
   
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X)+b)
   
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i]<=0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

#### Functions end here

## LETS BUILD THE MODEL

def plot_loss(costs_train, costs_test):
    plt.figure()
    temp = np.arange(0, len(costs_train), 1)
    plt.plot(temp,costs_train, label='Train Loss function')
    plt.plot(temp,costs_test, label='Test Loss function')
    plt.xlabel('epochs')
    plt.ylabel('Loss values, Cost function')
    plt.legend()
    plt.title('Evolution of the Loss(Cost) function')
    plt.show()


def model(X_train, Y_train, X_test, Y_test, num_iterations = 500, learning_rate = 0.5, print_cost = False):
    #Builds the single neuron model using logistic regression by calling the function implemented previously

    # initialize parameters with zeros
    w, b = initialize_parameters(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs_train, costs_test = optimize(w, b, X_train, Y_train, X_test, Y_test, num_iterations, learning_rate,print_cost)
   
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
   
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

     # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs_train": costs_train,
         "costs_test": costs_test,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
   
    return d

####
single_neuron = model(X_train, y_train,X_test,y_test,500,0.25,True)
plot_loss(single_neuron["costs_train"],single_neuron["costs_test"])