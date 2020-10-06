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

def initialize_parameters(n_x,n_h,n_y):
    # n_x = size of the input layer
    # n_h = size of the hidden layer
    # n_y = size of the output layer
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1,X)+b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def backward_propagation(parameters, cache, X, Y):
    """
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = A2-Y
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def compute_cost(A,Y):
    m = Y.shape[1]
    cost = -np.sum([Y*np.log(A)+(1-Y)*np.log(1-A)])/m  
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
    return cost

def update_parameters(parameters, grads, learning_rate = 0.01):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### (≈ 4 lines of code)
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    ## END CODE HERE ###
    
    # Update rule for each parameter
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    ### END CODE HERE ###
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def predict(X, parameters):
    #Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    
    A, cache = forward_propagation(X, parameters)
    
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i]<=0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    #Y_prediction = A2>0.5
    return Y_prediction

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
#### Functions end here

## LETS BUILD THE MODEL

def Hidden_Layer_model(X_train, Y_train, X_test, Y_test, num_iterations = 500, learning_rate = 0.25, print_cost = False):
    np.random.seed(23)
    grads = {}
    costs_train = []                              # to keep track of the cost
    costs_test = []  
    #Network Dimensions
    n_x = X_train.shape[0] # Input image size
    n_h = 64    # Set by the Lab
    n_y = Y_train.shape[0] # size of output layer

    # Initialize parameters dictionary
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Loop (gradient descent)

    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X_train, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost_train = compute_cost(A2, Y_train)
 
         # To compute and store test cost/loss
        A_test, cache_temp = forward_propagation(X_test, parameters)
        cost_test = compute_cost(A_test, Y_test)


        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X_train, Y_train)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads,learning_rate)
        


        # Print the cost every 1000 iterations
        if print_cost and i % 100 == 0:
            print ("Training Cost after iteration %i: %f" %(i, cost_train))
            print ("Test Cost after iteration %i: %f" %(i, cost_test))
        # Record the costs
        if i % 1 == 0:
            costs_train.append(cost_train)
            costs_test.append(cost_test)
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(X_test, parameters)
    Y_prediction_train = predict(X_train, parameters)



    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs_train": costs_train,
         "costs_test": costs_test,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "parameters" : parameters,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


NN_model = Hidden_Layer_model(X_train, y_train,X_test,y_test,500,0.25,True)
plot_loss(NN_model["costs_train"],NN_model["costs_test"])