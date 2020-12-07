from sklearn import datasets
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

#Create dummy data
np.random.seed(0)
feature_set, labels = datasets.make_moons(100, noise=0.1)
plt.figure(figsize=(10,7))
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap=plt.cm.winter)

labels = labels.reshape(100,1)

#plt.show()

#Coding feedforward propagation

#Outputs from the input layer are passed to corresponding nodes in the
#hidden layer, and multiplied by the weight values of that node. These initial
#input values represent two features of each data point in our dataset (could be
#more, but we will use only two).

#1st step: compute the dot product of these 2 features of our 2 dimensional input.
#zl1 = x1w1 + x2w2 (x are features and w are weights)

#2nd step: pass the product through an activation function to get the value of
#the node for that layer.

def sigmoid (x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))

#Initializing weights with random numbers
    #4 is the number of nodes in the hidden layer
wh = np.random.rand(len(feature_set[0]),4)

#Creating multiple layers: adding output of each node in the previous layer
#together and run through activation function.

#Initializing output weights
wo = np.random.rand(4,1)

#Initializing learning rate
lr = 0.5

#Put feedforward and backpropagation inside a for loop. Each pass is an epoch.
for epoch in range(20000):
    ##Feedforward propagation
    zh = np.dot(feature_set,wh)
    ah = sigmoid(zh)

    zo = np.dot(ah,wo)
    ao= sigmoid(zo)  #Feedforward output

    #Phase 1: for output weights

    #Now, backpropagation: calculation of the loss.

    ##Gradient descent: network makes a backward pass through its layers to
    #correct nodes based on a loss function.
    #Mean squared error
    error_out = ((1/2) * (np.power((ao - labels), 2)))
    print(error_out.sum())
    #Derivative cost
    dcost_dao = ao-labels  #cost is output minus labels

    #Passing dot product from previous layer into derivative of sigmoid 
    dao_dzo = sigmoid_der(zo)

    #Getting the output:
    dzo_dwo = ah

    #Derivative cost for output weights:
    dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)


    #Phase 2: for hidden layer weights
    dcost_dzo = dcost_dao * dao_dzo #Derivative cost of final output is
                                    #derivative cost of out_layer multiplied by
                                    #sigmoidal derivative of output
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo, dzo_dah.T)

    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)


    #Update weights
    wh -= lr * dcost_wh
    wo -= lr * dcost_wo
    
    
                 

    
