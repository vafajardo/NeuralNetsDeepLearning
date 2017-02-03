import numpy as np

class pcn:
    def __init__(self, inputs, targets, weights):
        self.inputs = inputs
        self.nData = len(inputs)
        self.nOut = targets.shape[1]
        self.inputs = np.concatenate((inputs, np.ones(4).reshape(4,1)), axis=1) # add bias node
        self.weights = weights
        self.targets = targets
        self.arch = (inputs.shape[1],self.nOut)
    
    def train(self, T=5, eta=0.25,printscn = False):
        """
        This method trains the Perceptron.
        """
        for i in range(T): 
            # calculate sums for the output nodes
            h = np.dot(self.inputs,self.weights)
            # calculate outputs for the output nodes
            y = np.where(h>0, 1, 0)
            self.weights -= eta*np.dot(self.inputs.T, (y-self.targets)) # update the weights
            if printscn:
                print('iteration {0}'.format(i))
                print(y)
                print(self.weights)

    def recall(self):
        """
        This method uses the current sets of weights, and outputs
        the firing of the output nodes when fed the original input data.
        """
        # calculate sums for the output nodes
        h = np.dot(self.inputs,self.weights)
        # calculate outputs for the output nodes
        y = np.where(h>0, 1, 0)
        return y   
 
    def forward(self, newdata):
        """
        This method uses the current sets of weights, and outputs
        the firing of the output nodes when fed an input vector.
        """
        # calculate sums for the output nodes
        h = np.dot(newdata,self.weights)
        # calculate outputs for the output nodes
        y = np.where(h>0, 1, 0)
        return y
