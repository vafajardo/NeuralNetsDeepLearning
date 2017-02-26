import numpy as np

def initializeWeights(nIn,nWeights,shape=None):
    """
    Returns an np.array of randomized weights according to the UNIF(-1/sqrt(nIn),1/sqrt(nIn)).
    """
    if shape:
        assert type(shape) == tuple
        return (-1/np.sqrt(nIn) + np.random.rand(nWeights)*(2/np.sqrt(nIn))).reshape(shape)
    else:
        return -1/np.sqrt(nIn) + np.random.rand(nWeights)*(2/np.sqrt(nIn))

class mlp:
    def __init__(self, inputs, targets, hiddenlayers,hiddenact = 'sigmoid',seed=None):
        """
        hiddenlayers = an np.array providing the number hidden nodes in each hidden layer
        hiddenact = the acitivation function used for all of the nodes in the hidden layerz
        """
        self.inputs = inputs
        self.hiddenlayers = hiddenlayers
        self.targets = targets
        self.arch = np.hstack((np.array([self.inputs.shape[1]]),
                                self.hiddenlayers,
                                np.array([self.targets.shape[1]])))
        self.nlayers = len(self.arch)
        if seed:
            np.random.seed(seed)
        self.weights = [initializeWeights(nIn=self.arch[i]+1,
                            nWeights=(self.arch[i] + 1)*self.arch[i+1],
                            shape=(self.arch[i]+1,self.arch[i+1]))
                        for i in range(len(self.arch))[:-1]]
        self.updates = [np.zeros((self.arch[i]+1,self.arch[i+1])) for i in range(self.nlayers - 1)] # list of np.arrays storing weight updates
        self.nepochs = None
        self.hiddenact = hiddenact

    def simpletrain(self,inputs,targets,eta=0.25,T=int(1e4),momentum=0.9):
        """
        This method trains the NN for a specified number of iterations.
        """
        for ite in range(T):
            randomizedIndex = np.random.permutation(inputs.shape[0])
            actsWithBias = [np.ones(self.arch[i] + 1) for i in range(self.nlayers)] # note redundant bias included for output layer
            deltas = [np.ones(self.arch[i]) for i in range(self.nlayers)]
            for ix in randomizedIndex:
                actsWithBias[0][1:] = inputs[ix]
                thisTarget = targets[ix]
                # Forward Pass
                for l in range(self.nlayers)[1:-1]: # compute signals and activations of nodes in every layer
                    signals = np.dot(actsWithBias[l-1],self.weights[l-1]) # signals of layer l nodes
                    if self.hiddenact == 'tanh':
                        actsWithBias[l][1:] = np.tanh(signals) # tanh activation functions
                    else:
                        actsWithBias[l][1:] = 1 / (1 + np.exp(-signals)) # Sigmoid activations of layer l nodes
                # compute signal and activations of output nodes (always sigmoid)
                l += 1
                signals = np.dot(actsWithBias[l-1],self.weights[l-1])
                actsWithBias[l][1:] = 1 / (1 + np.exp(-signals))

                # print(actsWithBias)
                # Backward Pass
                # ... compute deltas starting from the end of the NN
                outputs = actsWithBias[self.nlayers - 1][1:]
                deltas[self.nlayers - 1] = (outputs - thisTarget)*outputs*(1-outputs)
                # ... compute deltas for hidden layers
                for j in range(self.nlayers)[1:-1][::-1]: # traverse weights backwards
                    if self.hiddenact == "tanh":
                        deltas[j] = np.dot(deltas[j + 1], self.weights[j][1:,].T) \
                            * (1-actsWithBias[j][1:]**2)
                    else:
                        deltas[j] = np.dot(deltas[j + 1], self.weights[j][1:,].T) \
                            * actsWithBias[j][1:] * (1-actsWithBias[j][1:])
                # print(deltas)
                for i,wt in enumerate(self.weights):
                    self.updates[i] = eta*np.kron(deltas[i+1],actsWithBias[i][:,None]) + momentum*self.updates[i]
                    self.weights[i] -= self.updates[i]

    def train(self, trainIn, trainT, validIn, validT, eta=0.25, method='earlystop', valRatio=0.5, interval=10, momentum=0.9):
        """
        This method trains the NN and must be supplied train and validation sets.
        """
        # this early stopping will use two orders of error
        # both of these steps in error must be increasing to stop training
        thisError = int(1e6)
        order1Error = int(1e6) + 1
        order2Error = int(1e6) + 2
        nepochs = 0
        while (order1Error - thisError > 0) or (order2Error - order1Error> 0):
            nepochs += 1
            self.simpletrain(trainIn, trainT, eta, T=interval, momentum=momentum)
            order2Error = order1Error
            order1Error = thisError
            validOut = self.predict(validIn)
            thisError = 0.5*((validOut - validT)**2).sum()
        self.nepochs = nepochs * interval
        print("Stopped after {0} epochs".format(nepochs))

    def predict(self, inputs, method='raw'):
        """
        This method takes an np.array of input vectors and produces the outputs
        of each input vector. In other words, it performs the forward pass of the NN.
        """
        actsWithBias = [np.ones(self.arch[i] + 1) for i in range(self.nlayers)] # note redundant bias included for output layer
        outputs = np.zeros(len(inputs)*self.arch[-1]).reshape(len(inputs), self.arch[-1])
        for i,thisInput in enumerate(inputs):
            actsWithBias[0][1:] = thisInput
            # Forward Pass
            for l in range(self.nlayers)[1:-1]: # compute signals and activations of nodes in every layer
                signals = np.dot(actsWithBias[l-1],self.weights[l-1]) # signals of layer l nodes
                if self.hiddenact == 'tanh':
                    actsWithBias[l][1:] = np.tanh(signals) # tanh activation functions
                else:
                    actsWithBias[l][1:] = 1 / (1 + np.exp(-signals)) # Sigmoid activations of layer l nodes
            # compute signal and activations of output nodes (always sigmoid)
            l += 1
            signals = np.dot(actsWithBias[l-1],self.weights[l-1])
            actsWithBias[l][1:] = 1 / (1 + np.exp(-signals))
            outputs[i,:] = actsWithBias[-1][1:]
        if method == 'max':
            assert self.arch[-1] > 1 # should only use this when output layer has multiple nodes
            maxOutputs = outputs.max(axis=1)[:,None]
            outputs = (outputs == maxOutputs).astype(int)
        return outputs

    def cfnmatrix(self, inputs, targets):
        """
        Produce a confusion matrix of the result of feeding in the labelled data
        (inputs, targets) to the NN.
        """
        predicted = self.predict(inputs)
        if self.arch[-1] == 1:
            binaryOut = np.where(predicted > 0.5, 1, 0)
            cmatrix = np.array([[(binaryOut==0).sum() - targets[binaryOut == 0].sum(),targets[binaryOut == 0].sum()],
            [(binaryOut == 1).sum() - targets[binaryOut == 1].sum(),targets[binaryOut == 1].sum()]])
        else:
            predictedClass = predicted.max(axis=1)[:,None] # predicted class of each pattern
            binaryOut = (predicted == predictedClass).astype(int)
            cmatrix = np.zeros((self.arch[-1],self.arch[-1]))
            for c in range(self.arch[-1]):
                cmatrix[:,c] = targets[binaryOut[:,c] == 1].sum(axis=0)
        print('Error: {0}'.format(1-cmatrix.trace()/cmatrix.sum()))
        return cmatrix
