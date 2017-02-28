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
        self.npatterns = inputs.shape[0]
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
        self.outputs = None

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
                    self.updates[i] = eta*np.outer(actsWithBias[i],deltas[i+1]) + momentum*self.updates[i]
                    self.weights[i] -= self.updates[i]

    def train(self, trainIn, trainT, validIn, validT, eta=0.25, burnin=100, interval=10, maxchecks=1000, **kwargs):
        """
        This method trains the NN and must be supplied train and validation sets.
        """
        # this early stopping will use two orders of error
        # both of these steps in error must be increasing to stop training
        thisError = int(1e6)
        order1Error = int(1e6) + 1
        order2Error = int(1e6) + 2
        nchecks = 0
        self.simpletrain(trainIn, trainT, eta, T=burnin, **kwargs) # burnin phase
        while (order1Error - thisError > -0.01) or (order2Error - order1Error> -0.01):
            nchecks += 1
            self.simpletrain(trainIn, trainT, eta, T=interval, **kwargs)
            order2Error = order1Error
            order1Error = thisError
            validOut = self.predict(validIn)
            thisError = 0.5*((validOut - validT)**2).sum()
            if (nchecks % 100 == 0):
                print('...val error after {0} checks: '.format(nchecks), thisError)
            if nchecks > maxchecks:
                print('...reached maximum number of checks before convergence')
                break
        self.nepochs = nchecks * interval + burnin
        print("Validation error after {0} checks: {1}".format(nchecks, thisError))

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
                elif self.hiddenact == 'sigmoid':
                    actsWithBias[l][1:] = 1 / (1 + np.exp(-signals)) # Sigmoid activations of layer l nodes
                else:
                    raise ValueError('invalid hiddenact')
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

class mlpbatch(mlp):
    def __init__(self, inputs, targets, hiddenlayers,hiddenact = 'sigmoid',wtdecay=1,seed=None):
        mlp.__init__(self, inputs, targets, hiddenlayers,hiddenact,seed)
        self.wtdecay = 1

    def simpletrain(self,inputs,targets,eta=0.25,T=int(1e4),momentum=0.9):
        """
        This method trains the NN for a specified number of iterations.
        """
        for ite in range(T):
            actsWithBias = [np.concatenate((np.ones((self.npatterns, 1)), inputs), axis=1)]
            actsWithBias += [np.ones((self.npatterns, self.arch[i] + 1)) for i in range(1,self.nlayers)] # note redundant bias included for output layer
            deltas = [np.ones((self.npatterns, self.arch[i])) for i in range(self.nlayers)]

            # Forward pass
            for l in range(self.nlayers)[1:-1]: # compute signals and activations of nodes in every layer
                signals = np.dot(actsWithBias[l-1],self.weights[l-1]) # signals of layer l nodes
                if self.hiddenact == 'tanh':
                    actsWithBias[l][:,1:] = np.tanh(signals) # tanh activation functions
                else:
                    actsWithBias[l][:,1:] = 1 / (1 + np.exp(-signals)) # Sigmoid activations of layer l nodes
            # compute signal and activations of output nodes (always sigmoid)
            l += 1
            signals = np.dot(actsWithBias[l-1],self.weights[l-1])
            actsWithBias[l][:,1:] = 1 / (1 + np.exp(-signals))

            # Backward pass
            outputs = actsWithBias[self.nlayers - 1][:,1:]
            deltas[self.nlayers - 1] = (outputs - targets)*outputs*(1-outputs) / self.npatterns
            # ... accumulate deltas for hidden layers
            for j in range(self.nlayers)[1:-1][::-1]: # traverse weights backwards
                if self.hiddenact == "tanh":
                    deltas[j] = np.dot(deltas[j + 1], self.weights[j][1:,].T) \
                        * (1-actsWithBias[j][:,1:]**2)
                else:
                    deltas[j] = np.dot(deltas[j + 1], self.weights[j][1:,].T) \
                        * actsWithBias[j][:,1:] * (1-actsWithBias[j][:,1:])

            # Update weights
            for i,wt in enumerate(self.weights):
                self.updates[i] = eta*np.dot(actsWithBias[i].T,deltas[i+1]) + momentum*self.updates[i]
                self.weights[i] -= self.updates[i]
                self.weights[i] *= self.wtdecay

class mlpRprop(mlpbatch):
    def __init__(self, inputs, targets, hiddenlayers,hiddenact = 'sigmoid',wtdecay=1,wtbacktrack='errincrease',seed=None):
        mlp.__init__(self, inputs, targets, hiddenlayers,hiddenact,seed)
        self.wtdecay = 1
        self.gradients = [np.zeros((self.arch[i]+1,self.arch[i+1])) for i in range(self.nlayers - 1)]
        self.stepsizes = [np.ones((self.arch[i]+1,self.arch[i+1]))*0.1 for i in range(self.nlayers - 1)]
        self.wtbacktrack = wtbacktrack

    def simpletrain(self,inputs,targets,eta=0.25,T=int(1e4),deltamax=50.0,deltamin=1e-6,etaplus=1.2,etaminus=0.5):
        """
        This method trains the NN for a specified number of iterations.
        """
        oldtrainError = -1.0
        currentGrads = [np.zeros((self.arch[i]+1,self.arch[i+1])) for i in range(self.nlayers - 1)]
        for ite in range(T):
            actsWithBias = [np.concatenate((np.ones((self.npatterns, 1)), inputs), axis=1)]
            actsWithBias += [np.ones((self.npatterns, self.arch[i] + 1)) for i in range(1,self.nlayers)] # note redundant bias included for output layer
            deltas = [np.ones((self.npatterns, self.arch[i])) for i in range(self.nlayers)]

            # Forward pass
            for l in range(self.nlayers)[1:-1]: # compute signals and activations of nodes in every layer
                signals = np.dot(actsWithBias[l-1],self.weights[l-1]) # signals of layer l nodes
                if self.hiddenact == 'tanh':
                    actsWithBias[l][:,1:] = np.tanh(signals) # tanh activation functions
                else:
                    actsWithBias[l][:,1:] = 1 / (1 + np.exp(-signals)) # Sigmoid activations of layer l nodes
            # compute signal and activations of output nodes (always sigmoid)
            l += 1
            signals = np.dot(actsWithBias[l-1],self.weights[l-1])
            actsWithBias[l][:,1:] = 1 / (1 + np.exp(-signals))

            # Backward pass
            outputs = actsWithBias[self.nlayers - 1][:,1:]
            deltas[self.nlayers - 1] = (outputs - targets)*outputs*(1-outputs) / self.npatterns
            # ... accumulate deltas for hidden layers
            for j in range(self.nlayers)[1:-1][::-1]: # traverse weights backwards
                if self.hiddenact == "tanh":
                    deltas[j] = np.dot(deltas[j + 1], self.weights[j][1:,].T) \
                        * (1-actsWithBias[j][:,1:]**2)
                else:
                    deltas[j] = np.dot(deltas[j + 1], self.weights[j][1:,].T) \
                        * actsWithBias[j][:,1:] * (1-actsWithBias[j][:,1:])

            # Update stepsizes
            for i in range(self.nlayers - 1):
                currentGrads[i] = np.dot(actsWithBias[i].T,deltas[i+1])
                self.stepsizes[i] = (currentGrads[i]*self.gradients[i] > 0)*np.minimum(deltamax,etaplus*self.stepsizes[i]) \
                            + (currentGrads[i]*self.gradients[i] < 0)*np.maximum(deltamin,etaminus*self.stepsizes[i]) \
                            + (currentGrads[i]*self.gradients[i] == 0)*self.stepsizes[i]

            # Update weights
            trainError = np.sum((outputs-targets)**2) # compute error
            for i,wt in enumerate(self.weights):
                if trainError > oldtrainError or self.wtbacktrack == 'basic': # extra condition to specify when to weight-backtrack
                    self.updates[i] = (currentGrads[i]*self.gradients[i] < 0)*(self.updates[i]) + \
                                (currentGrads[i]*self.gradients[i] >= 0)*(np.sign(currentGrads[i])*self.stepsizes[i])
                else:
                    self.updates[i] = np.sign(currentGrads[i])*self.stepsizes[i]
                self.weights[i] -= self.updates[i]
                self.weights[i] *= self.wtdecay
                # Update gradients...note gradients for those weights whose gradient changed directions is set to zero
                self.gradients[i] = currentGrads[i]*(currentGrads[i]*self.gradients[i] >= 0)
            oldtrainError = trainError

class mlpQprop(mlpbatch):
    def __init__(self, inputs, targets, hiddenlayers,hiddenact = 'sigmoid',wtdecay=1,seed=None):
        mlp.__init__(self, inputs, targets, hiddenlayers,hiddenact,seed)
        self.wtdecay = 1
        self.gradients = [np.zeros((self.arch[i]+1,self.arch[i+1])) for i in range(self.nlayers - 1)]
        self.growth = [np.zeros((self.arch[i]+1,self.arch[i+1])) for i in range(self.nlayers - 1)]

    def simpletrain(self,inputs,targets,eta=0.25,mu=1.75,T=int(1e4)):
        """
        This method trains the NN for a specified number of iterations.
        """
        currentGrads = [np.zeros((self.arch[i]+1,self.arch[i+1])) for i in range(self.nlayers - 1)]
        for ite in range(T):
            actsWithBias = [np.concatenate((np.ones((self.npatterns, 1)), inputs), axis=1)]
            actsWithBias += [np.ones((self.npatterns, self.arch[i] + 1)) for i in range(1,self.nlayers)] # note redundant bias included for output layer
            deltas = [np.ones((self.npatterns, self.arch[i])) for i in range(self.nlayers)]

            # Forward pass
            for l in range(self.nlayers)[1:-1]: # compute signals and activations of nodes in every layer
                signals = np.dot(actsWithBias[l-1],self.weights[l-1]) # signals of layer l nodes
                if self.hiddenact == 'tanh':
                    actsWithBias[l][:,1:] = np.tanh(signals) # tanh activation functions
                else:
                    actsWithBias[l][:,1:] = 1 / (1 + np.exp(-signals)) # Sigmoid activations of layer l nodes
            # compute signal and activations of output nodes (always sigmoid)
            l += 1
            signals = np.dot(actsWithBias[l-1],self.weights[l-1])
            actsWithBias[l][:,1:] = 1 / (1 + np.exp(-signals))

            # Backward pass
            outputs = actsWithBias[self.nlayers - 1][:,1:]
            deltas[self.nlayers - 1] = (outputs - targets)*outputs*(1-outputs) / self.npatterns
            # ... accumulate deltas for hidden layers
            for j in range(self.nlayers)[1:-1][::-1]: # traverse weights backwards
                if self.hiddenact == "tanh":
                    deltas[j] = np.dot(deltas[j + 1], self.weights[j][1:,].T) \
                        * (1-actsWithBias[j][:,1:]**2)
                else:
                    deltas[j] = np.dot(deltas[j + 1], self.weights[j][1:,].T) \
                        * actsWithBias[j][:,1:] * (1-actsWithBias[j][:,1:])

            # Compute current gradients
            for i in range(self.nlayers - 1):
                currentGrads[i] = np.dot(actsWithBias[i].T,deltas[i+1])

            # Update weights
            for i,wt in enumerate(self.weights):
                self.growth[i] = currentGrads[i] / (self.gradients[i] - currentGrads[i])
                self.growth[i] = np.where(((currentGrads[i]*self.gradients[i] > 0) & (np.abs(currentGrads[i]) - np.abs(self.gradients[i]) > 0)), -self.growth[i], self.growth[i])
                # ((currentGrads[i]*self.gradients[i] > 0) & (np.abs(currentGrads[i]) - np.abs(self.gradients[i]) > 0))*(-self.growth[i]) \
                #                     + (~((currentGrads[i]*self.gradients[i] > 0) & (np.abs(currentGrads[i]) - np.abs(self.gradients[i]) > 0)))*(self.growth[i])
                self.growth[i] = np.where(np.abs(self.growth[i]) > mu, np.sign(self.growth[i])*mu, self.growth[i])
                # (self.growth[i] < -mu)*(-mu) + (self.growth[i] > mu)*mu + (self.growth[i] <= mu & self.growth >= -mu)*self.growth[i] # cap off growth rates
                self.updates[i] = self.growth[i]*self.updates[i] \
                                        + (currentGrads[i]*self.gradients[i] >= 0)*eta*currentGrads[i]
                self.weights[i] -= self.updates[i]
                self.weights[i] *= self.wtdecay
                self.gradients[i] = currentGrads[i] # update gradients
