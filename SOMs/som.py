#!/bin/python

import numpy as np

"""
Source code for COSC 4P80 assignment 2
"""

__author__ = "Val Andrei Fajardo"

def euclidean(x,y):
    """
    This function returns the Euclidean distance between two vectors x and y -- which
    are both of np.array().
    """
    return np.sqrt(((x-y)**2).sum())

class som:
    def __init__(self,featdim,mapdim,wtsintvl,wrap=False):
        """
        featdim = dimension of the input space (i.e., number of features)
        mapdim = dimension of the map space
        nnodes = number of nodes in the map (i.e., the product of the dimesions in mapdim)
        mapc = provides the coordinates of each node (by default, all nodes are place in the
              positive quadrant of the Cartesian space), and ordered lexicographically
        mapdistances = a matrix storing the distances between any two nodes in the map
        weights = weights matrix for each node
        """
        self.featdim = featdim
        self.wrap = wrap
        self.mapdim = mapdim
        self.mapwidth, self.maplength = mapdim
        self.nnodes = np.prod(mapdim)
        self.mapc = np.array([[i,j] for i in range(self.mapdim[0]) for j in range(self.mapdim[1])])
        # compute distances between nodes
        self.mapdistances = np.zeros((self.nnodes,self.nnodes))
        if wrap: # if edges are wrapped make copies of lattice and find distances by taking minimum
            generators = [np.array([i,j]) for i in range(-1,2) for j in range(-1,2)]
            copies = {i: generators[i]*(self.mapdim) + self.mapc for i in range(len(generators))}
            for i in range(self.nnodes):
                for j in range(i+1, self.nnodes):
                    self.mapdistances[i][j] = min(euclidean(self.mapc[i],copy[j]) for copy in copies.values())
                    self.mapdistances[j][i] = self.mapdistances[i][j]
        else:
            for i in range(self.nnodes):
                for j in range(i+1, self.nnodes):
                    self.mapdistances[i][j] = euclidean(self.mapc[i],self.mapc[j])
                    self.mapdistances[j][i] = self.mapdistances[i][j]
        # initialize weights using UNIF[-1,1]
        self.weights = (np.ones(self.nnodes*self.featdim)*wtsintvl[0]
                                + (wtsintvl[1] - wtsintvl[0])*np.random.rand(self.nnodes*self.featdim)).reshape((self.nnodes, self.featdim))

    def neighbourhood(self, t, T, nb, neighborfunc = 'Gaussian'):
        """
        This function evaulates the neighbourhood function whose center node is nb
        for all nodes in the map. The output is stored in a nnodes x 1 column vector.
        """
        if neighborfunc == "Gaussian":
            return (np.exp(-t/T * self.mapdistances[:,nb]**2))[:,None]
        elif neighborfunc == "MexHat":
            return (4*(np.exp(-t/T * self.mapdistances[:,nb]**2)) - (np.exp(-2*t/T * self.mapdistances[:,nb]**2)))[:,None]
        else:
            raise ValueError("Must specify either 'Gaussian' or 'Mexhat' for neighborfunc")


    def train(self, data, eta, alpha, T, neighborfunc = 'Gaussian'):
        """
        This method trains the SOM. Arguments are as follows:

        eta = initial learning rate
        alpha = learning decay rate
        T = max number of iterations
        neighborfunc = neighbourhood function, default is Gaussian.
        """
        assert alpha < 1 and alpha > 0
        for t in range(1,T):
            for x in data:
                # find the winning neuron (i.e., closes to x in weight space)
                h = [(euclidean(x, self.weights[j]),j) for j in range(self.nnodes)]
                nb = max(h)[1] # index of closest node

                # update weight vectors
                eta = alpha*eta**(t/T)
                self.weights += eta*self.neighbourhood(t,T,nb,neighborfunc)*(x - self.weights)
