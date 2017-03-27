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
    def __init__(self,featdim,mapdim,wrap=False):
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
        self.weights = (-np.ones(self.nnodes*self.featdim)
                                + 2*np.random.rand(self.nnodes*self.featdim)).reshape((self.nnodes, self.featdim))

    def neighbourhood(self, center, x, t, T, func = "Gaussian"):
        """
        This method evaluates the neighbourhood function (i.e., radial basis function).
        A node x that is close to the reference node center will have a greater
        evaluation and thus will have a bigger weight update.
        """
        assert func in ['Gaussian','MexHat']
        if func == "Gaussian": # a Gaussian with variance = T/(2t)
            return np.exp(-t/T * self.mapdistances[x][center]**2)

    def train(self, data, neighbourhood, eta, alpha, T):
        """
        data = a np.ndarray of observations to train the som
        neighbourhood = neighbourhood function, default Gaussian.
        eta = learning rate
        alpha = learning rate decay
        T = max number of iterations
        """
