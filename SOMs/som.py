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
        mapnodes = number of nodes in the map (i.e., the product of the dimesions in mapdim)
        map = provides the coordinates of each node (by default, all nodes are place in the
              positive quadrant of the Cartesian space), and ordered lexicographically
        mapdistances = a matrix storing the distances between any two nodes in the map
        """
        self.featdim = featdim
        self.wrap = wrap
        self.mapdim = mapdim
        self.mapwidth, self.maplength = mapdim
        self.mapnodes = np.prod(mapdim)
        self.mapc = np.array([[i,j] for i in range(self.mapdim[0]) for j in range(self.mapdim[1])])
        self.mapdistances = np.zeros((self.mapnodes,self.mapnodes))
        if wrap: # if edges are wrapped make copies of lattice and find distances by taking minimum
            generators = [np.array([i,j]) for i in range(-1,2) for j in range(-1,2)]
            copies = {i: generators[i]*(self.mapdim) + self.mapc for i in range(len(generators))}
            for i in range(self.mapnodes):
                for j in range(i+1, self.mapnodes):
                    self.mapdistances[i][j] = min(euclidean(self.mapc[i],copy[j]) for copy in copies.values())
                    self.mapdistances[j][i] = self.mapdistances[i][j]
        else:
            for i in range(self.mapnodes):
                for j in range(i+1, self.mapnodes):
                    self.mapdistances[i][j] = euclidean(self.mapc[i],self.mapc[j])
                    self.mapdistances[j][i] = self.mapdistances[i][j]

    def train(self, data, neighbourhood, eta, alpha, T):
        """
        data = a np.ndarray of observations to train the som
        neighbourhood = neighbourhood function, default Gaussian.
        eta = learning rate
        alpha = learning rate decay
        T = max number of iterations
        """
