#!/usr/bin/env python
# -*- coding: utf-8 -*-
__version__ = "$Revision: 8 $"

import time
import sys
import random
import numpy as np

__all__ = ["streamNMF"]

class streamNMF:

	_VINFO = 'v0.1'
	EPS = 10**-12
	def __init__(self, data, pre_word_topic, num_bases=40, niter=200):
		self.data = data	
		self._num_bases = num_bases
		self._niter = niter
		self.ferr = np.zeros(self._niter)
		(self._data_dimension, self._num_samples) = self.data.shape	
		self.W = pre_word_topic/4.0	

	def initialization(self):

		self.H = np.random.random((self._num_bases, self._num_samples))

	def frobenius_norm(self):
		err = np.sqrt( np.sum((self.data - np.dot(self.W, self.H))**2 ))
		return err
		
	def updateH(self):
		H2 = np.dot(np.dot(self.W.T, self.W), self.H)+10**-9
		self.H *= np.dot(self.W.T, self.data)
		self.H /= H2				
	
	def updateW(self):
		W2 = np.dot(np.dot(self.W, self.H), self.H.T) + 10**-9
		self.W *= np.dot(self.data[:,:], self.H.T)
		self.W /= W2

	def converged(self, i):
		derr = np.abs(self.ferr[i] - self.ferr[i-1])/self._num_samples
		print "converged--",derr
		if derr < self.EPS:
			return True
		else:
			return False

	def factorize(self):
			
		for i in xrange(self._niter):

			self.updateW()									
			self.updateH()

			self.ferr[i] = self.frobenius_norm()		
											
			print ('iteration ' + str(i+1) + '/' + str(self._niter) + ' Fro:' + str(self.ferr[i]))

			if i > 1:
				if self.converged(i):	
					print "i---"	
					self.ferr = self.ferr[:i]			
					break

if __name__ == '__main__':
	pass
