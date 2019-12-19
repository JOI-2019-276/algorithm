#!/usr/bin/env python
# -*- coding: utf-8 -*-
__version__ = "$Revision: 8 $"

import time
import sys
import random
import numpy as np

__all__ = ["HiNMF"]

class HiNMF:

	_VINFO = 'v0.1'
	EPS = 10**-12
	def __init__(self, data, indicate_data, pro_w, lam=0.5, num_bases=10, niter=100):
		
		self.data = data	
		self.B    = indicate_data	
		self._num_bases    = num_bases
		self._niter = niter
		self._pro_w = pro_w
		self.lam =lam
		self.ferr   = np.zeros(self._niter)
		(self._data_dimension, self._num_samples) = self.data.shape			

	def initialization(self):

		self.H = np.random.random((self._num_bases, self._num_samples))
		self.W = self._pro_w
		self.M = np.random.random( self.B.shape )
	
	def frobenius_norm(self):

		err = np.sqrt( np.sum( self.data[:,:] - self.W.dot(self.B*self.M).dot(self.H)  )**2 )
		return err
		
	def updateH(self):
		WW  	= self.W.dot(self.B*self.M)
		H2 		= np.dot(np.dot(WW.T, WW), self.H)+10**-9
		self.H *= np.dot(WW.T, self.data)
		self.H /= H2								
	
	def updateM(self):
		M2 		= self.B*(self.W.T).dot(self.W).dot(self.B*self.M).dot(self.H).dot(self.H.T)+ 10**-9
		self.M *= self.B*(self.W.T).dot(self.data).dot(self.H.T)
		self.M /= (M2+self.lam)

	def converged(self, i):
		derr = np.abs(self.ferr[i] - self.ferr[i-1])/self._num_samples
		print "converged--",derr
		if derr < self.EPS:
			return True
		else:
			return False
		
	def factorize(self):
			
		for i in xrange(self._niter):

			self.updateM()									
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
