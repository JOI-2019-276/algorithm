# -*- coding: utf-8 -*-
import os, sys 
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

class generate_matrix():
    def __init__(self, xdata, ydata, data):
        self.xdata = xdata
        self.ydata = ydata
        self.data  = data

    def generate(self,tfidf=True):
        matrix=np.zeros((len(self.xdata),len(self.ydata)),dtype=np.float64)
        for x in self.xdata.keys():
            for y in self.data[x]:
                if self.ydata.has_key(y):
                    matrix[self.xdata[x]][self.ydata[y]]+=1.0
        if tfidf:
            transformer = TfidfTransformer()
            tfidf = transformer.fit_transform(matrix)
            matrix=tfidf.toarray()

        return matrix
