# -*- coding: utf-8 -*-
import os, sys 
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

class generate_matrix_by_year():
    def __init__(self, paper_id, word_id, paper_year, year, paper_word):
        self.word_id    = word_id
        self.paper_word = paper_word
        self.paper_year = paper_year
        self.year   = year

    def generate_list_id_dict(self,x):
        return dict(zip(x,range(len(x))))

    def generate(self, tfidf=True):
        paper_list=[paper for paper in self.paper_year.keys() if int(self.paper_year[paper])==self.year]
        matrix=np.zeros((len(paper_list),len(self.word_id)),dtype=np.float64)
        self.paper_id=self.generate_list_id_dict(paper_list)
        for x in paper_list:
            for y in self.paper_word[x]:
                if self.word_id.has_key(y):
                    matrix[self.paper_id[x]][self.word_id[y]]+=1.0
        if tfidf:
            transformer = TfidfTransformer()
            tfidf = transformer.fit_transform(matrix)
            matrix=tfidf.toarray()

        return (matrix,self.paper_id)
