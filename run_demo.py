#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os, sys 
import time
import json
import nltk
import copy
import numpy as np
from util           import handle_file
from util           import handle_dict
from util 			import generate_matrix_by_year
from collections    import Counter
from model          import NMF
from model          import streamNMF
from model          import HiNMF

paper_word  = handle_file().open_json("dataset/document_word.json")
stem_word	= handle_file().open_json("dataset/stem_word.json")
word_id     = handle_file().open_json("dataset/word_id.json")
paper_id 	= handle_file().open_json("dataset/paper_id.json")
paper_year	= handle_file().open_json("dataset/paper_year.json")

def get_top_word(top_k,word_topic,num_bases):
	topic_word_weight = {}
	topic_word={}
	for i in xrange(num_bases):
		topic_word_weight[i]={}
		for w in word_id.keys():
			topic_word_weight[i][stem_word[w]]=word_topic[i][word_id[w]]
		topic_word[i]=handle_dict().sort_dict_by_values(topic_word_weight[i],descent=True)[0:top_k]
	return topic_word

year_list = [2016,2017,2018] #In this demo, we test three year's data

num_bases  = 100  #the topic number in the third layer 
num_bases2 = 20	  #the topic number in the second layer
num_bases3 = 5    #the topic number in the first layer

##generate the third-layer topics
matrix,paper_id = generate_matrix_by_year(paper_id=paper_id,word_id=word_id,paper_year=paper_year,year=year_list[0],paper_word=paper_word).generate(tfidf=True)

nmf =NMF(data=matrix.T, num_bases=num_bases, niter=100)
nmf.initialization()
nmf.factorize()
word_topic  = nmf.W.T

#get topic top-words 
topic_word=get_top_word(top_k=50,word_topic=word_topic,num_bases=num_bases)
#save the result
handle_file().save_json("result/tree4/topic_word_weight_"+str(year_list[0])+".json" ,topic_word)
np.save("result/tree4/word_topic_"+str(year_list[0]),word_topic)
matrix_W=word_topic

##generate the second-layer topics
B 	= np.ones((num_bases,num_bases2))
nmf  =  HiNMF(data=matrix.T,indicate_data=B, pro_w=matrix_W.T, lam=0.3, num_bases=num_bases2, niter=100)
nmf.initialization()
nmf.factorize()
word_topic  = nmf.W.dot(nmf.M).T
topic_word=get_top_word(top_k=50,word_topic=word_topic,num_bases=num_bases2)
handle_file().save_json("result/tree3/topic_word_weight_"+str(year_list[0])+".json" ,topic_word)
np.save("result/tree3/word_topic_"+str(year_list[0]),word_topic)
matrix_W=word_topic

##generate the first-layer topics
B 	= np.ones((num_bases2,num_bases3))
nmf  =  HiNMF(data=matrix.T,indicate_data=B, pro_w=matrix_W.T, lam=0.3, num_bases=num_bases3, niter=100)
nmf.initialization()
nmf.factorize()
word_topic  = nmf.W.dot(nmf.M).T
topic_word=get_top_word(top_k=50,word_topic=word_topic,num_bases=num_bases3)
handle_file().save_json("result/tree2/topic_word_weight_"+str(year_list[0])+".json" ,topic_word)
np.save("result/tree2/word_topic_"+str(year_list[0]),word_topic)


for date in year_list[1:]:
	##generate the third-layer topics
	matrix,paper_id=generate_matrix_by_year(paper_id=paper_id,word_id=word_id,paper_year=paper_year,year=date,paper_word=paper_word).generate(tfidf=True)
	nmf =streamNMF(data=matrix.T, pre_word_topic=word_topic.T, num_bases=num_bases, niter=100)
	nmf.initialization()
	nmf.factorize()
	word_topic  = nmf.W.T
	topic_word=get_top_word(top_k=50,word_topic=word_topic,num_bases=num_bases)
	handle_file().save_json("result/tree4/topic_word_weight_"+str(date)+".json" ,topic_word)
	np.save("result/tree4/word_topic_"+str(date),word_topic)
	matrix_W=word_topic

	##generate the second-layer topics
	B 	= np.ones((num_bases,num_bases2))
	nmf  =  HiNMF(data=matrix.T,indicate_data=B, pro_w=matrix_W.T, lam=0.3, num_bases=num_bases2, niter=100)
	nmf.initialization()
	nmf.factorize()
	word_topic  = nmf.W.dot(nmf.M).T
	topic_word=get_top_word(top_k=50,word_topic=word_topic,num_bases=num_bases2)
	handle_file().save_json("result/tree3/topic_word_weight_"+str(year_list[0])+".json" ,topic_word)
	np.save("result/tree3/word_topic_"+str(year_list[0]),word_topic)
	matrix_W=word_topic

	##generate the first-layer topics
	B 	= np.ones((num_bases2,num_bases3))
	nmf  =  HiNMF(data=matrix.T,indicate_data=B, pro_w=matrix_W.T, lam=0.3, num_bases=num_bases3, niter=100)
	nmf.initialization()
	nmf.factorize()
	word_topic  = nmf.W.dot(nmf.M).T
	topic_word=get_top_word(top_k=50,word_topic=word_topic,num_bases=num_bases3)
	handle_file().save_json("result/tree2/topic_word_weight_"+str(year_list[0])+".json" ,topic_word)
	np.save("result/tree2/word_topic_"+str(year_list[0]),word_topic)

