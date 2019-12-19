# -*- coding: utf-8 -*-
import random
import os, sys 
import re
import codecs
import json
import copy
import nltk
from nltk.corpus import BracketParseCorpusReader

class handle_file():

    def __init__(self):
        pass

    def save_file(self, path, data):
        f = open (path, 'w' ) 
        f.write(data)
        f.close() 

    def save_json(self, path,data):
        with open(path,'w') as f:
            json.dump(data,f,indent=4)      

    def open_json(self,path):
        with open(path,'r') as f:
            dic = json.load(f)
        return dic
    	
    def open_file(self,path):
        f = open(path,'r')
        list_read_line=[]
        for i in f:
            if len(i.strip())>0:
                list_read_line.append(i.strip())
        return list_read_line

    def open_flod(self, root_path, file_type ):
        ptb         = BracketParseCorpusReader(root_path, file_type)
        files_list  = ptb.fileids()
        files_path  = []
        for f in files_list:
            files_path.append(os.path.join(root_path,f))
        return (files_path,files_list)

    def save_sparse_csr(filename, array):
        np.savez(filename, data = array.data, indices=array.indices, indptr =array.indptr, shape=array.shape)

    def load_sparse_csr(filename):
        loader = np.load(filename)
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])


class handle_string:

    def __init__(self):
        pass

    def print_json(self, dic):
    	print(json.dumps(dic, indent=2))

    def find_string_part(self,reg):
    	pass

    def match_string_part(self,reg):
    	pass

    def determin_string(self,reg):
    	pass

    def split_string(self,reg,string):
        return re.split(reg,string)

    def asciiOnly(self,string):
        return "".join([char if ord(char) < 128 else "" for char in string])

    def isPunctuation(self,word):
        return word in [".", "?", "!", ",", "\"", ":", ";", "'", "-",".",","]


class handle_matrix:

    def save_sparse_csr(self,filename,array):
        np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

    def load_sparse_csr(self,filename):
        loader = np.load(filename)
        return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                             shape = loader['shape'])

class handle_dict:

    def __init__(self):
        pass
        
    def generate_id_list_dict(self,x):
        lis=xrange(len(x))
        dic={}
        for l in lis:
            dic[l]=x[l]
        return dic
        
    def generate_list_id_dict(self,x):
        return dict(zip(x,range(len(x))))

    def sort_list_index(self,s,descent=False):
        return sorted(range(len(s)), reverse=descent, key=lambda k:s[k])

    def sort_dict_by_values(self,dict,descent=False):
        return sorted(dict.keys(), reverse=descent, key=lambda k:dict[k])
