# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:28:13 2017

@author: aneesh.c
"""

import os
import pandas as pd
import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
stop_words = stopwords.words('english')
os.chdir('C:/Users/aneesh.c/Downloads/rdany_conversations_2016-03-01')
import MySQLdb

#frame = pd.read_csv('rdany_conversations_2016-03-01.csv', usecols = ['source',
#                                                                     'text'])
#frame = pd.read_csv('C:/Users/aneesh.c/Desktop/conversations.csv', usecols = ['user_input',
#                                                                     'bot_response'])

db = MySQLdb.connect("192.168.0.28","user","kreara@1","ggc" )
query = 'SELECT user_input, robot_output from ggc.ayur'
table = pd.read_sql(query, db)
table.dropna(inplace=True)

#frame = frame[frame.text!='[START]']
#frame["text"] =  [s.encode('ascii', 'ignore').strip()
#               for s in frame.text.str.decode('unicode_escape')]
#frame = frame.dropna()
#frame = frame[frame.text!='']
#df = frame[frame.source =='human']
#df2 = frame[frame.source =='robot']




def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)




def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    #words = [w for w in words if not w in stop_words]
    #words = [w for w in words if w.isalpha()]
    M = []
    #print words
    for w in words:
        try:
            M.append(model[w])
        except Exception as e:
            continue

    M = np.array(M)

    v = M.sum(axis=0)

    return v / np.sqrt((v ** 2).sum())
    
data = frame
sentence_list = data['user_input']
#data.rename(columns=lambda x: x.replace('text', 'question1'), inplace=True)
#sentence_list = data['text']
word_list  = []
for sent in sentence_list:
    word_list.append(word_tokenize(sent))
#    
model = gensim.models.Word2Vec(word_list,min_count=1)
#model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/aneesh.c/Downloads/train/GoogleNews-vectors-negative300.bin.gz', binary=True)

question1_vectors = np.zeros((data.shape[0], 100))
error_count = 0

for i, q in tqdm(enumerate(data.user_input.values)):
    question1_vectors[i, :] = sent2vec(q)
data['question2'] = 'pain on my back side'



question2_vectors  = np.zeros((data.shape[0], 100))
for i, q in tqdm(enumerate(data.question2.values)):
    question2_vectors[i, :] = sent2vec(q)
    
    
    
data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]
l1 = data.user_input.apply(lambda x: len(str(x)))
l2 = data.question2.apply(lambda x: len(str(x)))
data['diff_len'] = l1 - l2                                                      
                                                          
                                                          
data = data.dropna()

j=data[['cosine_distance','cityblock_distance','jaccard_distance',
'canberra_distance','euclidean_distance','minkowski_distance','braycurtis_distance',
'diff_len']]

data['sum'] = j.sum(axis=1)

data = data.sort(['sum'], ascending=[True])

#ind = data.index[0]
#ind = ind+1
#print df2.loc[[ind]]