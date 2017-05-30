# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:21:34 2017

@author: aneesh.c
"""




import os
os.chdir('C:\Users\\aneesh.c\Downloads')
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

#file_frame = pd.read_csv('train.csv')

#def compute_score(row):
#    "Computes Jaccard similarity score between question1 and question2"
#    q1_list = []
#    q2_list = []
#    for word in str(row['question1']).lower().split():
#        if word not in stops:
#            q1_list.append(word)
#    for word in str(row['question2']).lower().split():
#        if word not in stops:
#            q2_list.append(word)
#    num = len(set(q1_list) & set(q2_list))
#    deno = len(set(q1_list) | set(q2_list))
#    if len(q1_list) == 0 or len(q2_list) == 0:
#        return 0
#    score = float(num)/float(deno)
#    return score
#
#
#score_list = []
#for index, row in file_frame.iterrows():
#    score = compute_score(row)
#    score_list.append(score)
#
#file_frame['jaccard_score'] = score_list
#
#A = np.array([score_list, np.ones(len(score_list))])
#
#w = np.linalg.lstsq(A.T,file_frame['is_duplicate'])[0]
#
#
#test_frame = pd.read_csv('test.csv')
#
#score_list = []
#for index, row in test_frame.iterrows():
#    score = compute_score(row)
#    score_list.append(score)
#
#test_frame['jaccard_score'] = score_list
#
#test_frame['is_duplicate'] = test_frame['jaccard_score']*w[0]+w[1]
#
#
#sub = pd.DataFrame()
#sub['test_id'] = test_frame['test_id']
#sub['is_duplicate'] = test_frame['is_duplicate']
#sub.to_csv('least_square_submission.csv', index=False)



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


def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    print 'list', M
    M = np.array(M)
    print 'Array', M
    v = M.sum(axis=0)
    print 'sum',v
    return v / np.sqrt((v ** 2).sum())


data = pd.read_csv('C:/Users/aneesh.c/Downloads/train/train.csv')
data = data.drop(['id', 'qid1', 'qid2'], axis=1)


data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
data['diff_len'] = data.len_q1 - data.len_q2
data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/aneesh.c/Downloads/train/GoogleNews-vectors-negative300.bin.gz', binary=True)
data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)


norm_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)
data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

question1_vectors = np.zeros((data.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(data.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((data.shape[0], 300))
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

data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

cPickle.dump(question1_vectors, open('data/q1_w2v.pkl', 'wb'), -1)
cPickle.dump(question2_vectors, open('data/q2_w2v.pkl', 'wb'), -1)

data.to_csv('D:\sheduler\quora_features.csv', index=False)

data = pd.read_csv('D:\sheduler\quora_features.csv')
from sklearn.model_selection import train_test_split
features = data.columns.drop(['question1', 'question2', 'is_duplicate'])
X = sklearn.preprocessing.normalize(data[features], norm='l2', axis=1, copy=True, 
                                    return_norm=False)
x_train,x_test,y_train,y_test = train_test_split(X, data['is_duplicate'],
                                                 random_state=1)
result_cols = ["Classifier", "Accuracy"]
result_frame = pd.DataFrame(columns=result_cols)

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    MultinomialNB()]


for clf in classifiers:
    name = clf.__class__.__name__

    clf.fit(x_train, y_train)
    
    predicted = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test,predicted)
    print (name+' accuracy = '+str(acc*100)+'%')
    acc_field = pd.DataFrame([[name, acc*100]], columns=result_cols)
    result_frame = result_frame.append(acc_field)
    
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=result_frame, color="r")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()



