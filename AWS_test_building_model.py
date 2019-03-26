
# coding: utf-8

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score 
import pickle

## only need to remove punctuation and stemize
stemmer = SnowballStemmer('english')

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

## remove special symbol
def rm_sym(df):
    df['review'] = df['review'].str.replace("&#039;",'\'')
    df['review'].head()
    df['rating_cate'] = ''
    df.loc[df['rating'] >= 7,'rating_cate'] = 'high'
    df.loc[df['rating'] <= 4,'rating_cate'] = 'low'
    df.loc[(df['rating'] > 4) & (df['rating'] < 7),'rating_cate'] = 'medium'
    return df

def build_lr_subsample(n_sample):
    df = pd.read_csv('drugsCom_raw/drugsComTrain_raw.tsv',sep='\t',index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    df = rm_sym(df)
    df = df.sample(n_sample)

    ## Generate table of words with their counts
    con_vec = TfidfVectorizer(stop_words='english',tokenizer=tokenize)
    X_train = con_vec.fit_transform(df['review'])
    #target_3 = pd.get_dummies(df_tem['rating_cate'])
    X_train = pd.DataFrame(X_train.toarray(),columns=con_vec.get_feature_names())
    y_train = df['rating_cate']


    test = pd.read_csv("drugsCom_raw/drugsComTest_raw.tsv",sep='\t', index_col=0)
    test = rm_sym(test)
    X_test = con_vec.transform(test['review'])
    X_test = pd.DataFrame(X_test.toarray(),columns=con_vec.get_feature_names())
    y_test = test['rating_cate']

    ## Buiding model
    lr = LogisticRegression(penalty='l1',multi_class='auto',solver='saga',n_jobs=-1)
    lr.fit(X_train,y_train)

    y_test_predict = lr.predict(X_test)

    accu_score = accuracy_score(y_test,y_test_predict)

    with open(str(n_sample) + "_accuracy_score_test.txt", 'w') as outfile:
        outfile.write(str(accu_score))
        
    # save the model to disk

    pickle.dump(con_vec, open(str(n_sample)+"_tfidf.sav", 'wb'))
    pickle.dump(lr, open(str(n_sample)+"_lr.sav", 'wb'))

for i in [1000,5000,10000,20000,40000,50000]:
    build_lr_subsample(1000)

