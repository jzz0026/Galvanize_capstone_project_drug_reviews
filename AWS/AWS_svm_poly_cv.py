import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
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
from sklearn.svm import SVC


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

df = pd.read_csv('drugsCom_raw/drugsComTrain_raw.tsv',sep='\t',index_col=0)
df['date'] = pd.to_datetime(df['date'])
df = rm_sym(df)


def add_sample_svm(n):
    df_tem2 = df.sample(n)
    #df_tem2.groupby('rating_cate').size() / df_tem2.groupby('rating_cate').size().sum()

    ## Generate table of words with their counts
    con_vec = TfidfVectorizer(stop_words='english',tokenizer=tokenize)
    X_train = con_vec.fit_transform(df_tem2['review'])
    y_train = df_tem2['rating_cate']

    ## test set
#     test = pd.read_csv("drugsCom_raw/drugsComTest_raw.tsv",sep='\t', index_col=0)
#     test = rm_sym(test)
#     X_test = con_vec.transform(test['review'])
#     y_test = test['rating_cate']


    #pickle.dump(con_vec, open("svm_lin_20000_tfidf.sav", 'wb'))

    svm_rbf = SVC(kernel='rbf')
    svm_rbf_cv_score = cross_val_score(svm_rbf,X_train,y_train,scoring='accuracy',cv=3,n_jobs=-1)
    with open("svm_rbf_"+str(n)+"_cv.txt", 'w') as outfile:
        outfile.write(str(svm_rbf_cv_score))
        
for n in [5000,10000,20000,40000,80000]:
    add_sample_svm(n)
