import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_chisquare
from scipy.stats import chisquare
import pickle
from collections import defaultdict
from statsmodels.stats.multitest import fdrcorrection_twostage

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import string
import re

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score , f1_score

## remove special symbol
def rm_sym(df):
    df['review'] = df['review'].str.replace("&#039;",'\'')
    df['review'].head()
    df['rating_cate'] = ''
    df.loc[df['rating'] >= 7,'rating_cate'] = 'high'
    df.loc[df['rating'] <= 4,'rating_cate'] = 'low'
    df.loc[(df['rating'] > 4) & (df['rating'] < 7),'rating_cate'] = 'medium'
    return df

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

if __name__ == "__main__":
    
    df = pd.read_csv('drugsCom_raw/drugsComTrain_raw.tsv',sep='\t',index_col=0)
    df_tem = rm_sym(df)
    test = pd.read_csv("drugsCom_raw/drugsComTest_raw.tsv",sep='\t', index_col=0)
    test = rm_sym(test)

    ## Generate table of words with their counts
    ## TfidfVectorizer transform train and test
    con_vec = TfidfVectorizer(stop_words='english',tokenizer=tokenize)
    X_train = con_vec.fit_transform(df_tem['review'])
    y_train = df_tem['rating_cate']
    X_test = con_vec.transform(test['review'])
    y_test = test['rating_cate']

    ## modeling using RandomForestClassifier 
    rfc = RandomForestClassifier(n_estimators=400,n_jobs=-1,oob_score=True)
    rfc.fit(X_train,y_train)

    ## accuracy
    y_test_rfc_predict = rfc.predict(X_test)
    acc = accuracy_score(y_test,y_test_rfc_predict)
    print(acc)

    ## save the built model 
    filename = 'finalized_rfc_model.sav'
    pickle.dump(rfc, open(filename, 'wb'))

    