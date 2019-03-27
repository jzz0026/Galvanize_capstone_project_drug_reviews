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
from sklearn.ensemble import GradientBoostingClassifier

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
df_tem2 = df.sample(40000)
#df_tem2.groupby('rating_cate').size() / df_tem2.groupby('rating_cate').size().sum()

## Generate table of words with their counts
con_vec = TfidfVectorizer(stop_words='english',tokenizer=tokenize)
X_train = con_vec.fit_transform(df_tem2['review'])
#target_3 = pd.get_dummies(df_tem['rating_cate'])
#X_train = pd.DataFrame(X_train.toarray(),columns=con_vec.get_feature_names())
y_train = df_tem2['rating_cate']

pickle.dump(con_vec, open("gbc_40000_tfidf.sav", 'wb'))

for i in [400,600,800]:
    gbc = GradientBoostingClassifier(n_estimators=i)
    gbc_cv_score = cross_val_score(gbc,X_train,y_train,scoring='accuracy',cv=3,n_jobs=-1)
    pickle.dump(gbc, open("gbc_40000_"+str(i)+"_tfidf.sav", 'wb'))
    with open("gbc_40000_"+str(i)+"_cv.txt", 'w') as outfile:
        outfile.write(str(gbc_cv_score))
