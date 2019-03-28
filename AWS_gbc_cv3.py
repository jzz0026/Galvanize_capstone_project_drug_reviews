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
df_tem2 = df.sample(20000)
#df_tem2.groupby('rating_cate').size() / df_tem2.groupby('rating_cate').size().sum()

## Generate table of words with their counts
con_vec = TfidfVectorizer(stop_words='english',tokenizer=tokenize)
X_train = con_vec.fit_transform(df_tem2['review'])
y_train = df_tem2['rating_cate']

## test set
test = pd.read_csv("drugsCom_raw/drugsComTest_raw.tsv",sep='\t', index_col=0)
test = rm_sym(test)
X_test = con_vec.transform(test['review'])
y_test = test['rating_cate']


pickle.dump(con_vec, open("gbc_20000_600_tfidf.sav", 'wb'))


gbc = GradientBoostingClassifier(n_estimators=600)
gbc.fit(X_train,y_train)
y_test_predict = gbc.predict(X_test)
acc = accuracy_score(y_test,y_test_predict)
with open("gbc_20000_600_accuracy.txt", 'w') as outfile:
    outfile.write(str(acc))

pickle.dump(gbc, open("gbc_20000_600_gbc.sav", 'wb'))
