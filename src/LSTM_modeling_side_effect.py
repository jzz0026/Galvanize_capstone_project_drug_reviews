import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from sklearn.svm import SVC
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

## remove special symbol
def rm_sym(df):
    df['review'] = df['review'].str.replace("&#039;",'\'')
    df['review'].head()
    df['rating_cate'] = ''
    df.loc[df['rating'] >= 7,'rating_cate'] = 'high'
    df.loc[df['rating'] <= 4,'rating_cate'] = 'low'
    df.loc[(df['rating'] > 4) & (df['rating'] < 7),'rating_cate'] = 'medium'
    return df

def clean_text(df_tem3):
    df_tem3['review'] = df_tem3['review'].str.replace("\"","").str.lower()
    df_tem3['review'] = df_tem3['review'].str.replace( r"(\\r)|(\\n)|(\\t)|(\\f)|(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(&#039;)|(\d\s)|(\d)|(\/)","")
    df_tem3['review'] = df_tem3['review'].str.replace("\"","").str.lower()
    df_tem3['review'] = df_tem3['review'].str.replace( r"(\$)|(\-)|(\\)|(\s{2,})"," ")
    df_tem3['review'].sample(1).iloc[0]

    stemmer = SnowballStemmer('english')
    df_tem3['review'] = df_tem3['review'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split(" ")]))
    return df_tem3

if __name__ == "__main__":
    ## clean text
    df = pd.read_csv('drugsCom_raw/drugsComTrain_raw.tsv',sep='\t',index_col=0)#.sample(40000)
    df_tem3 = rm_sym(df)
    test = pd.read_csv("drugsCom_raw/drugsComTest_raw.tsv",sep='\t', index_col=0)
    test = rm_sym(test)
    df_tem3 = clean_text(df_tem3)
    test = clean_text(test)

    # Tokenize the data
    tokenizer = Tokenizer(num_words = MAX_NB_WORDS, 
                        filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                        lower=True, split=' ', char_level=False, 
                        oov_token=None, document_count=0)

    tokenizer.fit_on_texts(df_tem3['review'])
    train_sequences = tokenizer.texts_to_sequences(df_tem3['review'])
    test_sequences = tokenizer.texts_to_sequences(test['review'])

    # truncate and pad input sequences
    X_train = sequence.pad_sequences(train_sequences, maxlen=max_review_length)
    X_test = sequence.pad_sequences(test_sequences, maxlen = max_review_length)

    # transform y to get_dummies
    y_train = pd.get_dummies(df_tem3['rating_cate'])
    y_test = pd.get_dummies(test['rating_cate'])

    ## set parameter for LSTM
    MAX_NB_WORDS = 500
    max_review_length = 500
    EMBEDDING_DIM = 160
    word_index = tokenizer.word_index

    # Split Training & Validation Data
    X_train_t, X_train_val, Y_train_t, y_train_val = train_test_split(X_train, y_train,test_size = 0.2)

    # Set up Model Sequential
    nb_words  = min(MAX_NB_WORDS, len(word_index))
    lstm_out = max_review_length

    model = Sequential()
    model.add(Embedding(nb_words,EMBEDDING_DIM,input_length=max_review_length))
    # model.add(MaxPool1D(pool_size=2))
    model.add(Bidirectional(LSTM(40, return_sequences=True)))
    #model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(40)))
    model.add(Dense(3, activation = 'softmax'))

    ## one-code mutiple categories targets use 'categorical_crossentropy' not 'binary_crossentropy'
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics =['accuracy'])
    print(model.summary())

    # Run LSTM Model
    batch = 32 
    epoch = 40

    ## set name for the mdoel
    title = "LSTM_modeling_side_effects_"


    ## save the best model
    best_model_path = title + 'best.h5'
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only = True) ## save only best model

    ## if 4 steps without decreasing of loss in valid set, stop the trainning
    early_stopping = EarlyStopping(patience = 4)
    LSTM_model = model.fit(X_train_t, Y_train_t, batch_size=batch, epochs=epoch,
                        validation_data=(X_train_val, y_train_val),callbacks=[model_checkpoint], shuffle = True)
    best_score = min(LSTM_model.history['val_loss'])

    ## predict test set
    accr = model.evaluate(X_test,y_test, batch_size = 100)
    print("Test set accuracy:" + accr)

    ## plot loss against iteration
    plt.plot(LSTM_model.history['loss'],label='train')
    plt.plot(LSTM_model.history['val_loss'],label='validation')
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig("loss_iteration.png")