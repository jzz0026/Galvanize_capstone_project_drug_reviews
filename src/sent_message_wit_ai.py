from wit import Wit
import pandas as pd
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from bs4 import BeautifulSoup
from collections import defaultdict
import requests

def sent_one_message(idx):
    b = SE_tem_l400.loc[idx]
    try:
        res = client.message(b)#['entities']
        return [idx,res]
    
    except Exception:
        pass

if __name__ == "__main__":
    
    SE_tem_l400 = pd.read_csv("wit_ai_sentment/SE_tem_l400.csv",index_col=0,header=None)[1]
    SE_tem_l400 = SE_tem_l400.iloc[3:].astype(str)
    idxs = list(SE_tem_l400.index)


    access_token='TPGKFHFWL6UWQ2MNBTX7LBVKZP4FNYWN'
    client = Wit(access_token)

    sentments = {}
    test = SE_tem_l400

    pool_size = 6
    pool = multiprocessing.Pool(pool_size)
    
    test = pool.map(sent_one_message, idxs)
    test2= [a for a in test if a != None]

    first_quest = pd.DataFrame(test2).set_index(0)
    first_quest[2] = first_quest[1].apply(lambda x : x['entities'])
    first_quest2 = first_quest[first_quest[2] != {}]
    first_quest2["confidence"] = first_quest2[2].apply(lambda x : x['sentiment'][0]['confidence'])
    first_quest2['value'] = first_quest2[2].apply(lambda x : x['sentiment'][0]['value'])
    first_quest2['text'] = first_quest2[1].apply(lambda x: x['_text'])# = 

    first_quest2 = first_quest2[['text','confidence','value']]
    first_quest2.to_csv("wit_ai_sentment/SE_tem_l400_first_request.ccv")
    print("Total results:",len(first_quest2))