import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_chisquare
from scipy.stats import chisquare

from bs4 import BeautifulSoup
from collections import defaultdict
import requests
import re
from collections import Counter
import string

def drug_link(letter):
    '''
    INPUT: string
    OUTPUT: list of strings

    Take a query and return a list of all of the etsy usernames who have a
    result on the first page of that query result.
    '''

    url = "http://www.druglib.com/drugindex/rating/%s/" % letter
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    search = soup.find_all('a', attrs={'href':re.compile('/ratingsreviews/.*')})
    drug_link = ["http://www.druglib.com"+ a['href'] for a in search]
    return drug_link#[a for a in search]


def scrap_one_page(test_link):
    html = requests.get(test_link).text
    soup = BeautifulSoup(html, 'html.parser')

    ## age_geneder
    age_geneder = soup.find_all('h2')
    age_geneder = [a.contents[0] for a in age_geneder]

    ## rating
    rating =soup.find_all('img',attrs={'hight':"18px"})
    rating = [a['src'] for a in rating]
    rating = [rating[x:x+10] for x in range(0, len(rating),10)]
    rating = [Counter(a)['/img/red_star.gif'] for a in rating ]

    all_contents = [a.contents[0] for a in soup.find_all('td',attrs={'class':"review3"})]
    ## effective
    effective = [a for a in all_contents if 'Effective' in a]

    ## effects
    effects = [a for a in all_contents if 'Effects' in a]

    all_contents_sorted = []
    i = 0
    for x in range(0, len(all_contents),10):
        all_contents_sorted.append([age_geneder[i]]+[rating[i]] + all_contents[x+1:x+10])

    return all_contents_sorted

if __name__ == "__main__":
    all_lo_list = []

    for l in string.ascii_lowercase:
        oletter_alist =[]
        for each in  drug_link(l):
            temp_list = scrap_one_page(each)
            oletter_alist = oletter_alist + temp_list
        all_lo_list = all_lo_list + oletter_alist

    df_ws = pd.DataFrame(all_lo_list)
    df_ws.to_csv("web_scrap.csv")
