import pandas as pd
from wit import Wit
import threading
import multiprocessing

access_token='TPGKFHFWL6UWQ2MNBTX7LBVKZP4FNYWN'
client = Wit(access_token)


SE_tem_l400 = pd.read_csv("SE_tem_l400.csv",index_col=0,header=None)[1]
test = SE_tem_l400.iloc[:10]
alist = list(test.index)
blist = list(test)

sentments = {}

def sent_one_message(a,b):
    res = client.message(b)['entities']
    sentments[a] = res

def mutiple_thread(alist,blist):
    jobs =[]

    for a,b in zip(alist,blist):
        try:
            thread = threading.Thread(target=sent_one_message(a,b))
            jobs.append(thread)
            thread.start()

        except Exception:
            continue

    for j in jobs:
            j.join()

pool_size = 4
pool = multiprocessing.Pool(pool_size)
pool.map(mutiple_thread, alist,blist)
pool.close()
pool.join()
#mutiple_thread(SE_tem_l400)


if __name__ == "__main__":
    print(sentments)