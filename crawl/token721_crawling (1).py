# Databricks notebook source
# MAGIC %md
# MAGIC # 데이터브릭스에서는 Header 설정을 어떻게 해야하지??????????

# COMMAND ----------

from urllib.request import Request, urlopen
import pandas as pd
import requests


# res = Request('https://etherscan.io/tokens-nft?p=1')
# res.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.53')
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
res = requests.get('https://etherscan.io/tokentxns-nft?p=1', headers=headers)
print(res)

# COMMAND ----------

soup.select('.text-primary')

# COMMAND ----------

import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.53'}



# COMMAND ----------

import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.53'}


tokens = [] 
token_contract = [] 

cnt = 0 
for i in range(1,3):#  1p ~ 890 p 
    url = 'https://etherscan.io/tokens-nft?p='+str(i)
    res = requests.get(url, headers=headers)
    # soup =  BeautifulSoup(res.text,'lxml')
    soup = BeautifulSoup(res.text, 'html.parser')
    
    for j in range(0,10):    
        print(soup.select('.text-primary')[j].get_text()  )
        print(soup.select('.text-primary')[j]['title']   )   
        
        tokens.append(     soup.select('.text-primary')[j].get_text()   )
        token_contract.append(   soup.select('.text-primary')[j]['title']   )
        cnt += 1
        print(cnt)
    
        time.sleep(1)
    time.sleep(1)
tmp_dict = {  name : value for name, value in zip(tokens,token_contract)        }
# token721 = pd.DataFrame(list(tmp_dict.items())   ,columns=['tokens','tokens_contract'])
# token721 = pd.DataFrame(list(tmp_dict.items())   ,columns=['tokens','tokens_contract'])

# COMMAND ----------

def make_token721():  # 중복 값은 제거해줌 
    tokens = [] 
    token_contract = [] 
    transfer_1D = []
    transfer_3D = []
    import time
    import datetime
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.53'}


    cnt = 0 
    for i in range(1, 101):#  1p ~ 890 p  중에    5000개의 데이터만 가지고 왔음 
        url = 'https://etherscan.io/tokens-nft?p='+str(i)
        res = requests.get(url, headers=headers)
        # soup =  BeautifulSoup(res.text,'lxml')
        soup = BeautifulSoup(res.text, 'html.parser')

        for j in range(0,50):    
            print(soup.select('.text-primary')[j].get_text()  )
            tokens.append(     soup.select('.text-primary')[j].get_text()   )
            time.sleep(2)
            try :print(soup.select('.text-primary')[j]['title']); token_contract.append(   soup.select('.text-primary')[j]['title'] );                time.sleep(2) 
            except:token_contract.append(soup.select('.text-primary')[j].get_text() )
                                 
            transfer_1D .append(soup.select('tbody >tr')[j].select('td')[2].get_text() )
            transfer_3D .append(soup.select('tbody >tr')[j].select('td')[3].get_text() )
            cnt += 1
            print(cnt)

            
        time.sleep(6)
#     tmp_dict = {  name : value for name, value in zip(tokens,token_contract)        }

    
    namelist = ['tokens','tokens_contract','Transfers (24H)','Transfers (3D)' ]
    token721 = pd.DataFrame(data= [tokens,token_contract, transfer_1D,  transfer_3D]).T
    token721.columns = namelist
    
    # csv 이름 설정 (날짜 형식으로 )
    now = datetime.datetime.now()
    now = now.strftime('%Y_%m_%d_%H_%M_%S')
    csvname = 'data/token721/token721_' + now + '.csv'
#     token721.to_csv('data/token721_0221.csv')
    token721.to_csv(csvname)
    
    return print('함수 종료')


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


