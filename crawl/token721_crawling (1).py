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


