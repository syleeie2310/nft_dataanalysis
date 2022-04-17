# Databricks notebook source
from multiprocessing import Process
from etherscan import Etherscan
import time
import numpy as np 
import multiprocessing
import parmap
import requests
import pandas as pd
eth = Etherscan('E9V452TNCZUVD73XHDQSQRXWYEZDDXY13B')


# COMMAND ----------

# MAGIC %md
# MAGIC # get_erc721_token_transfer_events_by_contract_address_paginated

# COMMAND ----------

# 멀티 프로세싱 사용하기 위한 함수 
# 
# 임의 10개 계약주소  ex
ex = ['0x69B9C98e8D715C25B330d0D4eB07e68CbB7F6CFC',
'0xDccc916bF4a0186065f4d1e5b94176F4d17b8C42',
'0x6669b006cae9e96573fb5b192310de47a3e8575b',
'0x600c411c1195605cc75a7645f2bca782fcfc4d6c',
'0xea745b608b2b61987785e87279db586f328eb8fc',
'0x36060a7313A65d4d510827003bdC7166F23C1c67',
'0xb4a4961eddeded48ca1a8c3a2fd0d89e586446d5',
'0xCa7cA7BcC765F77339bE2d648BA53ce9c8a262bD',
'0xc7df86762ba83f2a6197e1ff9bb40ae0f696b9e6',      
'0x9d8826881a2beab386a7858e59c228a85b3963e1'
]
num_cores = multiprocessing.cpu_count() # 12
def test(input_ex):
    x = pd.DataFrame(eth.get_erc721_token_transfer_events_by_contract_address_paginated(input_ex, 1, 10000, 'desc'))
    time.sleep(1)
    return x 

num_cores = multiprocessing.cpu_count()

start_time = time.time()
result = parmap.map(test, ex, pm_pbar=True, pm_processes=1)   # 주소가 담긴 리스트를 매개변수로 넣자.
print((time.time()-start_time),'초')
result

# COMMAND ----------

start_time = time.time()
result = parmap.map(test, ex, pm_pbar=True, pm_processes=4)   # 주소가 담긴 리스트를 매개변수로 넣자.
print((time.time()-start_time),'초')
result

# COMMAND ----------

# 1개당 1만개 x 10개  ERC721API 내부거래 테스트 

# 프로세스 4개 사용 -> 14초 
# 프로세스 1개 사용 -> 43초 
