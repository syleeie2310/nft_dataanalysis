# Databricks notebook source
# MAGIC %md # 721 API Test

# COMMAND ----------

from etherscan import Etherscan
eth = Etherscan('E9V452TNCZUVD73XHDQSQRXWYEZDDXY13B')

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# MAGIC %md ## info_pdh
# MAGIC 
# MAGIC Get a list of 'ERC721 - Token Transfer Events' by Address
# MAGIC 
# MAGIC get_erc721_token_transfer_events_by_address
# MAGIC 
# MAGIC 필요 매개변수 = address: str, startblock: int, endblock: int, sort: str
# MAGIC (주소, 시작 블록, 끝 블록, (내림차순/오름차순 선택))
# MAGIC 
# MAGIC 비교 사이트 https://etherscan.io/tokentxns-nft?a=0xec8a26d9a20c81cbd2f47348c708215302a2490d

# COMMAND ----------

# MAGIC %md ##info
# MAGIC ```
# MAGIC https://api.etherscan.io/api
# MAGIC    ?module=account
# MAGIC    &action=tokennfttx
# MAGIC    &contractaddress=0x06012c8cf97bead5deae237070f9587f8e7a266d
# MAGIC    &address=0x6975be450864c02b4613023c2152ee0743572325
# MAGIC    &page=1
# MAGIC    &offset=100
# MAGIC    &startblock=0
# MAGIC    &endblock=27025780
# MAGIC    &sort=asc
# MAGIC    &apikey=YourApiKeyToken
# MAGIC ```

# COMMAND ----------

a = eth.get_erc721_token_transfer_events_by_address('0xEC8A26D9a20c81cBd2f47348C708215302A2490D',0,99999999,'desc')

# COMMAND ----------

a

# COMMAND ----------

# blockNumber = 해당 거래가 있는 블록 넘버? 
# time stamp = timestamp 
# hash = ?
# nonce = 작업증명 
# blockHash = 해당 거래가 있는 블록 해쉬
# from = 보내는 지갑 주소 
# contract Address = 거래 주소
# to = 받는 지갑 주소 
# token ID = 해당 토큰 식별 id
# token NAme = erc 721 이름 (NFT 이름)
# token Symbol = NFT 약자
# tokenDecimal = (decimal 은 십진법이라는 뜻 아니면 소수 라는 뜻인데 잘 모르겠음)
# transaction Index = 거래 index 번호같음
# gas = 가스량?
# gas price = 가스비
# gas Used = 사용된 가스량
# cumulative gas used = 누적 사용 가스량
# input : deprecated 라고 되어있는데 뜻: 중요도가 사라져 곧 없어질 것 이라는 의미. 
# confirmations = 확인. 확정? 

# COMMAND ----------

# MAGIC %md ## contract address로도 가져올 수 있는지 확인해보기 

# COMMAND ----------

gucci = eth.get_erc721_token_transfer_events_by_address(contractaddress ='0xBFc1FbfDB3c440459D0ECe2de312086fFBba1bEa', 0, 9999999, 'desc')
# No transactions found => 구찌 자체 contract address로는 가져올 수 없는듯. 

# COMMAND ----------

# MAGIC %md ## transaction hash (Txn hash)로 가져올 수 있는지 확인 
# MAGIC 
# MAGIC - 안됨!

# COMMAND ----------

gucci_1 = eth.get_erc721_token_transfer_events_by_address('0xb699cd6f9164d93a8957344c8a1cbfac3212a4341d23b5674b71af162512d225', 0, 9999999, 'desc')

# COMMAND ----------

0xf5c5a3545d56ab2e5dfe7079cb05e893b48dde92

# COMMAND ----------

gucci_1 = eth.get_erc721_token_transfer_events_by_address('0xf5c5a3545d56ab2e5dfe7079cb05e893b48dde92', 0, 9999999, 'desc')

# COMMAND ----------

# input address 
address = input()

# get erc 721 
erc_721 = eth.get_erc721_token_transfer_events_by_address('address',0, 99999999, 'desc')

# 'address', 'startblock', 'endblock', and 'sort'

time_stamp = erc_721["timeStamp"]
hash_721 = erc_721["hash"]
from_721 = erc_721["from"]
to_721 = erc_721["to"]
contract_address = erc_721["contractAddress"]
token_id = erc_721["tokenID"]
token_name = erc_721["tokenName"]
token_symbol = erc_721["tokenSymbol"]
gas = erc_721["gas"]
gas_price = erc_721["gasPrice"]
gas_used = erc_721["gasUsed"]

time_message = "timestamp:" + time_stamp

# COMMAND ----------

# message 출력 

time_message = "timestamp" + time_stamp

print(time_message)

# COMMAND ----------

# MAGIC %md ## pcko code 가져왔었음

# COMMAND ----------

# from functools import reduce
# from typing import List

# from etherscan.enums.actions_enum import ActionsEnum as actions
# from etherscan.enums.fields_enum import FieldsEnum as fields
# from etherscan.enums.modules_enum import ModulesEnum as modules
# from etherscan.enums.tags_enum import TagsEnum as tags

# COMMAND ----------

# class Accounts:
#     @staticmethod
#     def get_erc721_token_transfer_events_by_address(
#         address: str, startblock: int, endblock: int, sort: str,
#     ) -> str:
#         url = (
#             f"{fields.MODULE}"
#             f"{modules.ACCOUNT}"
#             f"{fields.ACTION}"
#             f"{actions.TOKENNFTTX}"
#             f"{fields.ADDRESS}"
#             f"{address}"
#             f"{fields.START_BLOCK}"
#             f"{str(startblock)}"
#             f"{fields.END_BLOCK}"
#             f"{str(endblock)}"
#             f"{fields.SORT}"
#             f"{sort}"
#         )
#         return url

# COMMAND ----------

# MAGIC %md # contract address로 데이터 가져왔음! 
# MAGIC 
# MAGIC - 사용 함수는 erc_721_token_transfer_events_by_contract_address_paginated

# COMMAND ----------

# MAGIC %md ## desc로 최신순 가져옴

# COMMAND ----------

# contract address로 가져오기
# 함수 이름 해석하자면 페이지수? 별?로 Erc 721 전송 이벤트 가져왔음. 
g = eth.get_erc721_token_transfer_events_by_contract_address_paginated('0x6d51cc615526Bd2adFDc050649c65df70112baeC', 0, 10000, 'desc')

# COMMAND ----------

g

# COMMAND ----------

k = g[0].keys()
print(k)

# COMMAND ----------

g[0].values()

# COMMAND ----------

list = []

for i in range(len(g)):
    list.append(g[i].values())
print(list)

# COMMAND ----------

gucci = pd.DataFrame(list, columns = k)

# COMMAND ----------

gucci
# 8471개임

# COMMAND ----------

# MAGIC %md ## asc로 과거순으로 가져옴

# COMMAND ----------

g_asc = eth.get_erc721_token_transfer_events_by_contract_address_paginated('0x6d51cc615526Bd2adFDc050649c65df70112baeC', 0, 10000, 'asc')

# COMMAND ----------

g_asc

# COMMAND ----------

k_asc = g_asc[0].keys()

list_asc = []

for i in range(len(g_asc)):
    list_asc.append(g_asc[i].values())
print(list_asc)

# COMMAND ----------

gucci_asc = pd.DataFrame(list_asc, columns = k_asc)

# COMMAND ----------

gucci_asc
# 8993개임 block Number

# COMMAND ----------

# MAGIC %md ## 만개 이상 가져와질까? 
# MAGIC - 답 : No.

# COMMAND ----------

lv = eth.get_erc721_token_transfer_events_by_contract_address_paginated('0x6dBC1F3961fd606fEaF5fb73245aA4b2674d1e66', 0, 10000, 'asc')

# COMMAND ----------

lv

# COMMAND ----------

lv_k = lv[0].keys()

# COMMAND ----------

list_lv = []

for i in range(len(lv)):
    list_lv.append(lv[i].values())
print(list_lv)

# COMMAND ----------

louisvuitton = pd.DataFrame(list_lv, columns = lv_k)

# COMMAND ----------

louisvuitton

# COMMAND ----------

# MAGIC %md #저장하기

# COMMAND ----------

gucci.to_csv("/dbfs/FileStore/nft/etherscan/gucci_ether.csv" )

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # desc, asc 사이 블록넘버로 계약 거래 가져오기 
# MAGIC 
# MAGIC - 문제 하나 있음. desc, asc 바꾸어서 해보았을 때 중간에 블록 넘버가 비는 지점이 생김
# MAGIC ```
# MAGIC 따라서 그 비는 지점을 블록 넘버로 가져올 수 있는 함수 사용. 
# MAGIC 하지만 또 문제 발생. contract address가 전부 없음
# MAGIC ```

# COMMAND ----------

eth.get_internal_txs_by_block_range_paginated(14336319, 14344060, 0, 10000, 'asc')
# 대충격 contract address 안나옴.... 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # 다른 함수 써보기

# COMMAND ----------

eth.get_erc721_token_transfer_events_by_address_and_contract_paginated('')

# COMMAND ----------


