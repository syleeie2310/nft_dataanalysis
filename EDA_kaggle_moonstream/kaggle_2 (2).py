# Databricks notebook source
import numpy as np
import pandas as pd
import sqlite3
from matplotlib.pyplot import figure

import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style = "white", color_codes = True)

# COMMAND ----------

import os
file_list = os.listdir('/dbfs/FileStore/nft/kaggle/k_sql')
len(file_list), print(file_list)

# COMMAND ----------

# MAGIC %md # 특정 NFT 관련 분석
# MAGIC 
# MAGIC * 콜렉션 별 분석 (art blocks)
# MAGIC - 최저, 최고가 
# MAGIC - 콜렉션 별 가격 변동폭, 최저 / 최고 / 평균 / 중간 (변동성이 가장 큰 아이템?)
# MAGIC - 거래 수가 가장 많은 아이템 
# MAGIC 
# MAGIC ** 현재 유통중인 NFT 수? (가격을 곱하면 volume이 나옴)
# MAGIC - 가격(바닥가 / 현재?)
# MAGIC - NFT 거래 수 (현재) -> 유통 수 대비 거래수 비교 가능 
# MAGIC - 현재 활동중인 전체 사용자 수? (구매, 판매, 홀딩한 시간)
# MAGIC - 콜렉션 시가 총액 시장 점유율? 

# COMMAND ----------

# MAGIC %md ## 쓸 수 있는 column / dataset
# MAGIC 
# MAGIC ** 이 3개는 row 수 같음. (transfer_values_quantele 10이것도)
# MAGIC * current_market_values 
# MAGIC * current_owners
# MAGIC * market_values_distribution
# MAGIC 
# MAGIC ** row 수 다르지만 컬럼이 같음. 
# MAGIC * mints
# MAGIC * transfers 
# MAGIC 
# MAGIC ** 타임끼리 묶어서 볼 수 있나? 
# MAGIC * mint_holding_times
# MAGIC * transfer_holding_times 

# COMMAND ----------

nfts = pd.read_csv("/dbfs/FileStore/nft/kaggle/nfts.csv", low_memory = False)

# COMMAND ----------

current_market_values = pd.read_csv("/dbfs/FileStore/nft/kaggle/current_market_values.csv", low_memory = False)

# COMMAND ----------

current_market_values

# COMMAND ----------

# current_market_values.drop(['Unnamed: 0'], axis = 1, inplace = True)
# 이미 지워서 에러뜨는거임! inplace True로 날려줬음~

# COMMAND ----------

current_market_values.head(5)

# COMMAND ----------

nfts.head(5)

# COMMAND ----------

# nfts.drop(['Unnamed: 0'], axis = 1, inplace = True)
# 드랍해줬음 쓸모 없는 컬럼

# COMMAND ----------

nfts.head(5)

# COMMAND ----------

nfts_market_value = nfts[['address', 'name']].merge(current_market_values, how = 'inner', left_on = 'address', right_on = 'nft_address')

# COMMAND ----------

nfts_market_value.head()

# COMMAND ----------

nfts_market_value.tail()

# COMMAND ----------

# nfts_market_value.drop(['Unnamed: 0', 'nft_address', 'token_id'], axis = 1, inplace = True)

# COMMAND ----------

nfts_market_value

# COMMAND ----------

nfts_market_value['name'].unique()

# COMMAND ----------

nfts_market_value['name'].nunique()
# nft name 수 8470개!! 

# COMMAND ----------

nfts_market_value['market_value'] = pd.to_numeric(nfts_market_value['market_value'])
nfts_market_value.info()
# nfts_market_value['name'].unique()

# COMMAND ----------

# is_name_unique = nfts_market_value['name'].unique()

# name_unique = nfts_market_value[is_name_unique]

# name_unique

# COMMAND ----------

nfts_value = nfts_market_value['market_value'].groupby(nfts_market_value['name']).sum()

# COMMAND ----------

# MAGIC %md ## nft별 value 값 뽑았음.

# COMMAND ----------

nfts_value

# COMMAND ----------

transfers = pd.read_csv("/dbfs/FileStore/nft/kaggle/transfers.csv", low_memory = False)

# COMMAND ----------

total_value_per_nft = transfers[["nft_address", "transaction_value"]].groupby(transfers["nft_address"]).sum()
total_value_per_nft.head(10)

# COMMAND ----------

transfer_values = nfts[['address', 'name']].merge(total_value_per_nft, how = 'inner', left_on = 'address', right_on = 'nft_address')

# COMMAND ----------

transfer_values

# COMMAND ----------

transfer_values = transfer_values['transaction_value'].groupby(transfer_values['name']).sum()

# COMMAND ----------

transfer_values

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

current_market_values['market_value'].sum()

# COMMAND ----------

current_market_values

# COMMAND ----------

# MAGIC %md ## 메모장

# COMMAND ----------

co_df = pd.read_csv("/dbfs/FileStore/nft/kaggle/current_owners.csv", low_memory = False)

# COMMAND ----------

co_df.head()

# COMMAND ----------

co_df

# COMMAND ----------

nfts
