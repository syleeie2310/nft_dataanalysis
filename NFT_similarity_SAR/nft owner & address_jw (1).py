# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
sns.set(style = "white", color_codes = True)

%matplotlib inline

# 관련 라이브러리 임포트 
import matplotlib.font_manager as fm

# COMMAND ----------

import os
file_list = os.listdir('/dbfs/FileStore/nft/kaggle')
len(file_list), print(file_list)

# COMMAND ----------

# MAGIC %md # 분포 알아보기

# COMMAND ----------

# MAGIC %md ## current_owners
# MAGIC 컬럼 : nft_address, token_id, owner (nft주소, 식별자, 소유자 지갑 주소)
# MAGIC * nft_address: 소유권을 나타내는 토큰이 포함된 NFT 컬렉션 주소입니다.
# MAGIC 2. token_id: 소유권을 나타내는 토큰(수집 내)의 ID입니다.
# MAGIC 3. owner: 이 데이터 집합을 구성할 때 토큰을 소유했던 주소입니다.

# COMMAND ----------

co_df = pd.read_csv("/dbfs/FileStore/nft/kaggle/current_owners.csv", low_memory = False)

# COMMAND ----------

co_df.head(5)

# COMMAND ----------

co_df = co_df.drop(['Unnamed: 0'], axis = 1)

# COMMAND ----------

co_df.head(5)

# COMMAND ----------

# MAGIC %md ## nfts
# MAGIC 컬럼: address, name, symbol
# MAGIC * 주소 : NFT 계약의 이더리움 주소 
# MAGIC * 이름 : 계약이 나타내는 NFT 컬렉션 이름
# MAGIC * 기호 : 계약이 나타내는 NFT 컬렉션 기호 

# COMMAND ----------

nfts_df = pd.read_csv("/dbfs/FileStore/nft/kaggle/nfts.csv", low_memory = False)

# COMMAND ----------

nfts_df.head(5)

# COMMAND ----------

co_df.corr()
# 숫자가 아니라서 corr로는 안되나봄.

# COMMAND ----------

# MAGIC %md ## 고유값 알아보기

# COMMAND ----------

# 1. owner 고유값
co_df['owner'].nunique()

# COMMAND ----------

co_df['owner'].unique()

# COMMAND ----------

co_df['owner'].value_counts()

# COMMAND ----------

# 2. nft_address 고유값
co_df['nft_address'].nunique()

# COMMAND ----------

co_df['nft_address'].unique()

# COMMAND ----------

co_df['nft_address'].value_counts()

# COMMAND ----------

# describe 찍어서 확인해보기
co_df.describe()

# COMMAND ----------

# 안돌아감 너무 사이즈 커서 
# plt.title("owner_address", fontsize = 15)
# frq, bins, fig = plt.hist(co_df['owner'].unique(), bins = 10, alpha = 8, color = 'grey')
# plt.ylabel('bindo', fontsize = 12)
# plt.xlabel('u', fontsize = 12)
# plt.grid()
# plt.show()
# print('bindo array :', frq)
# print('gugan array :', bins)

# COMMAND ----------

# co_df[['owner']].hist(bins=10, alpha=.3, color='r')
# plt.show()
# # 와! 너무 오래걸린다 ^^!!!! 

# COMMAND ----------

# co = co_df['owner'].groupby(co_df['nft_address'])
# # nft address 별로 owner를 groupby 한 것

# COMMAND ----------

# co_size = co.size()

# COMMAND ----------

owners = co_df.groupby(["owner"], as_index=False).size().rename(columns={"size": "num_tokens"})

# COMMAND ----------

owners

# COMMAND ----------

a = co_df.groupby(['owner'], as_index = False)
a.head(5)

# COMMAND ----------

sns.heatmap(a, annot=True, fmt="o", linewidths=1)
plt.show()

# COMMAND ----------

# MAGIC %md ## 샘플링 해서 히스토그램 만들어보자! 
# MAGIC 
# MAGIC * 데이터가 너무 커서 안만들어짐 그냥 분포로 ,, 

# COMMAND ----------

co_sam = co_df.sample(frac = 0.1, random_state = 1004)

# COMMAND ----------

co_sam.head(5)

# COMMAND ----------

plt.title('owner_address', fontsize = 15)
frq, bins, fig = plt.hist(co_sam['owner'], bins = 10, alpha = 8, color = 'grey')
plt.ylabel('bindo', fontsize = 12)
plt.xlabel('u', fontsize = 12)
plt.grid()
plt.show()
print('bindo array :', frq)
print('gugan array:', bins)

# COMMAND ----------



# COMMAND ----------

# plt.title("owner_address", fontsize = 15)
# frq, bins, fig = plt.hist(co_df['owner'].unique(), bins = 10, alpha = 8, color = 'grey')
# plt.ylabel('bindo', fontsize = 12)
# plt.xlabel('u', fontsize = 12)
# plt.grid()
# plt.show()
# print('bindo array :', frq)
# print('gugan array :', bins)

# COMMAND ----------


