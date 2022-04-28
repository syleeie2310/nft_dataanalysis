# Databricks notebook source
# MAGIC %md # Setting

# COMMAND ----------

import base64
import pandas as pd
from IPython.display import HTML

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# COMMAND ----------

# set the environment path to find Recommenders
import sys
sys.path.append("/databricks/driver/Recommenders")
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType, IntegerType, LongType

from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator

import itertools
import logging
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.utils.notebook_utils import is_jupyter
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkRatingEvaluation, SparkRankingEvaluation
from recommenders.utils.spark_utils import start_or_get_spark

# reco_utils가 이름을 바꿔서 지원함. recommenders cluster2에 install하고 als 예제 install목록 가져왔음
# from reco_utils.dataset import movielens
# from reco_utils.common.spark_utils import start_or_get_spark
# from reco_utils.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation

# from reco_utils.dataset.python_splitters import python_stratified_split
# from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
# from reco_utils.recommender.sar.sar_singlenode import SARSingleNode

print("System version: {}".format(sys.version))
print("Spark version: {}".format(pyspark.__version__))

# COMMAND ----------

# MAGIC %md #Data Load
# MAGIC * nft_address_cocount
# MAGIC ```
# MAGIC nft_address1, nft_address2 를 공통적으로 소유한 distinct 한 owner 수 = owner_cnts
# MAGIC distinct 하지 않은 전체 소유자 수 = cnts
# MAGIC ```

# COMMAND ----------

nft_address_cocount = spark.read.parquet("/FileStore/nft/kaggle/nft_address_cocount")

# COMMAND ----------

nft_address_cocount.show(5)

# COMMAND ----------

nft_address_cocount.createOrReplaceTempView("nft_address_cocount")

# COMMAND ----------

# MAGIC %md # Data 알아보기

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from nft_address_cocount
# MAGIC order by 3 desc

# COMMAND ----------

# MAGIC %sql
# MAGIC select max(owner_cnts), min(owner_cnts), avg(owner_cnts), count(owner_cnts), stddev_samp(owner_cnts)
# MAGIC from nft_address_cocount

# COMMAND ----------

# MAGIC %md #분포

# COMMAND ----------

nac_df = pd.read_parquet("/dbfs/FileStore/nft/kaggle/nft_address_cocount")

# COMMAND ----------

nac_df.head(5)

# COMMAND ----------

nac_df.describe()

# COMMAND ----------

nac_df.sort_values(by = ['owner_cnts'], axis = 0, ascending = False)

# COMMAND ----------

nac_df.sort_values(by = ['cnts'], axis = 0, ascending = False)

# COMMAND ----------

# MAGIC %md ## owner_cnts 10 이상, 10미만

# COMMAND ----------

over_10_num = (nac_df.owner_cnts >= 10)

# COMMAND ----------

a = nac_df.loc[over_10_num, ['owner_cnts']].count()
a

# COMMAND ----------

lower_10_num = (nac_df.owner_cnts < 10)

# COMMAND ----------

b = nac_df.loc[lower_10_num, ['owner_cnts']].count()
b

# COMMAND ----------

nac_df.loc[lower_10_num, ['owner_cnts']]

# COMMAND ----------

df_big = nac_df[nac_df['owner_cnts'] >= 10]

# COMMAND ----------

df_small = nac_df[nac_df['owner_cnts'] < 10]

# COMMAND ----------

same1 = (nac_df.owner_cnts == 1)
c = nac_df.loc[same1, ['owner_cnts']].count()
c

# COMMAND ----------

plt.figure(figsize=(24,24))
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(24, 5, forward=True)
fig.suptitle('About owner_cnt 10')
axs[0].hist(df_small["owner_cnts"], density=False, alpha=0.75, log=False, bins = 9, color='orange')
axs[0].set_title('lower than 10')
axs[1].hist(df_big["owner_cnts"], density=False, alpha=0.75, log=True, bins=20)
axs[1].set_title('bigger than 10')
# plt.setp(axs[0], xlabel='Number of transactions out', ylabel='Number of addresses')
# plt.setp(axs[1], xlabel='Number of transactions in', ylabel='Number of addresses')
print("")

# COMMAND ----------

# MAGIC %md # 유사도 정의
# MAGIC 
# MAGIC * Jaccard = cij / (A distinct owner + B distinct owner - cij)
# MAGIC * Lift = cij / (A distinct owner * B distinct owner)
# MAGIC * Confidence = cij / A distinct owner
# MAGIC * count = cij

# COMMAND ----------

# MAGIC %md ###분모 만들기

# COMMAND ----------

co_df = spark.read.parquet("/FileStore/nft/kaggle/current_owners_proc")

# COMMAND ----------

co_df.createOrReplaceTempView("co_df")

# COMMAND ----------

co_df.show()

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC count(distinct co_df.owner) 
# MAGIC from co_df

# COMMAND ----------

OUTPUT_PATH = "/FileStore/nft/kaggle/current_owners_proc"
co_df_parquet = spark.read.parquet(OUTPUT_PATH)
co_df_parquet.createOrReplaceTempView("co_df_parquet")

# COMMAND ----------

# co_df4 = spark.sql("""
# select t.*
# from(
# select a.nft_address as nft_address1, count(distinct a.owner) owner_cnts, count(*) cnts
# from co_df_parquet as b
# on 1=1
# group by a.nft_address, a.owner
# ) as t
# where 1=1 
# """)


# OUTPUT_PATH = "/FileStore/nft/kaggle/nft_address_test"
# co_df4.repartition(100).write.save(OUTPUT_PATH, format='parquet', mode='overwrite')

# COMMAND ----------

# co_df3 = spark.sql("""
# select t.*
# from(
# select co_df_parquet.nft_address, count(distinct co_df_parquet.owner) owner_Cnts, count(*) cnts
# from co_df_parquet as b
# on 1=1
# group by co_df_parquet.nft_address, co_df_parquet.owner
# )as t
# where 1=1
# """)

# COMMAND ----------

co_ddf = pd.read_parquet("/dbfs/FileStore/nft/kaggle/current_owners_proc")

# COMMAND ----------

co_ddf.head()

# COMMAND ----------

# nft_address로 owner 수 groupby 했음 
address_df = co_ddf.groupby(['nft_address'], as_index = False).size().rename(columns = {'size': 'num_owners'})

# COMMAND ----------

# nft_address의 owner 수 sort values 해봄
address_df.sort_values("num_owners", inplace = True, ascending = False)

# COMMAND ----------

address_df.head(20)

# COMMAND ----------

# MAGIC %md ###owner 수 많은거 20개 Jaccard 구해보자!

# COMMAND ----------

# 자카드 유사도 구하는 공식 함수
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

# COMMAND ----------

address_cocount_df = pd.read_parquet("/dbfs/FileStore/nft/kaggle/nft_address_cocount")

# COMMAND ----------

address_cocount_df.sort_values('owner_cnts', inplace = True, ascending = False)

# COMMAND ----------

address_cocount_df.head(10)

# COMMAND ----------

# MAGIC %md ### confidence
# MAGIC 
# MAGIC * cij / A distinct owner

# COMMAND ----------

# 기준 address 기준으로 merge, num_owners까지 한번에 보임. 
merge_df = pd.merge(address_cocount_df, address_df, how = 'left', left_on = 'nft_address1', right_on = 'nft_address')

# COMMAND ----------

merge_df.head(10)

# COMMAND ----------

# confidence 추가
merge_df['confidence'] = merge_df['owner_cnts'] / merge_df['num_owners']

# COMMAND ----------

merge_df.head()

# COMMAND ----------

# MAGIC %md ### Lift
# MAGIC 
# MAGIC * cij / (A distinct owner * B distinct owner)

# COMMAND ----------

# index에 있는 address1이랑 address2 
# 각자 num_owners 가져와야함. 


# COMMAND ----------

# a_owners = (merge_df.nft_address1 == merge_df.nft_address)

# a_owners_df = merge_df.loc[a_owners, ['num_owners']]

# 애초에 nft_address1 기준으로 합쳤기 때문에 num_owners는 Distinct A임!! 
# distnct B를 구해야함. 

# COMMAND ----------

b_owner_df = pd.merge(address_cocount_df, address_df, how = 'left', left_on = 'nft_address2', right_on = 'nft_address')

# COMMAND ----------

# nft_address 2 기준으로 num_owners인것. 
b_owner_df

# COMMAND ----------

merge_df['Lift'] = merge_df['owner_cnts'] / (merge_df['num_owners'] * b_owner_df['num_owners'])

# COMMAND ----------

merge_df.head()

# COMMAND ----------

# MAGIC %md ## Jaccard!!
# MAGIC 
# MAGIC * cij / (A distinct owner + B distinct owner - cij)

# COMMAND ----------

merge_df['Jaccard'] = merge_df['owner_cnts'] / (merge_df['num_owners'] + b_owner_df['num_owners'] - merge_df['owner_cnts'])

# COMMAND ----------

merge_df.head()

# COMMAND ----------

merge_df.drop(['nft_address', 'num_owners'], axis = 1, inplace = True)

# COMMAND ----------

merge_df.head()

# COMMAND ----------

# MAGIC %md 데이터 저장

# COMMAND ----------

# merge_df.to_csv("/dbfs/FileStore/nft/kaggle/Jaccard_nft_address.csv" )

# COMMAND ----------

# MAGIC %md # 완성된 데이터프레임 로드

# COMMAND ----------

Jaccard_nft_address = pd.read_csv("/dbfs/FileStore/nft/kaggle/Jaccard_nft_address.csv")

# COMMAND ----------

Jaccard_nft_address.head()

# COMMAND ----------

Jaccard_nft_address.sort_values('Jaccard', ascending = False)

# COMMAND ----------

Jaccard_nft_address.sort_values('confidence', ascending = False)

# COMMAND ----------

Jaccard_nft_address.sort_values('Lift', ascending = False)

# COMMAND ----------

# MAGIC %md ## Jaccard 분포 확인하기 

# COMMAND ----------

Jaccard_nft_address.describe()

# COMMAND ----------

# MAGIC %md ### Jaccard plot

# COMMAND ----------

over_half = (Jaccard_nft_address.Jaccard >= 0.001)

Ja = Jaccard_nft_address.loc[over_half, ['Jaccard']].count()
Ja

# COMMAND ----------

lower_half = (Jaccard_nft_address.Jaccard < 0.001)

Jb = Jaccard_nft_address.loc[lower_half, ['Jaccard']].count()
Jb

# COMMAND ----------

J_big = Jaccard_nft_address[Jaccard_nft_address['Jaccard'] >= 0.01]

# COMMAND ----------

J_small = Jaccard_nft_address[Jaccard_nft_address['Jaccard'] < 0.01]

# COMMAND ----------

plt.figure(figsize=(24,24))
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(24, 5, forward=True)
fig.suptitle('about_Jaccard')
axs[0].hist(J_small["Jaccard"], density=False, alpha=0.75, log=False, bins = 20, color='orange')
axs[0].set_title('lower than 0.01')
axs[1].hist(J_big["Jaccard"], density=False, alpha=0.75, log=False, bins=20)
axs[1].set_title('bigger than 0.01')
# plt.setp(axs[0], xlabel='Number of transactions out', ylabel='Number of addresses')
# plt.setp(axs[1], xlabel='Number of transactions in', ylabel='Number of addresses')
print("")

# COMMAND ----------

# MAGIC %md ### Lift plot

# COMMAND ----------

over_half = (Jaccard_nft_address.Lift >= 0.00001)

Ja_lift = Jaccard_nft_address.loc[over_half, ['Lift']].count()
Ja_lift

# COMMAND ----------

lower_half = (Jaccard_nft_address.Lift < 0.00001)

Jb_lift = Jaccard_nft_address.loc[lower_half, ['Lift']].count()
Jb_lift

# COMMAND ----------

L_big = Jaccard_nft_address[Jaccard_nft_address['Lift'] >= 0.00001]

# COMMAND ----------

L_small = Jaccard_nft_address[Jaccard_nft_address['Lift'] < 0.00001]

# COMMAND ----------

plt.figure(figsize=(24,24))
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(24, 5, forward=True)
fig.suptitle('about_Lift')
axs[0].hist(L_small["Lift"], density=False, alpha=0.75, log=False, bins = 20, color='orange')
axs[0].set_title('lower than 0.00001')
axs[1].hist(L_big["Lift"], density=False, alpha=0.75, log=False, bins=20)
axs[1].set_title('bigger than 0.00001')
# plt.setp(axs[0], xlabel='Number of transactions out', ylabel='Number of addresses')
# plt.setp(axs[1], xlabel='Number of transactions in', ylabel='Number of addresses')
print("")
# 로그 취해서 봐야하나? 함 생각해봐야할듯. 

# COMMAND ----------

# MAGIC %md ### confidence plot

# COMMAND ----------

over_half = (Jaccard_nft_address.confidence >= 0.01)

Ja_con = Jaccard_nft_address.loc[over_half, ['confidence']].count()
Ja_con

# COMMAND ----------

lower_half = (Jaccard_nft_address.confidence < 0.01)

Jb_con = Jaccard_nft_address.loc[lower_half, ['confidence']].count()
Jb_con

# COMMAND ----------

C_big = Jaccard_nft_address[Jaccard_nft_address['confidence'] >= 0.01]

# COMMAND ----------

C_small = Jaccard_nft_address[Jaccard_nft_address['confidence'] < 0.01]

# COMMAND ----------

plt.figure(figsize=(24,24))
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(24, 5, forward=True)
fig.suptitle('about_Confidence')
axs[0].hist(C_small["confidence"], density=False, alpha=0.75, log=False, bins = 20, color='orange')
axs[0].set_title('lower than 0.01')
axs[1].hist(C_big["confidence"], density=False, alpha=0.75, log=False, bins=20)
axs[1].set_title('bigger than 0.01')
# plt.setp(axs[0], xlabel='Number of transactions out', ylabel='Number of addresses')
# plt.setp(axs[1], xlabel='Number of transactions in', ylabel='Number of addresses')
print("")

# COMMAND ----------

# MAGIC %md ### nft 이름 붙여넣기

# COMMAND ----------

Jaccard_nft_address.head()

# COMMAND ----------

nfts_df = pd.read_csv("/dbfs/FileStore/nft/kaggle/nfts.csv")

# COMMAND ----------

nfts_df.head()

# COMMAND ----------

nfts_merge = pd.merge(Jaccard_nft_address, nfts_df, how = 'left', left_on = 'nft_address1', right_on = 'address')

# COMMAND ----------

nfts_merge.head()

# COMMAND ----------

# merge_df.drop(['nft_address', 'num_owners'], axis = 1, inplace = True)

# COMMAND ----------

nfts_merge.drop(['Unnamed: 0_y', 'address', 'symbol'], axis = 1, inplace = True)

# COMMAND ----------

nfts_merge.head()

# COMMAND ----------

nfts_merge.sort_values('confidence', ascending = False).head(20)

# COMMAND ----------

nfts_merge.sort_values('Lift', ascending = False).head(20)

# COMMAND ----------

nfts_merge.sort_values('Jaccard', ascending = False).head(20)

# COMMAND ----------

not_one = nfts_merge[nfts_merge['owner_cnts'] != 1 ]

# COMMAND ----------

not_one.sort_values('Jaccard', ascending = False).head(20)

# COMMAND ----------

not_one.sort_values('Lift', ascending = False).head(20)

# COMMAND ----------

not_one.sort_values('confidence', ascending = False).head(20)

# COMMAND ----------

not_one.sort_values('owner_cnts', ascending = False).head(30)

# COMMAND ----------

nfts_merge.head()

# COMMAND ----------

nfts_merge = nfts_merge.drop(['Unnamed: 0_x'], axis = 1)

# COMMAND ----------

nfts_merge.head()

# COMMAND ----------

plt.figure(figsize=(24, 12))
# new_df.unstack()
ax = sns.lineplot(data=nfts_merge, x='owner_cnts', y='confidence', hue = 'name')

# ax.set(yscale="log")

# COMMAND ----------

# nfts_merge.to_csv("/dbfs/FileStore/nft/kaggle/nfts_merge.csv" )

# COMMAND ----------

nfts_merge = spark.read.csv("/FileStore/nft/kaggle/nfts_merge.csv", header = 'true')

# COMMAND ----------

nfts_merge.show(5)

# COMMAND ----------

nfts_merge.display()

# COMMAND ----------

display(nfts_merge)

# COMMAND ----------



# COMMAND ----------


