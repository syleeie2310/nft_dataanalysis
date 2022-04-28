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

# MAGIC %md # Data Load

# COMMAND ----------

Jaccard_nft = pd.read_csv("/dbfs/FileStore/nft/kaggle/Jaccard_nft_address.csv")

# COMMAND ----------

nac_df = pd.read_parquet("/dbfs/FileStore/nft/kaggle/nft_address_cocount", columns = ['nft_address1', 'nft_address2', 'owner_cnts'])

# COMMAND ----------

nac_df.head()

# COMMAND ----------

co_df = pd.read_parquet("/dbfs/FileStore/nft/kaggle/current_owners_proc", columns = ['nft_address', 'token_id', 'owner'])

# COMMAND ----------

co_df.head()

# COMMAND ----------

# MAGIC %md # 행렬곱 준비

# COMMAND ----------

# MAGIC %md ## transpose

# COMMAND ----------

nac_df_t = nac_df.transpose()

# COMMAND ----------

nac_df_t.head()

# COMMAND ----------

# MAGIC %md ## Error(memory)
# MAGIC 
# MAGIC - 메모리 부족 에러 발생. 

# COMMAND ----------

# MAGIC %md ## 1. join 쓰는 방법
# MAGIC 
# MAGIC - join 으로 nft_address1과 owner를 맞춰서 행렬 수 맞춤 + 메모리 절약? 
# MAGIC - 장렬히 실패

# COMMAND ----------

# nac_df.set_index('nft_address1', inplace = True)

# COMMAND ----------

# co_df.set_index('nft_address', inplace = True)

# COMMAND ----------

# df_join = nac_df.join(co_df, how = 'left')

# COMMAND ----------

nac_df = nac_df.drop(['nft_address2'], axis = 1)

# COMMAND ----------

co_df = co_df.drop(['token_id'], axis = 1)

# COMMAND ----------

nac_df.head()

# COMMAND ----------

# MAGIC %md ## 2. 데이터 크기 조절?
# MAGIC 
# MAGIC - 데이터 크기 자체를 조절(object => category)해서 가능한가?
# MAGIC - 장렬히 실패

# COMMAND ----------

# nac_df.info(memory_usage = 'deep')

# COMMAND ----------

# nac_df = nac_df.astype({'nft_address1':'category'})

# COMMAND ----------

# nac_df.info(memory_usage = 'deep')

# COMMAND ----------

# co_df.info(memory_usage = 'deep')

# COMMAND ----------

# # 메모리 사용량 줄이기 co_df ver.
# # object -> category
# co_df.loc[:,['nft_address']] = co_df.loc[:,['nft_address']].astype('category')
# co_df.loc[:,['nft_address']].info(memory_usage='deep')
# # object -> category
# co_df.loc[:,['owner']] = co_df.loc[:,['owner']].astype('category')
# co_df.loc[:,['owner']].info(memory_usage='deep')

# COMMAND ----------

# co_df = co_df.astype({'owner':'category'})

# COMMAND ----------

# co_df = co_df.astype({'nft_address': 'category'})

# COMMAND ----------

# co_df.info(memory_usage = 'deep')

# COMMAND ----------

# co_df.info(memory_usage = 'deep')

# COMMAND ----------

# MAGIC %md ## 3. for문 돌려서?
# MAGIC 
# MAGIC - 구현 가능한가? 

# COMMAND ----------

# nac_df['owner']

# for nft_address1 in nac_df:
    

# COMMAND ----------

# for column_name in df:
#     print(type(column_name))
#     print(column_name)
#     print('======\n')

# COMMAND ----------

# merge_df = pd.DataFrame(columns=["nft_address", "owner_cnts", "owner"])
# for merge_df.loc[len(merge_df)]
# # for문을 돌리면서 together_df.loc[len(together_Df)] ...
# for age in df['age']:

# COMMAND ----------

# for nft_address in nac_df['nft_address1']:
#         if nac_df['nft_address1'] == co_df['nft_address']:
#             nac_df['owner'].append(co_df['owner'])

# COMMAND ----------

# owner_address = (nac_df.nft_address1 == co_df.nft_address ) : # 조건식 작성

# df[owner_address]

# COMMAND ----------

# nac_df.head()

# COMMAND ----------

# nac_df = nac_df.drop(['nft_address2', 'cnts'], axis = 1)

# COMMAND ----------

# co_df = co_df.drop(['token_id'], axis = 1)

# COMMAND ----------

# co_df.head()

# COMMAND ----------

# nac_df.info(memory_usage = 'deep')

# COMMAND ----------

# nac_df = nac_df.astype({'owner_cnts': 'int32'})

# COMMAND ----------

nac_df.iloc[:,0]

# COMMAND ----------

co_df = co_df.drop(['_c0'], axis = 1)

# COMMAND ----------

co_df.head()

# COMMAND ----------

nac_df.iloc[:,0]

# COMMAND ----------

if nac_df.iloc[:,0] == co_df.iloc[:,0]:
    nac_df['owner'].append(co_df.iloc[:, 1])
else: 
    pass

# COMMAND ----------

# 칼럼 정리 - 미사용 칼럼 삭제, 칼럼명 변경, 
rm_col = ['UnixTimeStamp', 'DateTime']

for i in range(len(data_list)):
    for j in range(len(rm_col)) : # 미사용 칼럼 제거
        if rm_col[j] in data_list[i].columns :
            data_list[i].drop(rm_col[j], axis=1, inplace=True)
    for col in range(len(data_list[i].columns)): # 칼럼명 변경
        culName = data_list[i].columns[col]
        changeName = file_name[i] + '_' + culName
        data_list[i].rename(columns={culName : changeName}, inplace=True)       
    print(data_list[i].columns)
    print('=' * 50)

# COMMAND ----------

#  데이터 통합
total = data_list[0]
for i in range(1, len(data_list)):
    total = pd.merge(total, data_list[i], left_index=True, right_index=True, how='left')
#     print(i, '\n', val_list[i-1], val_list[i],'\n')
#     print(len(total))
#     print('='*100)
#     print(total)
#     print('='*100)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

nac_df_t.dot(df_join)

# COMMAND ----------

# co_df2 = spark.sql("""
# select t.*
# from(
# select a.nft_address as nft_address1, b.nft_address as nft_address2, count(distinct a.owner) owner_cnts, count(*) cnts
# from co_df_parquet as a 
# inner join co_df_parquet as b
# on 1=1
# and a.owner = b.owner
# group by a.nft_address, b.nft_address
# ) as t
# where 1=1
# and t.nft_address1 != t.nft_address2
# """)

# OUTPUT_PATH = "/FileStore/nft/kaggle/nft_address_cocount"
# co_df2.repartition(100).write.save(OUTPUT_PATH, format='parquet', mode='overwrite')

# COMMAND ----------

# nac_df.transpose()
# nac_df.rename(columns = nac_df.iloc[0], inplace = True)
# nac_df = nac_df.drop(nac_df.index[0])

# COMMAND ----------

# nac_df.head()

# COMMAND ----------

# nac_df.dot(co_df)

# COMMAND ----------

# MAGIC %md # 다시해보자

# COMMAND ----------

nac_df = pd.read_parquet("/dbfs/FileStore/nft/kaggle/nft_address_cocount", columns = ['nft_address1', 'owner_cnts'])

# COMMAND ----------

nac_df.info(memory_usage = 'deep')

# COMMAND ----------



# COMMAND ----------


