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

Jaccard_nft.head()

# COMMAND ----------

nac_df = pd.read_parquet("/dbfs/FileStore/nft/kaggle/nft_address_cocount")

# COMMAND ----------

nac_df.head()

# COMMAND ----------

nfts_df = pd.read_csv("/dbfs/FileStore/nft/kaggle/nfts.csv")

# COMMAND ----------

nfts_df.head()

# COMMAND ----------

nfts_merge = pd.merge(nac_df, nfts_df, how = 'left', left_on = 'nft_address1', right_on = 'address')

# COMMAND ----------

nfts_merge.head()

# COMMAND ----------

nfts_merge.drop(['Unnamed: 0', 'address', 'symbol'], axis = 1, inplace = True)

# COMMAND ----------

nfts_merge.head()

# COMMAND ----------

co_df = pd.read_parquet("/dbfs/FileStore/nft/kaggle/current_owners_proc")

# COMMAND ----------

co_df = co_df.drop(['_c0'], axis = 1)

# COMMAND ----------

co_df.head()

# COMMAND ----------

co_df = co_df.drop(['token_id'], axis =1)

# COMMAND ----------

nac_df = nac_df.drop(['cnts'], axis = 1)

# COMMAND ----------

# MAGIC %md ## data 만들기(SAR 알고리즘에 넣을)

# COMMAND ----------



# COMMAND ----------

# df_merge = pd.merge(co_df, nac_df, how = 'left', left_on = 'nft_address', right_on = 'nft_address1')
# 스파크로 하는것으로 해결해야할듯. 

# COMMAND ----------

df_merge = pd.merge(nac_df, co_df, how = 'left', left_on = 'nft_address1', right_on = 'nft_address')

# COMMAND ----------

co_df.info()

# COMMAND ----------

# MAGIC %md ### spark로 만들어보기 (메모리 에러)

# COMMAND ----------

nft_address_cocount = spark.read.parquet("/FileStore/nft/kaggle/nft_address_cocount")

# COMMAND ----------

nft_address_cocount.createOrReplaceTempView("nft_address_cocount")

# COMMAND ----------

nft_address_cocount.show(5)

# COMMAND ----------

co_df_s= spark.read.parquet("/FileStore/nft/kaggle/current_owners_proc")

# COMMAND ----------

co_df_s.createOrReplaceTempView("co_df_s")

# COMMAND ----------

co_df_s.show()

# COMMAND ----------

# %sql
# select *
# from nft_address_cocount left outer join co_df_s
# on nft_address_cocount.nft_address1 = co_df_s.nft_address
# where co_df_s.nft_address is Null

# COMMAND ----------

# co_df3 = spark.sql("""
# select nft_address_cocount.nft_address1, nft_address_cocount.nft_address2, nft_address_cocount.owner_cnts, co_df_s.owner
# from nft_address_cocount left outer join co_df_s
# on nft_address_cocount.nft_address1 = co_df_s.nft_address
# """)

# OUTPUT_PATH = "/FileStore/nft/kaggle/nft_address_co_owner"
# co_df3.repartition(100).write.save(OUTPUT_PATH, format='parquet', mode='overwrite')

# COMMAND ----------

co_df4 = spark.sql("""
select t.*
from(
select a.nft_address1, a.nft_address2, a.owner_cnts, b.owner
from nft_address_cocount as a 
left outer join co_df_s as b
on 1=1
and a.nft_address1 = b.nft_address    
) as t
where 1=1
""")

OUTPUT_PATH = "/FileStore/nft/kaggle/nft_4"
co_df4.repartition(100).write.save(OUTPUT_PATH, format='parquet', mode='overwrite')

# COMMAND ----------

naco = pd.read_parquet("/dbfs/FileStore/nft/kaggle/nft_address_co_owner")

# COMMAND ----------

# MAGIC %md # SAR Algorithm

# COMMAND ----------

# 추천할 Top K items 
TOP_K = 10

# NFT data size 100k, 1m, 10m, or 20m
NFT_DATA_SIZE = '100k'

# COMMAND ----------

data = pd.read_parquet("/dbfs/FileStore/nft/kaggle/nft_address_co_owner", size = NFT_DATA_SIZE)

data.loc[:, 'owner_cnts'] = data['owner_cnts'].astype(np.float32)

data.head()

# COMMAND ----------

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=['UserId', 'MovieId', 'Rating', 'Timestamp'],
    title_col='Title'
)

# Convert the float precision to 32-bit in order to reduce memory consumption 
data.loc[:, 'Rating'] = data['Rating'].astype(np.float32)

data.head()

# COMMAND ----------

header = {
    "col_user": "UserId",
    "col_item": "MovieId",
    "col_rating": "Rating",
    "col_timestamp": "Timestamp",
    "col_prediction": "Prediction",
}

# COMMAND ----------

# 수정
header = {
    "col_user": "owner",
    "col_item": "nft_address1",
    "col_rating": "owner_cnts",
    "col_prediction": "Prediction",
}

# COMMAND ----------

train, test = python_stratified_split(data, ratio=0.75, col_user=header["col_user"], col_item=header["col_item"], seed=42)

# COMMAND ----------

# set log level to INFO
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s')

model = SARSingleNode(
    similarity_type="jaccard", 
    time_decay_coefficient=30, 
    time_now=None, 
    timedecay_formula=True, 
    **header
)

# COMMAND ----------

model.fit(train)

# COMMAND ----------

top_k = model.recommend_k_items(test, remove_seen=True)

# COMMAND ----------

top_k_with_titles = (top_k.join(data[['MovieId', 'Title']].drop_duplicates().set_index('MovieId'), 
                                on='MovieId', 
                                how='inner').sort_values(by=['UserId', 'Prediction'], ascending=False))
display(top_k_with_titles.head(10))

# COMMAND ----------

# 수정
top_k_with_titles = (top_k.join(data[['nft_address1', 'nft_name']].drop_duplicates().set_index('nft_address1'), 
                                on='nft_address1', 
                                how='inner').sort_values(by=['owner', 'Prediction'], ascending=False))
display(top_k_with_titles.head(10))

# COMMAND ----------

# MAGIC %md ## Evaluate the results

# COMMAND ----------

# all ranking metrics have the same arguments
args = [test, top_k]
kwargs = dict(col_user='UserId', 
              col_item='MovieId', 
              col_rating='Rating', 
              col_prediction='Prediction', 
              relevancy_method='top_k', 
              k=TOP_K)

eval_map = map_at_k(*args, **kwargs)
eval_ndcg = ndcg_at_k(*args, **kwargs)
eval_precision = precision_at_k(*args, **kwargs)
eval_recall = recall_at_k(*args, **kwargs)

# COMMAND ----------

# all ranking metrics have the same arguments
# 수정
args = [test, top_k]
kwargs = dict(col_user='owner', 
              col_item='nft_address1', 
              col_rating='owner_cnts', 
              col_prediction='Prediction', 
              relevancy_method='top_k', 
              k=TOP_K)

eval_map = map_at_k(*args, **kwargs)
eval_ndcg = ndcg_at_k(*args, **kwargs)
eval_precision = precision_at_k(*args, **kwargs)
eval_recall = recall_at_k(*args, **kwargs)

# COMMAND ----------

print(f"Model:",
      f"Top K:\t\t {TOP_K}",
      f"MAP:\t\t {eval_map:f}",
      f"NDCG:\t\t {eval_ndcg:f}",
      f"Precision@K:\t {eval_precision:f}",
      f"Recall@K:\t {eval_recall:f}", sep='\n')

# COMMAND ----------


