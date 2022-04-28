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

nfts_merge_df = pd.read_csv("/dbfs/FileStore/nft/kaggle/nfts_merge.csv")

# COMMAND ----------

nfts_merge_df = nfts_merge_df.drop(['Unnamed: 0'], axis = 1)

# COMMAND ----------

nfts_merge_df.head()

# COMMAND ----------

nfts = pd.read_csv("/dbfs/FileStore/nft/kaggle/nfts.csv")

# COMMAND ----------

nfts.head()

# COMMAND ----------

nfts['address'].nunique()

# COMMAND ----------

nfts_merge2 = pd.merge(nfts_merge_df, nfts, how = 'left', left_on = 'nft_address2', right_on = 'address')

# COMMAND ----------

nfts_merge2.head()

# COMMAND ----------

nfts_merge2 = nfts_merge2.drop(['Unnamed: 0', 'address', 'symbol'], axis =1)

# COMMAND ----------

nfts_merge2.head()

# COMMAND ----------

# MAGIC %md # Crypto Kitty

# COMMAND ----------

Crypto_j = nfts_merge_df['name'] == 'CryptoKitties'

Crypto_k = nfts_merge_df[Crypto_j]

Crypto_k

# COMMAND ----------

# MAGIC %md # Art Blocks

# COMMAND ----------

ArtBlock_j = nfts_merge_df['name'] == 'Art Blocks'

ArtBlock_s = nfts_merge_df[ArtBlock_j]

ArtBlock_s

# COMMAND ----------

# MAGIC %md # Crypto Kitty vs Art Blocks

# COMMAND ----------

# Crypto Kitties 기준 Art Blocks 유사도
Crypto_by_art = Crypto_k['nft_address2'] == '0xa7d8d9ef8D8Ce8992Df33D8b8CF4Aebabd5bD270'

Crypto_art = Crypto_k[Crypto_by_art]

Crypto_art

# COMMAND ----------

# Art Blocks 기준 Crypto Kitties 유사도

Art_by_Crypto = ArtBlock_s['nft_address2'] == '0x06012c8cf97BEaD5deAe237070F9587f8E7A266d'

Art_Crypto = ArtBlock_s[Art_by_Crypto]

Art_Crypto

# COMMAND ----------

# MAGIC %md ## Cryptokitty Art Blocks top 10

# COMMAND ----------

# Cryptokitty Art Blocks 

Crypto_j2 = nfts_merge2['name_x'] == 'CryptoKitties'

Crypto_k2 = nfts_merge2[Crypto_j2]

Crypto_k2.head(10)

# COMMAND ----------

pd.options.display.float_format = '{:.10f}'.format

# COMMAND ----------

Crypto_k2.sort_values('Lift', ascending = False).head(10)

# COMMAND ----------

Crypto_k2.sort_values('Lift', ascending = True).head(10)

# COMMAND ----------

Crypto_k2.sort_values('Jaccard', ascending = False).head(10)

# COMMAND ----------

Crypto_k2.sort_values('confidence', ascending = False).head(10)

# COMMAND ----------

ArtBlock_j2 = nfts_merge2['name_x'] == 'Art Blocks'

ArtBlock_s2 = nfts_merge2[ArtBlock_j2]

ArtBlock_s2.head(10)

# COMMAND ----------

ArtBlock_s2.sort_values('Jaccard', ascending = False).head(10)

# COMMAND ----------

ArtBlock_s2.sort_values('Lift', ascending = False).head(10)

# COMMAND ----------

ArtBlock_s2.sort_values('confidence', ascending = False).head(10)

# COMMAND ----------

# MAGIC %md ## pt용 결과도출 frame

# COMMAND ----------

Crypto_k2.head()

# COMMAND ----------

drop_k = Crypto_k2.drop(['nft_address1', 'nft_address2', 'owner_cnts', 'name_x'], axis = 1)

# COMMAND ----------

drop_k = drop_k.drop(['confidence'], axis = 1)

# COMMAND ----------

drop_k = drop_k.drop(['cnts'], axis = 1)

# COMMAND ----------

drop_k.head()

# COMMAND ----------

drop_lift = drop_k.drop(['Lift'], axis = 1)

# COMMAND ----------

Jaccard_k = drop_lift.sort_values(['Jaccard'], ascending = False)

# COMMAND ----------

Jaccard_k = Jaccard_k[['name_y', 'Jaccard']]

# COMMAND ----------

Jaccard_k = Jaccard_k.rename({'name_y': 'NFT'}, axis=1)

# COMMAND ----------

Jaccard_k.head()

# COMMAND ----------

Jaccard_k.index = Jaccard_k.index+1

# COMMAND ----------

Jaccard_k.head(10)

# COMMAND ----------

drop_Jaccard = drop_k.drop(['Jaccard'], axis = 1)

Jaccard_l = drop_k.sort_values(['Lift'], ascending = False)

Jaccard_l = Jaccard_l[['name_y', 'Lift']]

Jaccard_l = Jaccard_l.rename({'name_y': 'NFT'}, axis=1)

# COMMAND ----------

Jaccard_l.reset_index(drop = True, inplace= True)

# COMMAND ----------

Jaccard_l.index = Jaccard_l.index+1

# COMMAND ----------

Jaccard_l.head(10)

# COMMAND ----------

# MAGIC %md # graph

# COMMAND ----------

crypto_kitties = Crypto_k['confidence']
Art_blocks = ArtBlock_s['confidence']

plt.plot(crypto_kitties, color = 'r', label = 'Crypto Kitties')
plt.plot(Art_blocks, color = 'c', label = 'Art Blocks')

# COMMAND ----------

crypto_kitties = Crypto_k['Jaccard']
Art_blocks = ArtBlock_s['Jaccard']

plt.plot(crypto_kitties, color = 'r', label = 'Crypto Kitties')
plt.plot(Art_blocks, color = 'c', label = 'Art Blocks')

# COMMAND ----------

crypto_kitties = Crypto_k['Lift']
Art_blocks = ArtBlock_s['Lift']

plt.plot(crypto_kitties, color = 'r', label = 'Crypto Kitties')
plt.plot(Art_blocks, color = 'c', label = 'Art Blocks')

# COMMAND ----------

crypto_kitties = Crypto_k['owner_cnts']
Art_blocks = ArtBlock_s['owner_cnts']

plt.plot(crypto_kitties, color = 'r', label = 'Crypto Kitties')
plt.plot(Art_blocks, color = 'c', label = 'Art Blocks')

# COMMAND ----------

Crypto_k["Jaccard"].plot()

# COMMAND ----------

ArtBlock_s['Jaccard'].plot()
