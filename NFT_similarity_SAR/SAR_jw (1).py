# Databricks notebook source
# MAGIC %md #Setting

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

# MAGIC %md #Data load
# MAGIC 
# MAGIC - nft_address_cocount 설명
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

nft_address_cocount.printSchema()

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from nft_address_cocount
# MAGIC order by 3 desc

# COMMAND ----------

# MAGIC %sql
# MAGIC select max(owner_cnts)
# MAGIC from nft_address_cocount

# COMMAND ----------

# MAGIC %sql
# MAGIC select min(owner_cnts)
# MAGIC from nft_address_cocount

# COMMAND ----------

# MAGIC %sql
# MAGIC select avg(owner_cnts)
# MAGIC from nft_address_cocount

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(owner_cnts)
# MAGIC from nft_address_cocount

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(nft_address1)
# MAGIC from nft_address_cocount

# COMMAND ----------

# MAGIC %sql
# MAGIC select stddev_samp(owner_cnts)
# MAGIC from nft_address_cocount
# MAGIC -- 표준편차 구한것임. 

# COMMAND ----------

# row가 
# 4869110개 있음.

# COMMAND ----------

# MAGIC %md # 분포 

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

# MAGIC %md ## owner_cnt 10 이상, 10 미만

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

# owner_cnt 10 이상 = 688424
# owner_cnt 10 미만 = 4180686
# owner_cnt 1과 같음 = 2698738

# 따라서 owner_cnt 10 미만 인 수 중에 반이상은 1임. 

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

# MAGIC %md ## 특정 nft에 대해서 owner cnt 수 scatter로 찍어본것

# COMMAND ----------

# plt.scatter(df_small['nft_address1'], df_small['owner_cnts'], c='green')
# plt.show()
# 오래걸리는거 생각을 해보면 저번처럼 나누기는 해야할듯. 
# 10개 이상 / 10개 미만으로 함 보겠음.

# COMMAND ----------

# plt.figure(figsize=(24,24))
# plt.scatter(df_small['owner_cnts'], df_small['nft_address1'], c='green')
# plt.show()

# COMMAND ----------

# plt.scatter(nac_df['nft_address1'] == '0x57f1887a8BF19b14fC0dF6Fd9B2acc9Af147eA85', nac_df['owner_cnts'], c = 'green')
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## SAR algorithm
# MAGIC 
# MAGIC - 매우 높은 수준에서 두 개의 중간 행렬이 생성되고 추천 점수 집합을 생성하는 데 사용
# MAGIC - An item similarity matrix S estimates item-item relationships.
# MAGIC - An affinity matrix A estimates user-item relationships.
# MAGIC - Recommendation scores are then created by computing the matrix multiplication A * S
# MAGIC - Optional steps (e.g. "time decay" and "remove seen items") are described in the details below.
# MAGIC 
# MAGIC ### Compute item co-occurrence and item similarity
# MAGIC 
# MAGIC - SAR은 항목 간 동시 발생 데이터를 기반으로 유사성을 정의
# MAGIC - It is symmetric
# MAGIC - It is nonnegative
# MAGIC - The occurrences are at least as large as the co-occurrences. I.e., the largest element for each row (and column) is on the main diagonal
# MAGIC 
# MAGIC - 동시 발생 행렬 C에는 다음과 같은 속성이 있습니다.
# MAGIC 
# MAGIC - `Jaccard`
# MAGIC - `lift`
# MAGIC - `counts`
# MAGIC 
# MAGIC ![](https://drive.google.com/uc?id=1n3aKgY1X0Dp0TsYBZIAO0WWYYGhBAco9)

# COMMAND ----------

# MAGIC %md # jaccard

# COMMAND ----------

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

# COMMAND ----------

# def jaccard_similarity(df, address1, address2):
#     add1 = df[df['nft_address'] == 'address1']
#     add2 = df[df['nft_address'] == 'address2']
#     ab = add1['nft_address'].count() + add2['nft_address'].count()

# COMMAND ----------

co_df = pd.read_csv("/dbfs/FileStore/nft/kaggle/current_owners.csv", low_memory = False)

# COMMAND ----------

co_df = co_df.drop(['Unnamed: 0'], axis = 1)

# COMMAND ----------

# 0x57f1887a8BF19b14fC0dF6Fd9B2acc9Af147eA85 이 nft_address만 가지고 있는 행 뽑아오기 
a = co_df[co_df['nft_address'] == '0x57f1887a8BF19b14fC0dF6Fd9B2acc9Af147eA85']

# COMMAND ----------

a['nft_address'].count()

# COMMAND ----------

# 0xa7d8d9ef8D8Ce8992Df33D8b8CF4Aebabd5bD270 이 nft_address만 가지고 있는 행 뽑아옴 
b = co_df[co_df['nft_address'] == '0xa7d8d9ef8D8Ce8992Df33D8b8CF4Aebabd5bD270']

# COMMAND ----------

b.count()

# COMMAND ----------

# 정말 단순계산으로 해보자 
4774 / (145303 + 98652 - 4774)

# COMMAND ----------

nac_df.sort_values(by = ['owner_cnts'], axis = 0, ascending = False).head(10)

# COMMAND ----------

# MAGIC %md ## 분모 , 분자 테이블 조인
# MAGIC 
# MAGIC ```
# MAGIC - 멘토님 지시사항
# MAGIC 분모테이블 이랑 분자테이블 조인걸어서 
# MAGIC 분모테이블 address별로 distinct 한 owner값 조인하면 됨. 
# MAGIC left join 두번 걸면 됨. address 1, address2 에
# MAGIC 특정 address 별로 최상위 10개 lift 했을 때 상위 10개 confidence 상위 10개 
# MAGIC nft address1이 기준 nft라고 함
# MAGIC nft address2가 추천 nft라고 하자. 
# MAGIC 
# MAGIC 기준nft 기준으로 추천 nft address 결과가 다 다를거임. 
# MAGIC ```

# COMMAND ----------

# 분모 테이블이 
# cii + cjj - cij
# cii = address1 address1 owner cnts값 
# cjj = address2 address2 owner cnts값
# cij = address1 address2 owner cnts값 

# COMMAND ----------

nac_df.head()

# COMMAND ----------

select userid, (select name from member where userid = memberOther.userid)as name, blood from memberOther where blood='AB'

# COMMAND ----------

# MAGIC %sql
# MAGIC select * 
# MAGIC from nft_address_cocount as owner_cnts, from nft_address1 where ('nft_address1' = 'nft_address2')

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from nft_address_cocount
# MAGIC where nft_address1 = nft_address2

# COMMAND ----------

# MAGIC %sql
# MAGIC select nft_address_cocount.nft_address1, nft_address_cocount.nft_address2
# MAGIC from nft_address_cocount
# MAGIC inner join nft_address_cocount.nft_address2
# MAGIC on nft_address_cocount.nft_address1 = nft_address_cocount.nft_address2

# COMMAND ----------

# MAGIC %md ## 같은거 owner_cnt 있는 데이터프레임 만들기

# COMMAND ----------

co_df = spark.read.csv("/FileStore/nft/kaggle/current_owners.csv", header="true", inferSchema="true")

# COMMAND ----------

OUTPUT_PATH = "/FileStore/nft/kaggle/current_owners_proc"
co_df_parquet = spark.read.parquet(OUTPUT_PATH)
co_df_parquet.createOrReplaceTempView("co_df_parquet")

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from co_df_parquet
# MAGIC limit 10;

# COMMAND ----------

co_df.createOrReplaceTempView("co_df")

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
# and t.nft_address1 = t.nft_address2
# """)

# OUTPUT_PATH = "/FileStore/nft/kaggle/nft_address_cocount_2"
# co_df2.repartition(100).write.save(OUTPUT_PATH, format='parquet', mode='overwrite')

# COMMAND ----------

nft_address_cocount_2 = spark.read.parquet("/FileStore/nft/kaggle/nft_address_cocount_2")

# COMMAND ----------

nft_address_cocount_2.show()

# COMMAND ----------

nft_address_cocount_2.createOrReplaceTempView("nft_address_cocount_2")

# COMMAND ----------

nft_address_cocount_2.printSchema()

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from nft_address_cocount_2
# MAGIC order by 3 desc

# COMMAND ----------

# MAGIC %md ##분모 만들기

# COMMAND ----------

분모 테이블이 
cii + cjj - cij
cii = address1 address1 owner cnts값 
cjj = address2 address2 owner cnts값
cij = address1 address2 owner cnts값 

# COMMAND ----------

390002 / (64717 + 64717 - 390002)

# COMMAND ----------

# 0x57f1887a8BF19b14fC0dF6Fd9B2acc9Af147eA85

# COMMAND ----------

h = nac_df[nac_df['nft_address1'] == '0x57f1887a8BF19b14fC0dF6Fd9B2acc9Af147eA85']

# COMMAND ----------

h['owner_cnts'].sum()

# COMMAND ----------

nac_df.head()

# COMMAND ----------

# MAGIC %md # 뭔가 문제가 있다 다시 해보자

# COMMAND ----------

nac_df['nft_address1'] == nac_df['nft_address2']

# COMMAND ----------

b = nac_df[nac_df['nft_address1'] == nac_df['nft_address2']]

# COMMAND ----------

b

# COMMAND ----------

c = nac_df['nft_address1'] == '0x0B0b186841C55D8a09d53Db48dc8cab9dbf4Dbd6', nac_df['nft_address2'] == '0x0B0b186841C55D8a09d53Db48dc8cab9dbf4Dbd6'

# COMMAND ----------

# MAGIC %md ## 다시
# MAGIC 
# MAGIC - 멘토님 지시사항
# MAGIC 
# MAGIC ```
# MAGIC 분모테이블 이랑 분자테이블 조인걸어서 
# MAGIC 분모테이블 address별로 distinct 한 owner값 조인하면 됨. 
# MAGIC left join 두번 걸면 됨. address 1, address2 에
# MAGIC 특정 address 별로 최상위 10개 lift 했을 때 상위 10개 confidence 상위 10개 
# MAGIC nft address1이 기준 nft라고 함
# MAGIC nft address2가 추천 nft라고 하자. 
# MAGIC 
# MAGIC 기준nft 기준으로 추천 nft address 결과가 다 다를거임. 
# MAGIC ```

# COMMAND ----------

# 분자 테이블이 nft_address_cocount라고 생각했는데 아닌가???? 

# COMMAND ----------

# cii가 기준 address 기준으로 추천 address owner_cnts
# cjj가 추천 address 기준으로 기준 address owner_cnts 
# 였던건가????
# 그럼 cij랑 같은거 아닌가..? 

# COMMAND ----------

nac_df['nft_address1'].nunique()

# COMMAND ----------

u = nac_df[nac_df['nft_address1'] == '0x57f1887a8BF19b14fC0dF6Fd9B2acc9Af147eA85']

# COMMAND ----------

u['owner_cnts'].sum()

# COMMAND ----------

ea = nac_df[nac_df['nft_address1'] == '0x1dfe7Ca09e99d10835Bf73044a23B73Fc20623DF']

# COMMAND ----------

ea['owner_cnts'].sum()

# COMMAND ----------

# 0x1dfe7Ca09e99d10835Bf73044a23B73Fc20623DF
# 이거 기준으로 다시 구해보자 

# COMMAND ----------

148241 / (21596 + 21596 - 148241)

# COMMAND ----------

# MAGIC %md #피드백 받은 내용 적용

# COMMAND ----------

# MAGIC %md ## 재정의
# MAGIC * Jaccard = cij / (A distinct owner + B distinct owner - cij)
# MAGIC * Lift = cij / (A distinct owner * B distinct owner)
# MAGIC * Confidence = cij / A distinct owner
# MAGIC * count = cij

# COMMAND ----------

nac_df.head()

# COMMAND ----------

co_df.show()
