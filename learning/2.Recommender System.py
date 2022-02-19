# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Pyspark
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC 
# MAGIC ## 참고 문서
# MAGIC 
# MAGIC [Databricks Notebook Tutorial]()
# MAGIC 
# MAGIC [Python Microsoft Recommenders](https://github.com/microsoft/recommenders)
# MAGIC 
# MAGIC [Matrix Factorization (2)](https://velog.io/@vvakki_/Matrix-Factorization-2)

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

# MAGIC %md
# MAGIC ### 추천시스템 맛보기

# COMMAND ----------

# MAGIC %%bash
# MAGIC git clone https://github.com/Microsoft/Recommenders

# COMMAND ----------

# MAGIC %%bash
# MAGIC ls -alh

# COMMAND ----------

# MAGIC %%bash
# MAGIC #rm -rf Recommenders

# COMMAND ----------

# MAGIC %%bash
# MAGIC ls

# COMMAND ----------

# set the environment path to find Recommenders
import sys
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import sys
import pandas as pd
sys.path.append("/databricks/driver/Recommenders")


import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.ml.tuning import CrossValidator
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import FloatType, IntegerType, LongType

from recommenders.datasets import movielens
from recommenders.utils.spark_utils import start_or_get_spark
from recommenders.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation
from recommenders.tuning.parameter_sweep import generate_param_grid
from recommenders.datasets.spark_splitters import spark_random_split

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("PySpark version: {}".format(pyspark.__version__))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 데이터 로드
# MAGIC 
# MAGIC [MovieLens](https://grouplens.org/datasets/movielens/)
# MAGIC 
# MAGIC - 데이터 세트 (ml-25m) 는 영화 추천 서비스 인 MovieLens의 별 5 개 등급 및 자유 텍스트 태그 지정 활동을 설명
# MAGIC - 62,423 영화에 걸쳐 25,000,095개 rating 및 1,093,360개 tag 응용 프로그램을 포함. 1995년 1월 9일~ 2019년 11월 21일 사이에 162,541 명의 사용자가 데이터
# MAGIC - 사용자는 포함을 위해 무작위로 선택. 선택된 모든 사용자는 최소 20 개의 영화를 평가. 인구 통계학적 정보는 포함되지 않음 각 사용자는 ID로 표시되며 다른 정보는 제공되지 않음

# COMMAND ----------

# MAGIC %md
# MAGIC - 간단한 유사도 알고리즘
# MAGIC   - SAR (Smart Adaptive Recommendations) 알고리즘
# MAGIC   - SAR은 사용자 거래 내역을 기반으로 개인화 된 추천을위한 빠르고 확장 가능한 적응형 알고리즘. 항목 간의 유사성을 이해하고 사용자가 기존에 선호하는 항목에 유사한 항목을 추천함으로써 구동

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

# MAGIC %md
# MAGIC ### Compute user affinity scores
# MAGIC 
# MAGIC - SAR의 선호도 매트릭스는 각 개별 사용자와 사용자가 이미 상호 작용 한 항목 간의 관계 강도를 캡처. SAR은 사용자의 선호도에 영향을 미칠 수있는 두 가지 요소를 통합
# MAGIC - user-item interaction through differential weighting of different events
# MAGIC - user-item event occurred (e.g. it may discount the value of events that take place in the distant past.
# MAGIC - Remove seen item
# MAGIC - Top-k item calculation

# COMMAND ----------

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# COMMAND ----------

MOVIELENS_DATA_SIZE = "100k"

COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_PREDICTION = "prediction"
COL_TIMESTAMP = "Timestamp"

schema = StructType(
    (
        StructField(COL_USER, IntegerType()),
        StructField(COL_ITEM, IntegerType()),
        StructField(COL_RATING, FloatType()),
        StructField(COL_TIMESTAMP, LongType()),
    )
)

spark = start_or_get_spark("ALS Deep Dive", memory="8g")

dfs = movielens.load_spark_df(spark=spark, size=MOVIELENS_DATA_SIZE, schema=schema)

# COMMAND ----------

data.count()

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from data
# MAGIC limit 5

# COMMAND ----------

# MAGIC %sql
# MAGIC select Rating, count(*) cnts, count(distinct UserId) userid, count(distinct movieid) movieid 
# MAGIC from data
# MAGIC group by Rating
# MAGIC order by 1

# COMMAND ----------

# MAGIC %md
# MAGIC - python_stratified_split
# MAGIC 
# MAGIC """Pandas stratified splitter.
# MAGIC     
# MAGIC     For each user / item, the split function takes proportions of ratings which is
# MAGIC     specified by the split ratio(s). The split is stratified.
# MAGIC     Args:
# MAGIC         data (pd.DataFrame): Pandas DataFrame to be split.
# MAGIC         ratio (float or list): Ratio for splitting data. If it is a single float number
# MAGIC             it splits data into two halves and the ratio argument indicates the ratio of
# MAGIC             training data set; if it is a list of float numbers, the splitter splits
# MAGIC             data into several portions corresponding to the split ratios. If a list is
# MAGIC             provided and the ratios are not summed to 1, they will be normalized.
# MAGIC         seed (int): Seed.
# MAGIC         min_rating (int): minimum number of ratings for user or item.
# MAGIC         filter_by (str): either "user" or "item", depending on which of the two is to
# MAGIC             filter with min_rating.
# MAGIC         col_user (str): column name of user IDs.
# MAGIC         col_item (str): column name of item IDs.
# MAGIC     Returns:
# MAGIC         list: Splits of the input data as pd.DataFrame.
# MAGIC """

# COMMAND ----------

header = {
    "col_user": "UserId",
    "col_item": "MovieId",
    "col_rating": "Rating",
    "col_timestamp": "Timestamp",
    "col_prediction": "Prediction",
}
data_df = data.toPandas()

data_df.loc[:, 'Rating'] = data_df['Rating'].astype(np.float32)
train, test = python_stratified_split(data_df, ratio=0.75, col_user=header["col_user"], col_item=header["col_item"], seed=42)

# COMMAND ----------

print("""
Train:
Total Ratings: {train_total}
Unique Users: {train_users}
Unique Items: {train_items}

Test:
Total Ratings: {test_total}
Unique Users: {test_users}
Unique Items: {test_items}
""".format(
    train_total=len(train),
    train_users=len(train['UserId'].unique()),
    train_items=len(train['MovieId'].unique()),
    test_total=len(test),
    test_users=len(test['UserId'].unique()),
    test_items=len(test['MovieId'].unique()),
))

# COMMAND ----------

# MAGIC %md
# MAGIC - parameter values
# MAGIC 
# MAGIC |Parameter|Value|Description|
# MAGIC |---------|---------|-------------|
# MAGIC |`similarity_type`|`jaccard`|Method used to calculate item similarity.|
# MAGIC |`time_decay_coefficient`|30|Period in days (term of $T$ shown in the formula of Section 1.2)|
# MAGIC |`time_now`|`None`|Time decay reference.|
# MAGIC |`timedecay_formula`|`True`|Whether time decay formula is used.|
# MAGIC 
# MAGIC Time Decay
# MAGIC ![](https://render.githubusercontent.com/render/math?math=a_%7Bij%7D%3D%5Csum_k%20w_k%20%5Cleft%28%5Cfrac%7B1%7D%7B2%7D%5Cright%29%5E%7B%5Cfrac%7Bt_0-t_k%7D%7BT%7D%7D&mode=display)
# MAGIC 
# MAGIC - The (1/2)^n scaling factor causes the parameter T to serve as a half-life: events T units before t_0 will be given half the weight as those taking place at t_0

# COMMAND ----------

model = SARSingleNode(
    similarity_type="jaccard", 
    time_decay_coefficient=30, 
    time_now=None, 
    timedecay_formula=True, 
    **header
)

# COMMAND ----------

data_df.head()

# COMMAND ----------

model.fit(train)

# COMMAND ----------

top_k = model.recommend_k_items(test, remove_seen=True)

# COMMAND ----------

top_k.shape

# COMMAND ----------

top_k.head(n=30)

# COMMAND ----------

test.head()

# COMMAND ----------

top_k_with_titles = (top_k.join(data_df[['MovieId']].drop_duplicates().set_index('MovieId'), 
                                on='MovieId', 
                                how='inner').sort_values(by=['UserId', 'Prediction'], ascending=False))
display(top_k_with_titles.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate the results
# MAGIC 
# MAGIC [Ranking System 이론](https://iamksu.tistory.com/72)
# MAGIC 
# MAGIC - Precision (정밀도) : tp / (tp + fp)  (true로 판단한 횟수 중에서 실제 true 횟수)
# MAGIC - Recall (재현율) : tp / (tp + fn) (실제 정답의 true 중 얼마나 많은 true을 찾아냈는지?)
# MAGIC 
# MAGIC ![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/350px-Precisionrecall.svg.png)
# MAGIC 
# MAGIC - Precision@k - top의 k개 결과로 precision (정밀도) 계산, 정밀도@K는 랭킹을 구한 후 앞에서부터 K개의 정밀도를 검사, 랭킹은 높은 순위를 가장 신경을 많이쓰므로 상대적으로 중요하지 않은 순위들을 계산하지 않아 효율적
# MAGIC - Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
# MAGIC - 관련이 있는 경우 O, 관련이 없는 경우 X라고 했을 때
# MAGIC     - O, X, O, O, O, X 이라고 하면
# MAGIC     - Predcision 3의 값은 O, X, O중에 O의 갯수 2/3
# MAGIC     - Predcision 4의 값은 O, X, O, O중에 O의 갯수 3/4
# MAGIC     - Predcision 5의 값은 O, X, O, O, O중에 O의 갯수 4/5
# MAGIC     
# MAGIC 
# MAGIC - Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
# MAGIC - Mean Average Precision
# MAGIC     -  평균 정밀도의 평균을 취하여 여러 쿼리의 순위를 요약
# MAGIC     -  average precision query 1 = (1.0 + 0.67 + 0.5 + 0.44 + 0.5) / 5 = 0.62
# MAGIC     -  average precision query 2 = (0.5 + 0.4 + 0.43) / 3 = 0.44
# MAGIC     -  ( 0.62 + 0.44 ) / 2 = 0.53
# MAGIC 
# MAGIC [MAP](https://sites.google.com/site/hyunguk1986/personal-study/-ap-map-recall-precision)
# MAGIC     
# MAGIC - NDCG : 상위 1,2 사이는 큰 차이지만 낮은 순위의 차이는 거의 없음 (precision, recall 방법 측정 한계를 개선하기 위해)
# MAGIC     - 랭킹이 높을 수록 높은 가중치를 받아 계산한다.
# MAGIC     
# MAGIC - 추천 평가에 대해서는 다음의 블로그를 같이 보자
# MAGIC - https://ddiri01.tistory.com/321
# MAGIC - https://medium.com/@junhoher/%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C-%EB%91%90-%EA%B0%80%EC%A7%80-recall-k-%EB%B0%8F-precision-k-6b2032e2e360

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

print(f"Model:",
      f"Top K:\t\t {TOP_K}",
      f"MAP:\t\t {eval_map:f}",
      f"NDCG:\t\t {eval_ndcg:f}",
      f"Precision@K:\t {eval_precision:f}",
      f"Recall@K:\t {eval_recall:f}", sep='\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Matrix factorization for collaborative filtering problem
# MAGIC 
# MAGIC ![](https://media.vlpt.us/images/vvakki_/post/efb8e47e-25c8-48bb-909b-a579f6b67b02/image.png)
# MAGIC 
# MAGIC - Matrix Factorization은 
# MAGIC   - User-Item Matrix를 F차원의 User와 Item의 latent factor 행렬곱으로 분해하는 방법을 말함, User-Item Matrix의 유저 u의 아이템 i에 대한 선호도는 다음과 같이 User/Item Latent Matrix의 벡터의 곱으로 표현될 수 있음
# MAGIC 
# MAGIC - 평점 r은 사용자 i가 아이템 j을 얼마나 좋게 평가하는 지를 나타냄 (평점은 유틸리티 행렬 R에 저장)
# MAGIC     - 행 i는 사용자 i가 평가한 아이템 목록이 되고 열 j는 아이템 j를 평가한 모든 사용자 목록이 됨
# MAGIC     - P (N * K)는 특징 공간에서의 사용자 행렬, Q(M * K)는 특징 공간에서의 아이템의 투영 (projection)
# MAGIC     - 낮은 차원 (K) 으로 문제를 축소해 정의한 비용함수를 최소화 하는 P와 Q를 찾을 수 있음 
# MAGIC - 행렬 인수 분해는 일반적인 기술. 기본적으로 행렬 인수 분해 알고리즘은 더 낮은 차원에서 본질적인 사용자 및 항목 속성을 나타내는 잠재 요인을 찾으려고 함. 따라서 학습 결과는 가능한 한 관찰된 등급에 가깝게 분해 결과를 수렴하도록 개발. 또한 오버피팅 문제를 피하기 위해 학습 과정을 정규화. 예를 들어, 이러한 행렬 인수 분해 알고리즘의 기본 형태는 다음과 같음
# MAGIC 
# MAGIC $$\min\sum(r_{u,i} - q_{i}^{T}p_{u})^2 + \lambda(||q_{i}||^2 + ||p_{u}||^2)$$
# MAGIC 
# MAGIC where lambda is a the regularization parameter. 
# MAGIC 
# MAGIC - explict (명시적) 등급을 사용할 수 없는 경우 implicit (암시적) 등급은 일반적으로 사용자의 이전 항목 (예 : 클릭 수, 조회수, 구매수 등)과의 상호 작용에서 파생
# MAGIC 
# MAGIC $$\min\sum c_{u,i}(p_{u,i} - q_{i}^{T}p_{u})^2 + \lambda(||q_{i}||^2 + ||p_{u}||^2)$$
# MAGIC 
# MAGIC - 암묵적인 등급을 설명하기 위해, 원래의 행렬인수 분해 알고리즘은 다음과 같이 공식화 될 수 있음 (Cu,i, Pu,i)
# MAGIC 
# MAGIC ![](https://render.githubusercontent.com/render/math?math=c_%7Bu%2Ci%7D%3D1%2B%5Calpha%20r_%7Bu%2Ci%7D&mode=inline)
# MAGIC 
# MAGIC ![](https://render.githubusercontent.com/render/math?math=p_%7Bu%2Ci%7D%3D1&mode=inline) (if ![](https://render.githubusercontent.com/render/math?math=r_%7Bu%2Ci%7D%26gt%3B0&mode=inline))
# MAGIC 
# MAGIC ![](https://render.githubusercontent.com/render/math?math=p_%7Bu%2Ci%7D%3D0&mode=inline) (if ![](https://render.githubusercontent.com/render/math?math=r_%7Bu%2Ci%7D%3D0&mode=inline))
# MAGIC 
# MAGIC ![](https://render.githubusercontent.com/render/math?math=r_%7Bu%2Ci%7D&mode=inline) 는 numerical representation of users' preferences (e.g., number of clicks, etc.).
# MAGIC 
# MAGIC - 데이터가 존재할 경우 선호함을 나타내는 1, 반대의 경우 0으로 바꿈
# MAGIC - C u,i는 신뢰도를 나타내고, 선호하지 않는 0인 경우에도 계산할 수 있도록 구현 (낮은 c값을 갖게되어 loss에 포함되지만 영향력이 작음)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternating Least Square (ALS)
# MAGIC 
# MAGIC - User Latent Matrix와 Item Latent Matrix 를 둘다 모르기 때문에, Reqularized Squared Error는 Convex 하지 않지만, 만약 둘 중 하나를 고정시키면 2차식의 최적화 문제로 해결할 수 있음
# MAGIC   - 작은 랜덤값으로 초기화 (User Latent Matrix, Item Latent Matrix)
# MAGIC   - 둘 중 하나를 상수로 고정시켜 Loss Function을 Convex Function으로 만들고
# MAGIC   - 그 다음에 미분하여 미분 값을 0으로 만드는  User Latent Matrix, Item Latent Matrix 행렬을 계산
# MAGIC   - 이후 이 과정을 반복해서 최적값을 찾음 (반복은 10~15회 정도, 데이터마다 다름)
# MAGIC 
# MAGIC - ALS의 기본 개념은 최적화를 위해 한 번에 q 및 p 중 하나를 배우고 다른 하나는 일정하게 유지하는 것
# MAGIC   - 각 반복에서 목표를 볼록하고 해결할 수 있게함. q와 p를 번갈아 가며 최적으로 수렴하면 중지
# MAGIC   - 각 반복 계산은 병렬화 및 / 또는 분산 될 수 있으므로 알고리즘이 데이터 세트가 크고 사용자 항목 등급 매트릭스가 매우 희박한 사용 사례에 적합하게 만듦 (추천 시나리오에서 일반적으로 사용됨)
# MAGIC   
# MAGIC ![](http://aikorea.org/cs231n/assets/svm1d.png)
# MAGIC 
# MAGIC - 알고리즘이 데이터 세트가 크고 따라서 사용자 항목 평가 매트릭스가 (추천 시나리오에서와 같이) 매우 희소한 사용 사례에 바람직함. ALS와 그것의 분산 계산에 대한 포괄적 인 토론은 
# MAGIC [논문](http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf)에서 찾을 수 있음

# COMMAND ----------

RANK = 10
MAX_ITER = 15
REG_PARAM = 0.05

# COMMAND ----------

K = 10

# COMMAND ----------

COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_PREDICTION = "prediction"
COL_TIMESTAMP = "Timestamp"

schema = StructType(
    (
        StructField(COL_USER, IntegerType()),
        StructField(COL_ITEM, IntegerType()),
        StructField(COL_RATING, FloatType()),
        StructField(COL_TIMESTAMP, LongType()),
    )
)

spark = start_or_get_spark("ALS Deep Dive", memory="16g")

dfs = movielens.load_spark_df(spark=spark, size="100k", schema=schema, dbutils=dbutils)

# COMMAND ----------

dfs.show(10)

# COMMAND ----------

dfs.groupby('UserId').count().show()

# COMMAND ----------

dfs.groupby('MovieId').count().show()

# COMMAND ----------

from reco_utils.dataset.spark_splitters import spark_random_split
dfs_train, dfs_test = spark_random_split(dfs, ratio=0.75, seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC - 사용자가 테스트 데이터로 견고한 평가를 하는 것을 허용하도록 Spark ALS 모델을 사용하는 것이 중요. 
# MAGIC   - 콜드 스타트가 있는 경우 Spark ALS 구현을 사용하면 예측 결과에 대한 평가가 건전하다는 것을 확인하기 위해 콜드 스타트를 삭제할 수 있음
# MAGIC 
# MAGIC - Spark 문서에 따르면 콜드 스타트 전략
# MAGIC   - ALSModel을 사용하여 예측을 수행 할 때 모델을 학습하는 동안 존재하지 않은 테스트 데이터 집합의 사용자 및 / 또는 항목을 발견하는 것이 일반적입니다. 일반적으로 두 가지 시나리오에서 발생합니다.
# MAGIC   - 프로덕션 환경에서는 평가 기록이 없고 모델 훈련을 받지 않은 신규 사용자 또는 항목 ( "콜드 스타트 문제")
# MAGIC   - 교차 유효성 검사 도중 데이터는 학습 집합과 평가 집합으로 나뉩니다. Spark의 CrossValidator 또는 TrainValidationSplit에서와 같이 간단한 임의의 분할을 사용하는 경우 실제로는 훈련 세트에 포함되지 않은 평가 집합의 사용자 및 / 또는 항목을 만나는 것이 일반적
# MAGIC 
# MAGIC - 기본적으로 Spark는 사용자 및 / 또는 항목 요소가 모델에 없을 때 ALSModel.transform 중에 NaN 예측을 할당합니다. 이는 새로운 사용자 또는 항목을 나타내므로 프로덕션 시스템에서 유용 할 수 있으므로 시스템은 예측으로 사용할 일부 폴백에 대한 결정을 내릴 수 있음
# MAGIC   - 그러나 교차 유효성 검사 중에는 NaN 예측 값으로 인해 (예 : RegressionEvaluator 사용시) 평가 메트릭에 대한 NaN 결과가 나오므로 바람직하지 않습니다. 이로 인해 모델 선택이 불가능 해집니다.
# MAGIC   - Spark을 사용하면 ColdStartStrategy 매개 변수를 "drop"으로 설정하여 NaN 값이 포함 된 예측 데이터 프레임의 행을 삭제할 수 있습니다. 그런 다음 평가 메트릭은 NaN 이외의 데이터를 통해 계산되어 유효합니다. 이 매개 변수의 사용법은 아래 예제에 설명되어 있음

# COMMAND ----------

als = ALS(
    maxIter=MAX_ITER, 
    rank=RANK,
    regParam=REG_PARAM, 
    userCol=COL_USER, 
    itemCol=COL_ITEM, 
    ratingCol=COL_RATING, 
    coldStartStrategy="drop"
)

model = als.fit(dfs_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction with the model
# MAGIC 
# MAGIC The trained model can be used to predict ratings with a given test data.

# COMMAND ----------

dfs_pred = model.transform(dfs_test).drop(COL_RATING)

# COMMAND ----------

evaluations = SparkRatingEvaluation(
    dfs_test, 
    dfs_pred,
    col_user=COL_USER,
    col_item=COL_ITEM,
    col_rating=COL_RATING,
    col_prediction=COL_PREDICTION
)

print(
    "RMSE score = {}".format(evaluations.rmse()),
    "MAE score = {}".format(evaluations.mae()),
    "R2 score = {}".format(evaluations.rsquared()),
    "Explained variance score = {}".format(evaluations.exp_var()),
    sep="\n"
)

# COMMAND ----------

# MAGIC %md
# MAGIC - RMSE : 예측값과 실제값을 빼서 제곱시킨 값들을 다 더해서 n으로 나눠줌, 그리고 루트를 씌움, 크기 의존적 에러를 가지고 있음 (예측 대상의 크기에 따른 scale 문제)
# MAGIC - MAE : 모델의 예측값과 실제값의 차이를 모두 더한다는 개념, 절대값을 취하기 때문에 가장 직관적으로 알 수 있는 지표, MSE 보다 특이치에 robust, 절대값을 취하기 때문에 모델이 underperformance 인지 overperformance 인지 알 수 없음
# MAGIC - R2 : 결정계수(決定係數, 영어: coefficient of determination)는 추정한 선형 모형이 주어진 자료에 적합한 정도를 재는 척도이다. 반응 변수의 변동량 중에서 적용한 모형으로 설명가능한 부분의 비율을 가리킴
# MAGIC - Explained variance score : 통계에서 설명된 변형은 주어진 데이터 세트의 변형 (분산)을 수학 모델이 차지하는 비율을 측정. 종종 편차는 분산으로 계량화됨. 총 변동의 보완적인 부분을 unexplained or residual variation
# MAGIC - 종종 순위 측정 기준은 데이터 과학자들에게도 중요. 일반적으로 순위 메트릭은 항목 목록을 추천하는 시나리오에 적용. 추천 항목은 사용자가 평가 한 항목과 달라야 함

# COMMAND ----------

# Get the cross join of all user-item pairs and score them.
users = dfs_train.select('UserId').distinct()
items = dfs_train.select('MovieId').distinct()
user_item = users.crossJoin(items)
dfs_pred = model.transform(user_item)

# COMMAND ----------

# Remove seen items.
dfs_pred_exclude_train = dfs_pred.alias("pred").join(
    dfs_train.alias("train"),
    (dfs_pred['UserId'] == dfs_train['UserId']) & (dfs_pred['MovieId'] == dfs_train['MovieId']),
    how='outer'
)

dfs_pred_final = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train.Rating"].isNull()) \
    .select('pred.' + 'UserId', 'pred.' + 'MovieId', 'pred.' + "prediction")

dfs_pred_final.show()

# COMMAND ----------

dfs_pred_final.count()

# COMMAND ----------

dfs_pred_exclude_train.count()

# COMMAND ----------

evaluations = SparkRankingEvaluation(
    dfs_test, 
    dfs_pred_final,
    col_user=COL_USER,
    col_item=COL_ITEM,
    col_rating=COL_RATING,
    col_prediction=COL_PREDICTION,
    k=10
)

print(
    "Precision@k = {}".format(evaluations.precision_at_k()),
    "Recall@k = {}".format(evaluations.recall_at_k()),
    "NDCG@k = {}".format(evaluations.ndcg_at_k()),
    "Mean average precision = {}".format(evaluations.map_at_k()),
    sep="\n"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fine tune the model
# MAGIC 
# MAGIC - 스파크 ALS 모델의 예측 성능은 종종 파라미터에 의해 영향을 받음
# MAGIC 
# MAGIC |Parameter|Description|Default value|Notes|
# MAGIC |-------------|-----------------|------------------|-----------------|
# MAGIC |`rank`|Number of latent factors|10|The larger the more intrinsic factors considered in the factorization modeling.|
# MAGIC |`regParam`|Regularization parameter|1.0|The value needs to be selected empirically to avoid overfitting.|
# MAGIC |`maxIters`|Maximum number of iterations|10|The more iterations the better the model converges to the optimal point.|
# MAGIC 
# MAGIC - 기본 매개 변수 값을 사용하여 모델 작성을 시작한 다음 최적의 매개 변수 조합을 찾기 위해 범위에서 매개 변수를 스윕하는 것은 항상 좋은 방법 - 다음 매개 변수 세트는 비교 연구 목적으로 ALS 모델을 학습하는 데 사용됩니다.

# COMMAND ----------

# param_dict = {
#     "rank": [15, 20, 25],
#     "regParam": [0.001, 0.1, 1.0]
# }

# COMMAND ----------

# from reco_utils.tuning.parameter_sweep import generate_param_grid
# param_grid = generate_param_grid(param_dict)

# rmse_score = []

# for g in param_grid:
#     als = ALS(        
#         userCol=COL_USER, 
#         itemCol=COL_ITEM, 
#         ratingCol=COL_RATING, 
#         maxIter=3, 
#         coldStartStrategy="drop",
#         **g
#     )
    
#     model = als.fit(dfs_train)
    
#     dfs_pred = model.transform(dfs_test).drop(COL_RATING)
    
#     evaluations = SparkRatingEvaluation(
#         dfs_test, 
#         dfs_pred,
#         col_user=COL_USER,
#         col_item=COL_ITEM,
#         col_rating=COL_RATING,
#         col_prediction=COL_PREDICTION
#     )

#     rmse_score.append(evaluations.rmse())

# rmse_score = [float('%.4f' % x) for x in rmse_score]
# rmse_score_array = np.reshape(rmse_score, (len(param_dict["rank"]), len(param_dict["regParam"])))

# COMMAND ----------

# rmse_df = pd.DataFrame(data=rmse_score_array, index=pd.Index(param_dict["rank"], name="rank"), 
#                        columns=pd.Index(param_dict["regParam"], name="reg. parameter"))

# COMMAND ----------

# MAGIC %md
# MAGIC - 계산된 RMSE 점수를 시각화하여 모델 성능이 다른 매개 변수에 의해 어떻게 영향을 받는지 비교 연구 할 수 있음
# MAGIC - 시각화에서 과적합으로 인해 RMSE가 먼저 감소한 다음 순위가 증가함에 따라 증가한다는 것을 알 수 있음. 순위가 20이고 정규화 매개 변수가 0.1이면 모델이 가장 낮은 RMSE 점수를 얻음

# COMMAND ----------

# fig, ax = plt.subplots()
# sns.heatmap(rmse_df, cbar=False, annot=True, fmt=".4g")
# display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top K recommendation
# MAGIC 
# MAGIC #### Top k for all users (items)

# COMMAND ----------

dfs_rec = model.recommendForAllUsers(10)

# COMMAND ----------

dfs_rec.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Top k for a selected set of users (items)

# COMMAND ----------

users = dfs_train.select(als.getUserCol()).distinct().limit(3)

dfs_rec_subset = model.recommendForUserSubset(users, 10)

# COMMAND ----------

dfs_rec_subset.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC - 일반적으로 모든 사용자에 대한 최상위 - k 권장 사항을 계산하는 것이 ALS 기반 권장 시스템의 전체 파이프 라인 (모델 교육 및 채점)의 병목
# MAGIC - 모든 사용자 - 항목 쌍으로부터 최상위 k를 얻으려면 일반적으로 계산 비용이 많이 드는 크로스 조인이 필요
# MAGIC - 사용자 - 항목 쌍의 내부 제품은 특정 최신의 컴퓨팅 가속 라이브러리 (예 : BLAS)에서 사용할 수있는 매트릭스 블록 곱셈 기능을 활용하는 대신 개별적으로 계산
