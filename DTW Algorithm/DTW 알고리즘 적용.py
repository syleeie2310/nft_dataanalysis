# Databricks notebook source
import math 
import time 
import datetime

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from pandas import datetime
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import patches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from pandas import datetime
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import patches
import matplotlib.dates as mdates
from plotly.subplots import make_subplots
# 관련 라이브러리 임포트 
import matplotlib.font_manager as fm

#  한글글꼴로 변경
# plt.rcParams['font.family'] = '한글글꼴명'
plt.rcParams['font.size'] = 11.0
# plt.rcParams['font.family'] = 'batang'
plt.rcParams['font.family'] = 'Malgun Gothic'

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
# matplotlib.rcParams['axes.unicode_minus'] = False

# DTW
import dtw 
# Computation packages
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

#유클라디안 거리 구하는 함수
def compute_euclidean_distance_matrix(x, y) -> np.array:
    """Calculate distance matrix
    This method calcualtes the pairwise Euclidean distance between two sequences.
    The sequences can have different lengths.
    """
    dist = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            dist[i,j] = (x[j]-y[i])**2
    return dist
# 누적 비용 매트릭스
def compute_accumulated_cost_matrix(x, y) :
    
    """
    Compute accumulated cost matrix for warp path using Euclidean distance
    """
    distances = compute_euclidean_distance_matrix(x, y)

    # Initialization
    cost = np.zeros((len(y), len(x)))
    cost[0,0] = distances[0,0]
    
    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i-1, 0]  
        
    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j-1]  

    # Accumulated warp path cost
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(
                cost[i-1, j],    # insertion
                cost[i, j-1],    # deletion
                cost[i-1, j-1]   # match
            ) + distances[i, j] 
            
    return cost

# COMMAND ----------

# 일의 조건 10개 평균 가격

# COMMAND ----------

NFT = pd.read_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT',index_col=0)
NFT

# COMMAND ----------

NFT2= pd.read_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT2',index_col=0)
NFT2

# COMMAND ----------

df_dict = {}
for i in list(set(NFT['name'].values)):
    df_dict[i] = pd.DataFrame( NFT[NFT['name']== i] )
GR_dict={}
for k,v in df_dict.items():
    v['timestamp'] = pd.to_datetime(v['timestamp'])
    a = v.groupby('timestamp')['transaction_value'].count()
    b = v.groupby('timestamp')['transaction_value'].sum()
    c = v.groupby('timestamp')['transaction_value'].mean()
    d = v.groupby('timestamp')['transaction_value'].median()
    df = pd.concat( [a,b,c,d],axis=1  )
    df.columns =  ['count','sum','mean', 'median' ]
    GR_dict[k] = df
GR_dict = sorted(GR_dict.items(), key=lambda x: len(x[1]), reverse=True)
tmp = {}
for i in GR_dict:  
#     tmp[]= print(i[0]) # 인덱스
#     tmp[]=print(i[1]) # 값
    tmp[ i[0] ] = i[1]
    
df_dict = tmp.copy()

# COMMAND ----------

lista = [i for i in range(74)]
tmp = pd.DataFrame(lista)
tmp

# COMMAND ----------

for m,v in df_dict.items():
    X = df_dict.get(m)['mean']
    DTWLIST=[]
    for i,v in df_dict.items():
#     print(i,v)
        Y = df_dict.get(i)['mean']
        # FastDTW 사용한 경로 및 DTW 거리 계산
        dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
        DTWLIST.append( dtw_distance   )
    T = pd.DataFrame(DTWLIST)
    tmp = pd.concat([tmp,T],axis=1)

tmp = tmp.iloc[:,1:]
NFT10 =  tmp.copy()
NFT10.index= df_dict.keys()
NFT10.columns = df_dict.keys()
NFT10

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 일거래 9미만 NFT

# COMMAND ----------

x = pd.read_csv('/dbfs/FileStore/nft/kaggle/transfers.csv')
y = pd.read_csv('/dbfs/FileStore/nft/kaggle/nfts.csv')
xx = pd.merge(x,y,left_on='nft_address',right_on='address')
xx = xx.loc[:,['name','transaction_value','timestamp']]
xx['timestamp']= pd.to_datetime(xx.timestamp, unit='s', errors='coerce')
xx['timestamp'] = xx['timestamp'].dt.strftime("%Y-%m-%d")

# 일단 전부 DF로 나눴음
df_dict = {}
for i in list(set(xx['name'].values)):
    df_dict[i] = pd.DataFrame( xx[xx['name']== i] )
    
# 하루 데이터가 9 미만인 NFT 이름을 리스트에 담고 중복 제거
listx =[]
for i ,v in df_dict.items():
    for j in df_dict.get(i).groupby('timestamp').count()['name']:
        if j < 9:
            listx.append(i)
se = set(listx)
tmp = df_dict.copy()
# 하루 10 미만 NFT를 제거
for i in se:
    del(tmp[i])

# 거래 날짜 30 미만인 NFT 이름을 리스트에 담고 중복 제거
listx2=[]
for i,v in tmp.items():
    if len(v.groupby('timestamp').count() ) <30 : # 총 날짜 거래 수가 30이 안되면
        listx2.append(i)
se2 = set(listx2)
se2,len(se2)
    
# 거래 날짜 30 미만 NFT 제거
for i in se2:
    del(tmp[i])    
    
# se, se2 
asd =[]
# name이 se, se2안에 있는 데이터라면 O 아니면 X  
for i in range(len(xx)):
    if xx['name'][i] in se or xx['name'][i] in se2:
        asd.append('O')
    else:
        asd.append('X')
xx['check']= asd
cc = xx[xx['check']=='X'].iloc[:,:-1]
cc  
# cc.to_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT')
# a= cc.groupby('name')['transaction_value'].count()
# b= cc.groupby('name')['transaction_value'].sum()
# c = cc.groupby('name')['transaction_value'].mean()
# d = cc.groupby('name')['transaction_value'].median()
# NFT2 = pd.concat([a,b,c,d],axis=1)
# NFT2.columns=['tx_count','tx_sum','tx_mean','tx_median']
# NFT2

# COMMAND ----------

NFT = cc.copy()
NFT

# COMMAND ----------

df_dict = {}
for i in list(set(NFT['name'].values)):
    df_dict[i] = pd.DataFrame( NFT[NFT['name']== i] )
GR_dict={}
for k,v in df_dict.items():
    v['timestamp'] = pd.to_datetime(v['timestamp'])
    a = v.groupby('timestamp')['transaction_value'].count()
    b = v.groupby('timestamp')['transaction_value'].sum()
    c = v.groupby('timestamp')['transaction_value'].mean()
    d = v.groupby('timestamp')['transaction_value'].median()
    df = pd.concat( [a,b,c,d],axis=1  )
    df.columns =  ['count','sum','mean', 'median' ]
    GR_dict[k] = df
GR_dict = sorted(GR_dict.items(), key=lambda x: len(x[1]), reverse=True)
tmp = {}
for i in GR_dict:  
#     tmp[]= print(i[0]) # 인덱스
#     tmp[]=print(i[1]) # 값
    tmp[ i[0] ] = i[1]
    
df_dict = tmp.copy()

# COMMAND ----------

len(df_dict.keys())

# COMMAND ----------

lista = [i for i in range(len(df_dict.keys()))]
tmp = pd.DataFrame(lista)
tmp

# COMMAND ----------

for m,v in df_dict.items():
    X = df_dict.get(m)['mean']
    DTWLIST=[]
    for i,v in df_dict.items():
#     print(i,v)
        Y = df_dict.get(i)['mean']
        # FastDTW 사용한 경로 및 DTW 거리 계산
        dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
        DTWLIST.append( dtw_distance   )
    T = pd.DataFrame(DTWLIST)
    tmp = pd.concat([tmp,T],axis=1)

tmp = tmp.iloc[:,1:]
NFT9=  tmp.copy()
NFT9.index= df_dict.keys()
NFT9.columns = df_dict.keys()
NFT9

# COMMAND ----------

# MAGIC %md
# MAGIC # 일거래 5미만 NFT

# COMMAND ----------

x = pd.read_csv('/dbfs/FileStore/nft/kaggle/transfers.csv')
y = pd.read_csv('/dbfs/FileStore/nft/kaggle/nfts.csv')
xx = pd.merge(x,y,left_on='nft_address',right_on='address')
xx = xx.loc[:,['name','transaction_value','timestamp']]
xx['timestamp']= pd.to_datetime(xx.timestamp, unit='s', errors='coerce')
xx['timestamp'] = xx['timestamp'].dt.strftime("%Y-%m-%d")

# 일단 전부 DF로 나눴음
df_dict = {}
for i in list(set(xx['name'].values)):
    df_dict[i] = pd.DataFrame( xx[xx['name']== i] )
    
# 하루 데이터가 5 미만인 NFT 이름을 리스트에 담고 중복 제거
listx =[]
for i ,v in df_dict.items():
    for j in df_dict.get(i).groupby('timestamp').count()['name']:
        if j < 5:
            listx.append(i)
se = set(listx)
tmp = df_dict.copy()
# 하루 10 미만 NFT를 제거
for i in se:
    del(tmp[i])

# 거래 날짜 30 미만인 NFT 이름을 리스트에 담고 중복 제거
listx2=[]
for i,v in tmp.items():
    if len(v.groupby('timestamp').count() ) <30 : # 총 날짜 거래 수가 30이 안되면
        listx2.append(i)
se2 = set(listx2)
se2,len(se2)
    
# 거래 날짜 30 미만 NFT 제거
for i in se2:
    del(tmp[i])    
    
# se, se2 
asd =[]
# name이 se, se2안에 있는 데이터라면 O 아니면 X  
for i in range(len(xx)):
    if xx['name'][i] in se or xx['name'][i] in se2:
        asd.append('O')
    else:
        asd.append('X')
xx['check']= asd
cc = xx[xx['check']=='X'].iloc[:,:-1]
cc  
# cc.to_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT')
# a= cc.groupby('name')['transaction_value'].count()
# b= cc.groupby('name')['transaction_value'].sum()
# c = cc.groupby('name')['transaction_value'].mean()
# d = cc.groupby('name')['transaction_value'].median()
# NFT2 = pd.concat([a,b,c,d],axis=1)
# NFT2.columns=['tx_count','tx_sum','tx_mean','tx_median']
# NFT2

# COMMAND ----------

NFT = cc.copy()
NFT

# COMMAND ----------

df_dict = {}
for i in list(set(NFT['name'].values)):
    df_dict[i] = pd.DataFrame( NFT[NFT['name']== i] )
GR_dict={}
for k,v in df_dict.items():
    v['timestamp'] = pd.to_datetime(v['timestamp'])
    a = v.groupby('timestamp')['transaction_value'].count()
    b = v.groupby('timestamp')['transaction_value'].sum()
    c = v.groupby('timestamp')['transaction_value'].mean()
    d = v.groupby('timestamp')['transaction_value'].median()
    df = pd.concat( [a,b,c,d],axis=1  )
    df.columns =  ['count','sum','mean', 'median' ]
    GR_dict[k] = df
GR_dict = sorted(GR_dict.items(), key=lambda x: len(x[1]), reverse=True)
tmp = {}
for i in GR_dict:  
#     tmp[]= print(i[0]) # 인덱스
#     tmp[]=print(i[1]) # 값
    tmp[ i[0] ] = i[1]
    
df_dict = tmp.copy()

# COMMAND ----------

lista = [i for i in range(len(df_dict.keys()))]
tmp = pd.DataFrame(lista)
tmp

# COMMAND ----------

for m,v in df_dict.items():
    X = df_dict.get(m)['mean']
    DTWLIST=[]
    for i,v in df_dict.items():
#     print(i,v)
        Y = df_dict.get(i)['mean']
        # FastDTW 사용한 경로 및 DTW 거리 계산
        dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
        DTWLIST.append( dtw_distance   )
    T = pd.DataFrame(DTWLIST)
    tmp = pd.concat([tmp,T],axis=1)

tmp = tmp.iloc[:,1:]
NFT5=  tmp.copy()
NFT5.index= df_dict.keys()
NFT5.columns = df_dict.keys()
NFT5

# COMMAND ----------

# MAGIC %md 
# MAGIC # 결과
# MAGIC 
# MAGIC parallel X
# MAGIC 10개미만 -> 74개 pair fastdtw / 소요시간 1.95 min   
# MAGIC 9개미만 -> 85개 pair fastdtw / 소요시간 2.45 min           26.42m  
# MAGIC   
# MAGIC 5개미만=>115개  pair fastdtw /  소요시간 4.30 min        정제 26.68m  

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 하루조건 TEST   
# MAGIC 5개 미만으로 고정 
# MAGIC 
# MAGIC 일수 TEST   
# MAGIC 거래 30 이상   -> 20개  , 15개 , 10개    pair 시간 기록    
# MAGIC 
# MAGIC 
# MAGIC ㄴㄴㄴㄴ> 멀티프로세싱 사용해서 시간 단축      정리좀 잘하자
# MAGIC 
# MAGIC ㄴ> 마무리하고 포트폴리오 

# COMMAND ----------

# MAGIC %md
# MAGIC # 하루조건 5개 일수 30개

# COMMAND ----------

# start = time.time()

# for i in range(1000):
#     i=i

# time.time() - start

# COMMAND ----------

start = time.time()
x = pd.read_csv('/dbfs/FileStore/nft/kaggle/transfers.csv')
y = pd.read_csv('/dbfs/FileStore/nft/kaggle/nfts.csv')
xx = pd.merge(x,y,left_on='nft_address',right_on='address')
xx = xx.loc[:,['name','transaction_value','timestamp']]
xx['timestamp']= pd.to_datetime(xx.timestamp, unit='s', errors='coerce')
xx['timestamp'] = xx['timestamp'].dt.strftime("%Y-%m-%d")

# 일단 전부 DF로 나눴음
df_dict = {}
for i in list(set(xx['name'].values)):
    df_dict[i] = pd.DataFrame( xx[xx['name']== i] )
    
# 하루 데이터가 5 미만인 NFT 이름을 리스트에 담고 중복 제거
listx =[]
for i ,v in df_dict.items():
    for j in df_dict.get(i).groupby('timestamp').count()['name']:
        if j < 5:
            listx.append(i)
se = set(listx)
tmp = df_dict.copy()
# 하루 5 미만 NFT를 제거
for i in se:
    del(tmp[i])

# 거래 날짜 30 미만인 NFT 이름을 리스트에 담고 중복 제거
listx2=[]
for i,v in tmp.items():
    if len(v.groupby('timestamp').count() ) <30 : # 총 날짜 거래 수가 30이 안되면
        listx2.append(i)
se2 = set(listx2)
se2,len(se2)
    
# 거래 날짜 30 미만 NFT 제거
for i in se2:
    del(tmp[i])    
    
# se, se2 
asd =[]
# name이 se, se2안에 있는 데이터라면 O 아니면 X  
for i in range(len(xx)):
    if xx['name'][i] in se or xx['name'][i] in se2:
        asd.append('O')
    else:
        asd.append('X')
xx['check']= asd
cc = xx[xx['check']=='X'].iloc[:,:-1]
cc  
# cc.to_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT')
# a= cc.groupby('name')['transaction_value'].count()
# b= cc.groupby('name')['transaction_value'].sum()
# c = cc.groupby('name')['transaction_value'].mean()
# d = cc.groupby('name')['transaction_value'].median()
# NFT2 = pd.concat([a,b,c,d],axis=1)
# NFT2.columns=['tx_count','tx_sum','tx_mean','tx_median']
# NFT2
NFT = cc.copy()
eda_time = time.time() - start

df_dict = {}
for i in list(set(NFT['name'].values)):
    df_dict[i] = pd.DataFrame( NFT[NFT['name']== i] )
GR_dict={}
for k,v in df_dict.items():
    v['timestamp'] = pd.to_datetime(v['timestamp'])
    a = v.groupby('timestamp')['transaction_value'].count()
    b = v.groupby('timestamp')['transaction_value'].sum()
    c = v.groupby('timestamp')['transaction_value'].mean()
    d = v.groupby('timestamp')['transaction_value'].median()
    df = pd.concat( [a,b,c,d],axis=1  )
    df.columns =  ['count','sum','mean', 'median' ]
    GR_dict[k] = df
GR_dict = sorted(GR_dict.items(), key=lambda x: len(x[1]), reverse=True)
tmp = {}
for i in GR_dict:  
#     tmp[]= print(i[0]) # 인덱스
#     tmp[]=print(i[1]) # 값
    tmp[ i[0] ] = i[1]
    
df_dict = tmp.copy()

# COMMAND ----------

start = time.time()
lista = [i for i in range(len(df_dict.keys()))]
tmp = pd.DataFrame(lista)
for m,v in df_dict.items():
    X = df_dict.get(m)['mean']
    DTWLIST=[]
    for i,v in df_dict.items():
#     print(i,v)
        Y = df_dict.get(i)['mean']
        # FastDTW 사용한 경로 및 DTW 거리 계산
        dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
        DTWLIST.append( dtw_distance   )
    T = pd.DataFrame(DTWLIST)
    tmp = pd.concat([tmp,T],axis=1)

tmp = tmp.iloc[:,1:]
NFT5_30=  tmp.copy()
NFT5_30.index= df_dict.keys()
NFT5_30.columns = df_dict.keys()
DTW_time =  time.time() - start
NFT5_30

# COMMAND ----------

tmp1 = pd.DataFrame([27.09,4.27,115] ,columns=['NFT5_30'] )

# COMMAND ----------

# MAGIC %md
# MAGIC # 하루조건5개 일수 20개

# COMMAND ----------

start = time.time()
x = pd.read_csv('/dbfs/FileStore/nft/kaggle/transfers.csv')
y = pd.read_csv('/dbfs/FileStore/nft/kaggle/nfts.csv')
xx = pd.merge(x,y,left_on='nft_address',right_on='address')
xx = xx.loc[:,['name','transaction_value','timestamp']]
xx['timestamp']= pd.to_datetime(xx.timestamp, unit='s', errors='coerce')
xx['timestamp'] = xx['timestamp'].dt.strftime("%Y-%m-%d")

# 일단 전부 DF로 나눴음
df_dict = {}
for i in list(set(xx['name'].values)):
    df_dict[i] = pd.DataFrame( xx[xx['name']== i] )
    
# 하루 데이터가 5 미만인 NFT 이름을 리스트에 담고 중복 제거
listx =[]
for i ,v in df_dict.items():
    for j in df_dict.get(i).groupby('timestamp').count()['name']:
        if j < 5:
            listx.append(i)
se = set(listx)
tmp = df_dict.copy()
# 하루 5 미만 NFT를 제거
for i in se:
    del(tmp[i])

# 거래 날짜 20 미만인 NFT 이름을 리스트에 담고 중복 제거
listx2=[]
for i,v in tmp.items():
    if len(v.groupby('timestamp').count() ) <20 : # 총 날짜 거래 수가 20이 안되면
        listx2.append(i)
se2 = set(listx2)
se2,len(se2)
    
# 거래 날짜 20 미만 NFT 제거
for i in se2:
    del(tmp[i])    
    
# se, se2 
asd =[]
# name이 se, se2안에 있는 데이터라면 O 아니면 X  
for i in range(len(xx)):
    if xx['name'][i] in se or xx['name'][i] in se2:
        asd.append('O')
    else:
        asd.append('X')
xx['check']= asd
cc = xx[xx['check']=='X'].iloc[:,:-1]
cc  
# cc.to_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT')
# a= cc.groupby('name')['transaction_value'].count()
# b= cc.groupby('name')['transaction_value'].sum()
# c = cc.groupby('name')['transaction_value'].mean()
# d = cc.groupby('name')['transaction_value'].median()
# NFT2 = pd.concat([a,b,c,d],axis=1)
# NFT2.columns=['tx_count','tx_sum','tx_mean','tx_median']
# NFT2
NFT = cc.copy()
eda_time = time.time() - start

df_dict = {}
for i in list(set(NFT['name'].values)):
    df_dict[i] = pd.DataFrame( NFT[NFT['name']== i] )
GR_dict={}
for k,v in df_dict.items():
    v['timestamp'] = pd.to_datetime(v['timestamp'])
    a = v.groupby('timestamp')['transaction_value'].count()
    b = v.groupby('timestamp')['transaction_value'].sum()
    c = v.groupby('timestamp')['transaction_value'].mean()
    d = v.groupby('timestamp')['transaction_value'].median()
    df = pd.concat( [a,b,c,d],axis=1  )
    df.columns =  ['count','sum','mean', 'median' ]
    GR_dict[k] = df
GR_dict = sorted(GR_dict.items(), key=lambda x: len(x[1]), reverse=True)
tmp = {}
for i in GR_dict:  
#     tmp[]= print(i[0]) # 인덱스
#     tmp[]=print(i[1]) # 값
    tmp[ i[0] ] = i[1]
    
df_dict = tmp.copy()


# COMMAND ----------

start = time.time()
lista = [i for i in range(len(df_dict.keys()))]
tmp = pd.DataFrame(lista)
for m,v in df_dict.items():
    X = df_dict.get(m)['mean']
    DTWLIST=[]
    for i,v in df_dict.items():
#     print(i,v)
        Y = df_dict.get(i)['mean']
        # FastDTW 사용한 경로 및 DTW 거리 계산
        dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
        DTWLIST.append( dtw_distance   )
    T = pd.DataFrame(DTWLIST)
    tmp = pd.concat([tmp,T],axis=1)

tmp = tmp.iloc[:,1:]
NFT5_20=  tmp.copy()
NFT5_20.index= df_dict.keys()
NFT5_20.columns = df_dict.keys()
DTW_time =  time.time() - start
NFT5_20

# COMMAND ----------

tmp2 = pd.DataFrame([27.32,7.08,166] ,columns=['NFT5_20'] )

# COMMAND ----------

# MAGIC %md
# MAGIC # 하루조건 5개 일수 15개

# COMMAND ----------

start = time.time()
x = pd.read_csv('/dbfs/FileStore/nft/kaggle/transfers.csv')
y = pd.read_csv('/dbfs/FileStore/nft/kaggle/nfts.csv')
xx = pd.merge(x,y,left_on='nft_address',right_on='address')
xx = xx.loc[:,['name','transaction_value','timestamp']]
xx['timestamp']= pd.to_datetime(xx.timestamp, unit='s', errors='coerce')
xx['timestamp'] = xx['timestamp'].dt.strftime("%Y-%m-%d")

# 일단 전부 DF로 나눴음
df_dict = {}
for i in list(set(xx['name'].values)):
    df_dict[i] = pd.DataFrame( xx[xx['name']== i] )
    
# 하루 데이터가 5 미만인 NFT 이름을 리스트에 담고 중복 제거
listx =[]
for i ,v in df_dict.items():
    for j in df_dict.get(i).groupby('timestamp').count()['name']:
        if j < 5:
            listx.append(i)
se = set(listx)
tmp = df_dict.copy()
# 하루 5 미만 NFT를 제거
for i in se:
    del(tmp[i])

# 거래 날짜 20 미만인 NFT 이름을 리스트에 담고 중복 제거
listx2=[]
for i,v in tmp.items():
    if len(v.groupby('timestamp').count() ) <15 : # 총 날짜 거래 수가 15 안되면
        listx2.append(i)
se2 = set(listx2)
se2,len(se2)
    
# 거래 날짜 15 미만 NFT 제거
for i in se2:
    del(tmp[i])    
    
# se, se2 
asd =[]
# name이 se, se2안에 있는 데이터라면 O 아니면 X  
for i in range(len(xx)):
    if xx['name'][i] in se or xx['name'][i] in se2:
        asd.append('O')
    else:
        asd.append('X')
xx['check']= asd
cc = xx[xx['check']=='X'].iloc[:,:-1]
cc  
# cc.to_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT')
# a= cc.groupby('name')['transaction_value'].count()
# b= cc.groupby('name')['transaction_value'].sum()
# c = cc.groupby('name')['transaction_value'].mean()
# d = cc.groupby('name')['transaction_value'].median()
# NFT2 = pd.concat([a,b,c,d],axis=1)
# NFT2.columns=['tx_count','tx_sum','tx_mean','tx_median']
# NFT2
NFT = cc.copy()
eda_time = time.time() - start
df_dict = {}
for i in list(set(NFT['name'].values)):
    df_dict[i] = pd.DataFrame( NFT[NFT['name']== i] )
GR_dict={}
for k,v in df_dict.items():
    v['timestamp'] = pd.to_datetime(v['timestamp'])
    a = v.groupby('timestamp')['transaction_value'].count()
    b = v.groupby('timestamp')['transaction_value'].sum()
    c = v.groupby('timestamp')['transaction_value'].mean()
    d = v.groupby('timestamp')['transaction_value'].median()
    df = pd.concat( [a,b,c,d],axis=1  )
    df.columns =  ['count','sum','mean', 'median' ]
    GR_dict[k] = df
GR_dict = sorted(GR_dict.items(), key=lambda x: len(x[1]), reverse=True)
tmp = {}
for i in GR_dict:  
#     tmp[]= print(i[0]) # 인덱스
#     tmp[]=print(i[1]) # 값
    tmp[ i[0] ] = i[1]
    
df_dict = tmp.copy()


# COMMAND ----------

start = time.time()
lista = [i for i in range(len(df_dict.keys()))]
tmp = pd.DataFrame(lista)
for m,v in df_dict.items():
    X = df_dict.get(m)['mean']
    DTWLIST=[]
    for i,v in df_dict.items():
#     print(i,v)
        Y = df_dict.get(i)['mean']
        # FastDTW 사용한 경로 및 DTW 거리 계산
        dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
        DTWLIST.append( dtw_distance   )
    T = pd.DataFrame(DTWLIST)
    tmp = pd.concat([tmp,T],axis=1)

tmp = tmp.iloc[:,1:]
NFT5_15=  tmp.copy()
NFT5_15.index= df_dict.keys()
NFT5_15.columns = df_dict.keys()
DTW_time =  time.time() - start
NFT5_15

# COMMAND ----------

tmp3 = pd.DataFrame([27.22,8.36,193] ,columns=['NFT5_15'] )

# COMMAND ----------

# MAGIC %md
# MAGIC # 하루조건 5개 일수 10개

# COMMAND ----------

start = time.time()
x = pd.read_csv('/dbfs/FileStore/nft/kaggle/transfers.csv')
y = pd.read_csv('/dbfs/FileStore/nft/kaggle/nfts.csv')
xx = pd.merge(x,y,left_on='nft_address',right_on='address')
xx = xx.loc[:,['name','transaction_value','timestamp']]
xx['timestamp']= pd.to_datetime(xx.timestamp, unit='s', errors='coerce')
xx['timestamp'] = xx['timestamp'].dt.strftime("%Y-%m-%d")

# 일단 전부 DF로 나눴음
df_dict = {}
for i in list(set(xx['name'].values)):
    df_dict[i] = pd.DataFrame( xx[xx['name']== i] )
    
# 하루 데이터가 5 미만인 NFT 이름을 리스트에 담고 중복 제거
listx =[]
for i ,v in df_dict.items():
    for j in df_dict.get(i).groupby('timestamp').count()['name']:
        if j < 5:
            listx.append(i)
se = set(listx)
tmp = df_dict.copy()
# 하루 5 미만 NFT를 제거
for i in se:
    del(tmp[i])

# 거래 날짜 10 미만인 NFT 이름을 리스트에 담고 중복 제거
listx2=[]
for i,v in tmp.items():
    if len(v.groupby('timestamp').count() ) <10 : # 총 날짜 거래 수가 10 안되면
        listx2.append(i)
se2 = set(listx2)
se2,len(se2)
    
# 거래 날짜 10 미만 NFT 제거
for i in se2:
    del(tmp[i])    
    
# se, se2 
asd =[]
# name이 se, se2안에 있는 데이터라면 O 아니면 X  
for i in range(len(xx)):
    if xx['name'][i] in se or xx['name'][i] in se2:
        asd.append('O')
    else:
        asd.append('X')
xx['check']= asd
cc = xx[xx['check']=='X'].iloc[:,:-1]
cc  
# cc.to_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT')
# a= cc.groupby('name')['transaction_value'].count()
# b= cc.groupby('name')['transaction_value'].sum()
# c = cc.groupby('name')['transaction_value'].mean()
# d = cc.groupby('name')['transaction_value'].median()
# NFT2 = pd.concat([a,b,c,d],axis=1)
# NFT2.columns=['tx_count','tx_sum','tx_mean','tx_median']
# NFT2
NFT = cc.copy()
eda_time = time.time() - start
NFT
df_dict = {}
for i in list(set(NFT['name'].values)):
    df_dict[i] = pd.DataFrame( NFT[NFT['name']== i] )
GR_dict={}
for k,v in df_dict.items():
    v['timestamp'] = pd.to_datetime(v['timestamp'])
    a = v.groupby('timestamp')['transaction_value'].count()
    b = v.groupby('timestamp')['transaction_value'].sum()
    c = v.groupby('timestamp')['transaction_value'].mean()
    d = v.groupby('timestamp')['transaction_value'].median()
    df = pd.concat( [a,b,c,d],axis=1  )
    df.columns =  ['count','sum','mean', 'median' ]
    GR_dict[k] = df
GR_dict = sorted(GR_dict.items(), key=lambda x: len(x[1]), reverse=True)
tmp = {}
for i in GR_dict:  
#     tmp[]= print(i[0]) # 인덱스
#     tmp[]=print(i[1]) # 값
    tmp[ i[0] ] = i[1]
    
df_dict = tmp.copy()


# COMMAND ----------

start = time.time()
lista = [i for i in range(len(df_dict.keys()))]
tmp = pd.DataFrame(lista)
for m,v in df_dict.items():
    X = df_dict.get(m)['mean']
    DTWLIST=[]
    for i,v in df_dict.items():
#     print(i,v)
        Y = df_dict.get(i)['mean']
        # FastDTW 사용한 경로 및 DTW 거리 계산
        dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
        DTWLIST.append( dtw_distance   )
    T = pd.DataFrame(DTWLIST)
    tmp = pd.concat([tmp,T],axis=1)

tmp = tmp.iloc[:,1:]
NFT5_10=  tmp.copy()
NFT5_10.index= df_dict.keys()
NFT5_10.columns = df_dict.keys()
DTW_time =  time.time() - start
NFT5_10

# COMMAND ----------

tmp4 = pd.DataFrame([27.04,9.96,228] ,columns=['NFT5_10'] )

# COMMAND ----------

# MAGIC %md
# MAGIC # 시간 정리

# COMMAND ----------

NFT_DTW = pd.concat([tmp1,tmp2,tmp3,tmp4],axis=1)
NFT_DTW.index=['전처리 시간(m)','DTW 계산 시간(m)','NFT개수']
NFT_DTW

# COMMAND ----------

# MAGIC %md
# MAGIC # 멀티 프로세스 적용?

# COMMAND ----------

from multiprocessing import Process 
import multiprocessing
import parmap
import requests

# COMMAND ----------

# 멀티프로세싱 cpu 카운트
num_cores = multiprocessing.cpu_count() 
num_cores

# COMMAND ----------

# MAGIC %md
# MAGIC # 프로세스 1개

# COMMAND ----------

start_time = time.time()
result = parmap.map(test, ex, pm_pbar=True, pm_processes=1)   # 주소가 담긴 리스트를 매개변수로 넣자.
print((time.time()-start_time),'초')
result

# COMMAND ----------

# MAGIC %md
# MAGIC # 프로세스 4개

# COMMAND ----------

start_time = time.time()
result = parmap.map(test, ex, pm_pbar=True, pm_processes=4)   # 주소가 담긴 리스트를 매개변수로 넣자.
print((time.time()-start_time),'초')
result


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


