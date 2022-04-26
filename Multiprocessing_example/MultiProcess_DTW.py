# Databricks notebook source
# MAGIC %md
# MAGIC # 필요패키지 임포트

# COMMAND ----------

import math 
import time 
import datetime

import numpy as np
import pandas as pd 

# DTW
import dtw 
# Computation packages
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


# 멀티프로세스
from multiprocessing import Process
import multiprocessing
import parmap
import requests

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #유클라디안 거리 구하는 함수

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC # 필요 CSV파일 불러오기

# COMMAND ----------

NFT = pd.read_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT',index_col=0)
NFT

# COMMAND ----------

# NFT2= pd.read_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT2',index_col=0)
# NFT2

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
# MAGIC # 하루조건 1개 일수 30개

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
    
# 하루 데이터가 1 미만인 NFT 이름을 리스트에 담고 중복 제거
listx =[]
for i ,v in df_dict.items():
    for j in df_dict.get(i).groupby('timestamp').count()['name']:
        if j < 1:
            listx.append(i)
se = set(listx)
tmp = df_dict.copy()
# 하루 1 미만 NFT를 제거
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
NFT1_30=  tmp.copy()
NFT1_30.index= df_dict.keys()
NFT1_30.columns = df_dict.keys()
DTW_time =  time.time() - start
NFT1_30

# COMMAND ----------

# MAGIC %md
# MAGIC ↑ 넘 오래걸려서 멈춤

# COMMAND ----------

# MAGIC %md
# MAGIC # 하루조건 2개 일수 30개

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
    
# 하루 데이터가2미만인 NFT 이름을 리스트에 담고 중복 제거
listx =[]
for i ,v in df_dict.items():
    for j in df_dict.get(i).groupby('timestamp').count()['name']:
        if j <2:
            listx.append(i)
se = set(listx)
tmp = df_dict.copy()
# 하루2 미만 NFT를 제거
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
NFT2_30=  tmp.copy()
NFT2_30.index= df_dict.keys()
NFT2_30.columns = df_dict.keys()
DTW_time =  time.time() - start
NFT2_30

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
# MAGIC # 하루조건 5개 일수 5개

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
    if len(v.groupby('timestamp').count() ) <5 : # 총 날짜 거래 수가 30이 안되면
        listx2.append(i)
se2 = set(listx2)
se2,len(se2)
    
# 거래 날짜 5 미만 NFT 제거
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
NFT5_5=  tmp.copy()
NFT5_5.index= df_dict.keys()
NFT5_5.columns = df_dict.keys()
DTW_time =  time.time() - start
NFT5_5

# COMMAND ----------

# MAGIC %md
# MAGIC # 하루조건 1개 일수 5개

# COMMAND ----------

# start = time.time()
# x = pd.read_csv('/dbfs/FileStore/nft/kaggle/transfers.csv')
# y = pd.read_csv('/dbfs/FileStore/nft/kaggle/nfts.csv')
# xx = pd.merge(x,y,left_on='nft_address',right_on='address')
# xx = xx.loc[:,['name','transaction_value','timestamp']]
# xx['timestamp']= pd.to_datetime(xx.timestamp, unit='s', errors='coerce')
# xx['timestamp'] = xx['timestamp'].dt.strftime("%Y-%m-%d")

# # 일단 전부 DF로 나눴음
# df_dict = {}
# for i in list(set(xx['name'].values)):
#     df_dict[i] = pd.DataFrame( xx[xx['name']== i] )
    
# # 하루 데이터가 5 미만인 NFT 이름을 리스트에 담고 중복 제거
# listx =[]
# for i ,v in df_dict.items():
#     for j in df_dict.get(i).groupby('timestamp').count()['name']:
#         if j < 1:
#             listx.append(i)
# se = set(listx)
# tmp = df_dict.copy()
# # 하루 1 미만 NFT를 제거
# for i in se:
#     del(tmp[i])

# # 거래 날짜 30 미만인 NFT 이름을 리스트에 담고 중복 제거
# listx2=[]
# for i,v in tmp.items():
#     if len(v.groupby('timestamp').count() ) <5 : # 총 날짜 거래 수가 30이 안되면
#         listx2.append(i)
# se2 = set(listx2)
# se2,len(se2)
    
# # 거래 날짜 5 미만 NFT 제거
# for i in se2:
#     del(tmp[i])    
    
# # se, se2 
# asd =[]
# # name이 se, se2안에 있는 데이터라면 O 아니면 X  
# for i in range(len(xx)):
#     if xx['name'][i] in se or xx['name'][i] in se2:
#         asd.append('O')
#     else:
#         asd.append('X')
# xx['check']= asd
# cc = xx[xx['check']=='X'].iloc[:,:-1]

# NFT = cc.copy()
# eda_time = time.time() - start

# df_dict = {}
# for i in list(set(NFT['name'].values)):
#     df_dict[i] = pd.DataFrame( NFT[NFT['name']== i] )
# GR_dict={}
# for k,v in df_dict.items():
#     v['timestamp'] = pd.to_datetime(v['timestamp'])
#     a = v.groupby('timestamp')['transaction_value'].count()
#     b = v.groupby('timestamp')['transaction_value'].sum()
#     c = v.groupby('timestamp')['transaction_value'].mean()
#     d = v.groupby('timestamp')['transaction_value'].median()
#     df = pd.concat( [a,b,c,d],axis=1  )
#     df.columns =  ['count','sum','mean', 'median' ]
#     GR_dict[k] = df
# GR_dict = sorted(GR_dict.items(), key=lambda x: len(x[1]), reverse=True)
# tmp = {}
# for i in GR_dict:  
# #     tmp[]= print(i[0]) # 인덱스
# #     tmp[]=print(i[1]) # 값
#     tmp[ i[0] ] = i[1]
    
# df_dict = tmp.copy()

# COMMAND ----------

# start = time.time()
# lista = [i for i in range(len(df_dict.keys()))]
# tmp = pd.DataFrame(lista)
# for m,v in df_dict.items():
#     X = df_dict.get(m)['mean']
#     DTWLIST=[]
#     for i,v in df_dict.items():
# #     print(i,v)
#         Y = df_dict.get(i)['mean']
#         # FastDTW 사용한 경로 및 DTW 거리 계산
#         dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
#         DTWLIST.append( dtw_distance   )
#     T = pd.DataFrame(DTWLIST)
#     tmp = pd.concat([tmp,T],axis=1)

# tmp = tmp.iloc[:,1:]
# NFT1_5=  tmp.copy()
# NFT1_5.index= df_dict.keys()
# NFT1_5.columns = df_dict.keys()
# DTW_time =  time.time() - start
# NFT1_5

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ↑ 너무 오래걸려서 멈춤

# COMMAND ----------

# MAGIC %md
# MAGIC # 하루조건 2개 일수 5개

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
    
# 하루 데이터가 2 미만인 NFT 이름을 리스트에 담고 중복 제거
listx =[]
for i ,v in df_dict.items():
    for j in df_dict.get(i).groupby('timestamp').count()['name']:
        if j < 2:
            listx.append(i)
se = set(listx)
tmp = df_dict.copy()
# 하루 5 미만 NFT를 제거
for i in se:
    del(tmp[i])

# 거래 날짜 5 미만인 NFT 이름을 리스트에 담고 중복 제거
listx2=[]
for i,v in tmp.items():
    if len(v.groupby('timestamp').count() ) <5 : # 총 날짜 거래 수가 30이 안되면
        listx2.append(i)
se2 = set(listx2)
se2,len(se2)
    
# 거래 날짜 5 미만 NFT 제거
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
NFT2_5=  tmp.copy()
NFT2_5.index= df_dict.keys()
NFT2_5.columns = df_dict.keys()
DTW_time =  time.time() - start
NFT2_5

# COMMAND ----------

# MAGIC %md
# MAGIC # 멀티 프로세스 적용?
# MAGIC ### - parmap

# COMMAND ----------

# 멀티프로세싱 cpu 카운트
num_cores = multiprocessing.cpu_count() 
num_cores

# COMMAND ----------

key_list = list(df_dict.keys())
len(key_list)

# COMMAND ----------

# MAGIC %md
# MAGIC # 여러 조건 시간 정리

# COMMAND ----------

tmp1=  pd.DataFrame([27.09,4.27,115] ,columns=['NFT5_30'] )
tmp2=  pd.DataFrame([27.32,7.08,166] ,columns=['NFT5_20'] )
tmp3 = pd.DataFrame([27.22,8.36,193] ,columns=['NFT5_15'] )
tmp4=  pd.DataFrame([27.04,9.96,228] ,columns=['NFT5_10'] )
tmp5=  pd.DataFrame([26.48,30.78,450] ,columns=['NFT2_5'] )
tmp6=  pd.DataFrame([26.48,7.09,450] ,columns=['멀티프로세스사용 NFT2_5'] )
NFT_DTW = pd.concat([tmp1,tmp2,tmp3,tmp4,tmp5,tmp6],axis=1)
NFT_DTW.index=['전처리 시간(m)','DTW 계산 시간(m)','NFT개수']
NFT_DTW

# COMMAND ----------

# MAGIC %md
# MAGIC # 멀티프로세스 사용 비교

# COMMAND ----------

# MAGIC %md
# MAGIC ## 하루조건 2개 일수 5개 기준

# COMMAND ----------

# MAGIC %md
# MAGIC ## 프로세스 1개 사용

# COMMAND ----------

start = time.time()
def multiporcess_DTW(key):
    tmplist=[]
    for i ,v in df_dict.items():
#         print(i)
        X = df_dict.get(key)['mean']
        Y = df_dict.get(i)    
        Z = fastdtw(X,Y, dist=euclidean)
        tmplist.append(Z[0])
    return tmplist

dtw_list= []
# for i in range(len(mydict)):
result = parmap.map(multiporcess_DTW,key_list ,pm_pbar=True,pm_processes=1)
dtw_list.append(result)
dtw_list
parmap_fastdtw = pd.DataFrame(dtw_list[0], index=key_list , columns=key_list)
DTW_process1=  time.time() - start
parmap_fastdtw

# COMMAND ----------

# MAGIC %md
# MAGIC ## 프로세스 3개 사용

# COMMAND ----------

start = time.time()
def multiporcess_DTW(key):
    tmplist=[]
    for i ,v in df_dict.items():
#         print(i)
        X = df_dict.get(key)['mean']
        Y = df_dict.get(i)    
        Z = fastdtw(X,Y, dist=euclidean)
        tmplist.append(Z[0])
    return tmplist

dtw_list= []
# for i in range(len(mydict)):
result = parmap.map(multiporcess_DTW,key_list ,pm_pbar=True,pm_processes=3)
dtw_list.append(result)
dtw_list
parmap_fastdtw = pd.DataFrame(dtw_list[0], index=key_list , columns=key_list)
DTW_process3=  time.time() - start
parmap_fastdtw

# COMMAND ----------

# MAGIC %md
# MAGIC ## 프로세스 5개 사용

# COMMAND ----------

start = time.time()
def multiporcess_DTW(key):
    tmplist=[]
    for i ,v in df_dict.items():
#         print(i)
        X = df_dict.get(key)['mean']
        Y = df_dict.get(i)    
        Z = fastdtw(X,Y, dist=euclidean)
        tmplist.append(Z[0])
    return tmplist

dtw_list= []
# for i in range(len(mydict)):
result = parmap.map(multiporcess_DTW,key_list ,pm_pbar=True,pm_processes=5)
dtw_list.append(result)
dtw_list
parmap_fastdtw = pd.DataFrame(dtw_list[0], index=key_list , columns=key_list)
DTW_process5 =  time.time() - start
parmap_fastdtw

# COMMAND ----------

# MAGIC %md
# MAGIC ## 프로세스 8개 사용

# COMMAND ----------

start = time.time()
def multiporcess_DTW(key):
    tmplist=[]
    for i ,v in df_dict.items():
#         print(i)
        X = df_dict.get(key)['mean']
        Y = df_dict.get(i)    
        Z = fastdtw(X,Y, dist=euclidean)
        tmplist.append(Z[0])
    return tmplist

dtw_list= []
# for i in range(len(mydict)):
result = parmap.map(multiporcess_DTW,key_list ,pm_pbar=True,pm_processes=8)
dtw_list.append(result)
dtw_list
parmap_fastdtw = pd.DataFrame(dtw_list[0], index=key_list , columns=key_list)
DTW_process8 =  time.time() - start
parmap_fastdtw


# COMMAND ----------

# MAGIC %md
# MAGIC # 450개의NFT DF 생성해서 시간 비교 
# MAGIC ### - 하루조건 2 / 일수조건 5

# COMMAND ----------

pd.DataFrame([DTW_process1,DTW_process3,DTW_process5,DTW_process8],index =['단일 프로세스','멀티프로세스3','멀티프로세스5','멀티프로세스8'],columns=['소요시간(s)'])

# COMMAND ----------

# MAGIC %md
# MAGIC # 효과 → 단일 프로세스기준 프로세스 8 개를 사용하여 약 4배이상 시간 단축
