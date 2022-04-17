# Databricks notebook source
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

# COMMAND ----------

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
    
# # 하루 데이터가 10 미만인 NFT 이름을 리스트에 담고 중복 제거
# listx =[]
# for i ,v in df_dict.items():
#     for j in df_dict.get(i).groupby('timestamp').count()['name']:
#         if j < 10:
#             listx.append(i)
# se = set(listx)
# tmp = df_dict.copy()
# # 하루 10 미만 NFT를 제거
# for i in se:
#     del(tmp[i])

# # 거래 날짜 30 미만인 NFT 이름을 리스트에 담고 중복 제거
# listx2=[]
# for i,v in tmp.items():
#     if len(v.groupby('timestamp').count() ) <30 : # 총 날짜 거래 수가 30이 안되면
#         listx2.append(i)
# se2 = set(listx2)
# se2,len(se2)
    
# # 거래 날짜 30 미만 NFT 제거
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
# cc  
# cc.to_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT')
# a= cc.groupby('name')['transaction_value'].count()
# b= cc.groupby('name')['transaction_value'].sum()
# c = cc.groupby('name')['transaction_value'].mean()
# d = cc.groupby('name')['transaction_value'].median()
# NFT2 = pd.concat([a,b,c,d],axis=1)
# NFT2.columns=['tx_count','tx_sum','tx_mean','tx_median']
# NFT2

# COMMAND ----------

# 다시 transfer와 nfts의 merge data에서 제거해야할 NFT리스트 값만 빼고 남은 NFT 목록

# COMMAND ----------

NFT = pd.read_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT',index_col=0)
NFT

# COMMAND ----------

NFT2= pd.read_csv('/dbfs/FileStore/nft/kaggle/EDA_NFT/NFT2',index_col=0)
NFT2

# COMMAND ----------

NFT2['tx_mean']

# COMMAND ----------

# MAGIC %md
# MAGIC # 전체 데이터 분포

# COMMAND ----------

NFT2.sort_values('tx_count',ascending=False).index

# COMMAND ----------

NFT2.sort_values('tx_sum',ascending=False)[:30]

# COMMAND ----------

fig = px.pie(NFT2, values='tx_count',names=NFT2.index , title='tx_count' ,width=1000,height=1000)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=1.2
))
fig.show()
fig = px.pie(NFT2, values='tx_mean',names=NFT2.index , title='tx_mean' ,width=1000,height=1000)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=1.2
))
fig.show()
fig = px.pie(NFT2.sort_values('tx_sum',ascending=False)[:30], values='tx_sum',names=NFT2.sort_values('tx_sum',ascending=False)[:30].index , title='tx_sum' ,width=1000,height=1000)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=1.2
))
fig.show()
fig = px.pie(NFT2, values='tx_median',names=NFT2.index , title='tx_median' ,width=1000,height=1000)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=1.2
))
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 특정 NFT 분포
# MAGIC - 크립토 키티

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

layout = go.Layout(title='Count')
fig = go.Figure(layout=layout)
for k, v in  df_dict.items() :
    fig.add_trace(go.Scatter(x = v.index, y=v['count'],name=k)   )
fig.show()   
layout = go.Layout(title='Sum')
fig = go.Figure(layout=layout)

for k, v in  df_dict.items() :
    fig.add_trace(go.Scatter(x = v.index, y=v['sum'],name=k)   )
fig.show()   
layout = go.Layout(title='Mean')
fig = go.Figure(layout=layout)
for k, v in  df_dict.items() :
    fig.add_trace(go.Scatter(x = v.index, y=v['mean'],name=k)   )
fig.show()   
layout = go.Layout(title='Midean')
fig = go.Figure(layout=layout)
for k, v in  df_dict.items() :
    fig.add_trace(go.Scatter(x = v.index, y=v['median'],name=k)   )
fig.show()   

# COMMAND ----------

# MAGIC %md
# MAGIC # TX 정제 끝! DTW시작
# MAGIC 
# MAGIC - CryptoKitties
# MAGIC - Art Blocks

# COMMAND ----------

layout = go.Layout(title='Mean')
fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x = df_dict.get('Art Blocks').index, y=df_dict.get('Art Blocks')['mean'] ,name='Art Blocks' )   )

fig.show()  

# COMMAND ----------

layout = go.Layout(title='Mean')
fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x = df_dict.get('CryptoKitties').index, y=df_dict.get('CryptoKitties')['mean'],name='CryptoKitties' )  )
fig.add_trace(go.Scatter(x = df_dict.get('Art Blocks').index, y=df_dict.get('Art Blocks')['mean'] ,name='Art Blocks' )   )

fig.show()  

# COMMAND ----------

min(df_dict.get('CryptoKitties')['mean']) ,max(df_dict.get('CryptoKitties')['mean'])
min(df_dict.get('Art Blocks')['mean']) ,max(df_dict.get('Art Blocks')['mean'])

# COMMAND ----------

# MAGIC %md
# MAGIC # 유클라디안 거리 구하는 함수

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
# MAGIC # 누적 비용 매트릭스

# COMMAND ----------

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

X = df_dict.get('CryptoKitties')['mean']
Y= df_dict.get('Art Blocks')['mean']

XV=df_dict.get('CryptoKitties')['mean'].values
YV=df_dict.get('Art Blocks')['mean'].values


# COMMAND ----------

fig, ax = plt.subplots(figsize=(30, 10))

# Remove the border and axes ticks
# fig.patch.set_visible(False)
# ax.axis('off')

XX = [(X.index[i], X[i]) for i in np.arange(0, len(X))]
YY = [(Y.index[j], Y[j]) for j in np.arange(0, len(Y))]


for i, j in zip(XX, YY):
    ax.plot(   [ i[0], j[0]], [i[1], j[1]], '--k', linewidth=1)
# ax.plot(   )
ax.plot(X, '-bo', label='CryptoKitties', linewidth=3, markersize=4, markerfacecolor='lightcoral', markeredgecolor='lightcoral')
plt.legend( fontsize=(40) )

ax.plot(Y, '-ro', label='Art Blocks', linewidth=3, markersize=4, markerfacecolor='skyblue', markeredgecolor='skyblue')
plt.legend(fontsize=(40) )


ax.set_title("Euclidean Distance", fontsize=28, fontweight="bold")

# COMMAND ----------

dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
print("DTW distance: ", dtw_distance)
print("Warp path: ", warp_path)

# COMMAND ----------

cost_matrix = compute_accumulated_cost_matrix(X, Y)

# COMMAND ----------

cost_matrix = compute_accumulated_cost_matrix(X, Y)
fig, ax = plt.subplots(figsize=(120, 120))
ax = sns.heatmap(cost_matrix, annot=True, square=True, linewidths=10, cmap="YlGnBu", ax=ax, annot_kws={'size':30})
ax.invert_yaxis()

# Get the warp path in x and y directions
path_x = [p[0] for p in warp_path]
path_y = [p[1] for p in warp_path]

# Align the path from the center of each cell
path_xx = [x+0.5 for x in path_x]
path_yy = [y+0.5 for y in path_y]

ax.plot(path_xx, path_yy, color='blue', linewidth=30, alpha=0.2)

# COMMAND ----------


fig, ax = plt.subplots(figsize=(30, 10), )

# Remove the border and axes ticks
# ax.se
ax.axis('off')
# plt.setp(ax.get_xticklabels(), fo")ntsize=12, fontweight="bold", 
#          horizontalalignment="left
for [map_x, map_y] in warp_path:
    ax.plot( [map_x, map_y], [X[map_x], Y[map_y]], '--k', linewidth=0.7)
    
ax.plot(XV, '-bo', label='CryptoKitties', linewidth=3, markersize=4, markerfacecolor='lightcoral', markeredgecolor='lightcoral')
plt.legend(fontsize=(40) )
ax.plot(YV, '-ro', label='Art Blocks', linewidth=3, markersize=4, markerfacecolor='skyblue', markeredgecolor='skyblue')
plt.legend(fontsize=(40) )
ax.set_title(" DTW Distance", fontsize=28, fontweight="bold")


# COMMAND ----------

X = df_dict.get('CryptoKitties')['mean']
XV=df_dict.get('CryptoKitties')['mean'].values
XX = [(X.index[i], X[i]) for i in np.arange(0, len(X))]

DTWLIST=[]
for i,v in df_dict.items():
#     print(i,v)
    Y = df_dict.get(i)['mean']
    
   
    # FastDTW 사용한 경로 및 DTW 거리 계산
    dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
   
    DTWLIST.append( (i, dtw_distance )  )


DTWLIST    

# COMMAND ----------

T = pd.DataFrame(DTWLIST)
T.columns=['NFT','DTW_Distance']
T = T.sort_values('DTW_Distance')
T                                           


# COMMAND ----------

tmp = T.copy()
tmp = tmp[:11]
tmp['Top10'] = ['', 1,2,3,4,5,6,7,8,9,10  ]
tmp.set_index('Top10')


# COMMAND ----------

X = df_dict.get('Art Blocks')['mean']
XV=df_dict.get('Art Blocks')['mean'].values
XX = [(X.index[i], X[i]) for i in np.arange(0, len(X))]

DTWLIST=[]
for i,v in df_dict.items():
#     print(i,v)
    Y = df_dict.get(i)['mean']
    
   
    # FastDTW 사용한 경로 및 DTW 거리 계산
    dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
   
    DTWLIST.append( (i, dtw_distance )  )


DTWLIST    

# COMMAND ----------

T = pd.DataFrame(DTWLIST)
T.columns=['NFT','DTW_Distance']
T = T.sort_values('DTW_Distance')
T                                           


# COMMAND ----------

tmp = T.copy()
tmp = tmp[:11]
tmp.set_index()

# COMMAND ----------

# 데이터가 너무 깨져서 시각화 하기힘듦 => 설명은 8월~ 기준 으로 시각화하고 DATA 값은 전체 날짜로 진행하도록하자.

# COMMAND ----------

# MAGIC %md
# MAGIC 10개만

# COMMAND ----------

X = df_dict.get('Art Blocks')['mean']
XV=df_dict.get('Art Blocks')['mean'].values
XX = [(X.index[i], X[i]) for i in np.arange(0, len(X))]

DTWLIST=[]
for i,v in df_dict.items():
#     print(i,v)
    Y = df_dict.get(i)['mean']
    
   
    # FastDTW 사용한 경로 및 DTW 거리 계산
    dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
   
    DTWLIST.append( (i, dtw_distance )  )


DTWLIST    

# COMMAND ----------

T = pd.DataFrame(DTWLIST)
T.columns=['NFT','DTW_Distance']
T = T.sort_values('DTW_Distance')
T[:11]                                           


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # PPT 제출용 날짜 범위 수정 데이터 시각화

# COMMAND ----------

X = df_dict.get('CryptoKitties')['mean'].head(30)
Y= df_dict.get('Art Blocks')['mean'].head(30)

XV=df_dict.get('CryptoKitties')['mean'].head(30).values
YV=df_dict.get('Art Blocks')['mean'].head(30).values


# COMMAND ----------

fig, ax = plt.subplots(figsize=(30, 10))

# Remove the border and axes ticks
# fig.patch.set_visible(False)
# ax.axis('off')

XX = [(X.index[i], X[i]) for i in np.arange(0, len(X))]
YY = [(Y.index[j], Y[j]) for j in np.arange(0, len(Y))]


for i, j in zip(XX, YY):
    ax.plot(   [ i[0], j[0]], [i[1], j[1]], '--k', linewidth=1)
# ax.plot(   )
ax.plot(X, '-bo', label='CryptoKitties', linewidth=3, markersize=4, markerfacecolor='lightcoral', markeredgecolor='lightcoral')
plt.legend( fontsize=(40) )

ax.plot(Y, '-ro', label='Art Blocks', linewidth=3, markersize=4, markerfacecolor='skyblue', markeredgecolor='skyblue')
plt.legend(fontsize=(40) )


ax.set_title("Euclidean Distance", fontsize=28, fontweight="bold")

# COMMAND ----------

dtw_distance, warp_path = fastdtw(X, Y, dist=euclidean)
print("DTW distance: ", dtw_distance)
print("Warp path: ", warp_path)

# COMMAND ----------

cost_matrix = compute_accumulated_cost_matrix(X, Y)
fig, ax = plt.subplots(figsize=(120, 120))
ax = sns.heatmap(cost_matrix, annot=True, square=True, linewidths=10, cmap="YlGnBu", ax=ax, annot_kws={'size':30})
ax.invert_yaxis()

# Get the warp path in x and y directions
path_x = [p[0] for p in warp_path]
path_y = [p[1] for p in warp_path]

# Align the path from the center of each cell
path_xx = [x+0.5 for x in path_x]
path_yy = [y+0.5 for y in path_y]

ax.plot(path_xx, path_yy, color='blue', linewidth=30, alpha=0.2)

# COMMAND ----------


fig, ax = plt.subplots(figsize=(30, 10), )

# Remove the border and axes ticks
# ax.se
ax.axis('off')
# plt.setp(ax.get_xticklabels(), fo")ntsize=12, fontweight="bold", 
#          horizontalalignment="left
for [map_x, map_y] in warp_path:
    ax.plot( [map_x, map_y], [X[map_x], Y[map_y]], '--k', linewidth=0.7)
    
ax.plot(XV, '-bo', label='CryptoKitties', linewidth=3, markersize=4, markerfacecolor='lightcoral', markeredgecolor='lightcoral')
plt.legend(fontsize=(40) )
ax.plot(YV, '-ro', label='Art Blocks', linewidth=3, markersize=4, markerfacecolor='skyblue', markeredgecolor='skyblue')
plt.legend(fontsize=(40) )
ax.set_title(" Fast Time Distance", fontsize=28, fontweight="bold")

