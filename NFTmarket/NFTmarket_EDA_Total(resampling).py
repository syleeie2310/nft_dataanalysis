# Databricks notebook source
import numpy as np
import pandas as pddat

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. 데이터 로드 및 통합
# MAGIC - https://nonfungible.com/market/history
# MAGIC - 차트별 ALL-Time Daily CSV파일 다운로드
# MAGIC - 2017.06.22 ~ 현재

# COMMAND ----------

# 70개 항목 파일명 리스트로 가져오기
import os
file_list = os.listdir('/dbfs/FileStore/nft/nft_market_220221')
len(file_list), print(file_list)

# COMMAND ----------

# 파일명 추출(확장자 제거)
file_name = []
for file in file_list:
    if file.count(".") == 1: 
        name = file.split('.')[0]
        file_name.append(name)

file_name

# COMMAND ----------

# csv파일 모두 판다스로 읽기
# 자동 변수 선언 + 변수 목록 생성
val_list = []
for file in file_name :
    globals()['data_{}'.format(file)] = 0
    val_list.append(f'data_{file}')
val_list

# COMMAND ----------

# 데이터셋들을 개별 데이터프레임변수로 생성하기& 칼럼명변경(카테고리 붙이기)
data_list = val_list.copy()
for i in range(len(file_name)):
    data_list[i] = pd.read_csv(f'/dbfs/FileStore/nft/nft_market_220221/{file_name[i]}.csv', index_col = "Date", parse_dates=True, thousands=',')
    data_list[i].columns = [file_name[i]] # 칼럼명 변경
    print(data_list[i])

# COMMAND ----------

total = data_list[0]
for i in range(1, len(data_list)):
    total = pd.merge(total, data_list[i], left_index=True, right_index=True, how='left')
total

# COMMAND ----------

total.info()

# COMMAND ----------

total.describe()

# COMMAND ----------

# 지수(e)출력 문제 해결
pd.set_option('float_format', '{:.1f}'.format) # 숫자 많아서 보기 어렵다. 소수점은 1자리만 보자
total.info()
total.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. 데이터 클렌징

# COMMAND ----------

# 결측치 체크
import missingno as msno
msno.matrix(total)

# COMMAND ----------

pd.isna(total).sum()

# COMMAND ----------

# 결측치 채우기, 앞쪽 결측치는 0으로 채우자
total.fillna(0, inplace=True)
msno.matrix(total)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 정제된 통합 데이터 파일 저장

# COMMAND ----------

# total.to_csv("/dbfs/FileStore/nft/nft_market_220221/total_cleaned.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. 데이터 프로파일링
# MAGIC ## 요약
# MAGIC - 일단위는 파악하기 어렵다. 월단위도 큰 차이 없음, 분기별로 리샘플링하자
# MAGIC - 기간이 다른(결측치) 카테고리 유의해야함
# MAGIC - 전반적으로 값 범위가 매우 다름, 스케일링 필요
# MAGIC   - 스케일링 전에 각 변수별 시각화를 해보자
# MAGIC   - 상관관계도 보자
# MAGIC - 누적 형식이다 보니 우상향 왜도가 많음, log변환해보자
# MAGIC 
# MAGIC ## 형식
# MAGIC - 데이터 사이즈 : 1704 * 70
# MAGIC - 데이터 종류 : 피처 전부 [수치형-연속형]
# MAGIC 
# MAGIC ## 카테고리 (종합 포함 총 7개)
# MAGIC - ALL, Art, Collectible, DeFi, Game, Utility, Metaverse 
# MAGIC 
# MAGIC ## 피처 (10개)
# MAGIC - active_market_wallets : 자산을 매매한 고유 지갑 수
# MAGIC - number_of_sales : 총 판매 수
# MAGIC - sales_usd : 판매완료 건에 대한 총 USD
# MAGIC - average_usd : 판매 평균 USD
# MAGIC - primary_sales : 1차 또는 주요시장의 판매 총 수
# MAGIC   - _**1차 시장 : 창작자의 웹사이트에서 이루어지는 거래**_
# MAGIC - primary_sales_usd : 완료된 주요시장 판매에 지불된 총 USD
# MAGIC - secondary_sales : 2차 시장(사용자간) 판매 총 수
# MAGIC   - _**2차 시장 : opensea와 같은 마켓플레이스에서 이루어지는 거래**_
# MAGIC - secondary_sales_usd : 완료된 2차 시장 판매에 지불된 총 USD
# MAGIC - unique_buyers : 자산을 구매한 고유 지갑 수
# MAGIC - unique_sellers : 자산을 판매한 고유 지갑 수

# COMMAND ----------

# MAGIC %md
# MAGIC ## 기초 통계 체크

# COMMAND ----------

total.info()

# COMMAND ----------

total.describe(include='all')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw데이터 시각화
# MAGIC - 히스토그램(분포 체크) : 값의 갭차이가 크고 데이터 구간이 너무 길어서 분포차이가 눈에 안띔, >>>>   "로그변환 + 기간 리샘플링" 필요
# MAGIC - 라인 차트(추세 체크) : 갭차이가 커서 추세 체크 어렵고, 변동이벤트가 많아 해석이 어렵다. >>>> "로그변환 + 기간 리샘플링 + 그루핑" 필요 
# MAGIC - 박스 플롯(분위수 체크) : 전체 기간 분위수는 무의미함, >>>> "로그변환 + 기간 리샘플링" 필요

# COMMAND ----------

# MAGIC %md
# MAGIC ### 히스토그램 : 분포 체크

# COMMAND ----------

# 값의 갭차이가 크므로 로그변환해서 봐야함, 그런데 여전히 데이터 구간이 너무 길어서 차이가 눈에 안띈다, 월별/분기별로 리샘플링해보자
display(total)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 라인 플롯 : 추세 체크

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import patches
%matplotlib inline
plt.style.use("ggplot")

# COMMAND ----------

import matplotlib.dates as mdates

def line_plot(data):
    plt.figure(figsize=(25,80))
    plt.suptitle("Trend check", fontsize=40)

    cols = data.columns
    for i in range(len(cols)):
        plt.subplot(14,5,i+1)
        plt.title(cols[i], fontsize=20)
        plt.plot(data[cols[i]], color='b', alpha=0.7)
        # 추세선 그리기
        x = mdates.date2num(data[cols[i]].index)
        y = data[cols[i]]
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# COMMAND ----------

# 갭이 크고 피처가 너무 많다., 정규화(로그변환) 후 카테고리별로 시각화 하자
line_plot(total)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 박스 플롯 : 분위수 체크
# MAGIC - pass
# MAGIC - 전체 기간 분위수는 무의미함, 구간별 체크 필요

# COMMAND ----------

import matplotlib.dates as mdates

def box_plot(data):
    plt.figure(figsize=(25,80))
    plt.suptitle("Trend check", fontsize=40)

    cols = data.columns
    for i in range(len(cols)):
        plt.subplot(14,5,i+1)
        plt.title(cols[i], fontsize=20)
        plt.boxplot(data[cols[i]])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# COMMAND ----------

# 전체범위 분위수는 무의미하다.
box_plot(total)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. 데이터 리샘플링
# MAGIC - 일 -> 분기(월단위는 큰 차이 없음)

# COMMAND ----------

total['all_active_market_wallets']['2017-7':'2017-12'].mean()

# COMMAND ----------

# 분기단위 리샘플링, 딱 조으다
total2Q_avg = total.resample(rule='2Q').mean()
total2Q_avg

# COMMAND ----------

# MAGIC %md
# MAGIC ## 리샘플링 데이터 프로파일링

# COMMAND ----------

total2Q_avg.describe(include='all')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 리샘플링 데이터 시각화

# COMMAND ----------

# MAGIC %md
# MAGIC ### 히스토그램 : 분포 비교

# COMMAND ----------

# 데이터프로파일링에서 log 변환해서 볼 수 있음
display(total2Q_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 라인 플롯 : 추세 비교

# COMMAND ----------

# chart에서 로그변환 할 수 있음
temp = total2Q_avg.copy()
temp['index'] = total2Q_avg.index
display(temp)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 캔들 스틱 : 분위수+추이 비교
# MAGIC - 주식에서 많이 사용하는 캔들스틱 차트를 써보자

# COMMAND ----------

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 그래프 생성 함수
def candlestick(data, mediandata, ma2data, ma4data, col_data, i, j):
    num = 0
    fig = go.Figure(data=[go.Candlestick(x= col_data.index,
                                                    open=col_data['open'],
                                                    high=col_data['high'],
                                                    low=col_data['low'],
                                                    close=col_data['close']),
                                      go.Scatter(x=col_data.index, y=ma2data.iloc[:,j], line=dict(color='orange', width=1), name ='Median(MA2)'),
                                      go.Scatter(x=col_data.index, y=ma4data.iloc[:,j], line=dict(color='green', width=1), name ='Median(MA4)'),
                                      go.Scatter(x=col_data.index, y=mediandata.iloc[:,j], line=dict(color='blue', width=1), name ='Median(rule)')
                                     ])
    num += 1
    fig.layout = dict(title= f'[{num}], {data.columns[i][0]}', xaxis_title = 'Quater', yaxis_title= 'Value')

    #         주석 = [ dict ( 
    #         x ='2016-12-09' ,  y = 0.05 ,  xref = 'x' ,  yref = 'paper' , 
    #         showarrow = False ,  xanchor = 'left' ,  text = '증가 기간 시작' )] 
    fig.show()

# COMMAND ----------

def candle(data, rule, option, start):
        
    # 캔들스틱용 데이터셋 생성
    total_ohlc = data[start:].resample(rule).ohlc()
    total_median = data[start:].resample(rule).median()
    total_median_MA2 = total_median.rolling(2).mean() 
    total_median_MA4 = total_median.rolling(4).mean()

    # 조건별 그래프 생성 함수 호출
    if option == 'all':
        num = 0
        for i in range(0, len(total_ohlc.columns), 4):
            j = i//4 # location에서는 연산안됨
            col_data = total_ohlc[total_ohlc.columns[i][0]] # 멀티 칼럼이라 그래프 만들기 복잡해진다, 피처(상위칼럼)별로 데이터를 따로 빼주자
            candlestick(total_ohlc, total_median, total_median_MA2, total_median_MA4, col_data, i, j)
    elif option == 'test':
        i = 0
        j = i//4 # location에서는 연산안됨
        col_data = total_ohlc[total_ohlc.columns[i][0]]
        candlestick(total_ohlc, total_median, total_median_MA2, total_median_MA4, col_data, i, j)
    else :
        print("data와 rule, option(all or test), start(유효한 date index를 입력)를 입력하세요")
        

# COMMAND ----------

# 반기로 바꿔도 2020년도까지는 값을 보기 어렵다.  로그변환할 수는 없고.. 축기준에 맞도록 범위 조정필요
candle(total, '2Q', 'test', 0)  # 전체일자를 보려면 start인자에 0 입력

# COMMAND ----------

candle(total, 'M', 'test', '2021-10') 

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. 리샘플링 데이터 정규화

# COMMAND ----------

# MAGIC %md
# MAGIC ## 로그 변환

# COMMAND ----------

totalQ_avg_log = totalQ_avg.copy()
totalQ_avg_log = np.log1p(totalQ_avg_log)
totalQ_avg_log.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 스케일링
# MAGIC - 모형을 유지할 수 있고, 정규분포가 아니므로 min-max scaler가 적합
# MAGIC https://ysyblog.tistory.com/217

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
totalQ_avg_log_scaled = totalQ_avg_log.copy()
totalQ_avg_log_scaled.iloc[:,:] = minmax_scaler.fit_transform(totalQ_avg_log_scaled)
totalQ_avg_log_scaled.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. 인터렉티브 시각화(비교)
# MAGIC - 분기 기준 고정
# MAGIC - 추세 비교 그래프
# MAGIC - 비중 비교 그래프

# COMMAND ----------

# MAGIC %md
# MAGIC ## Category : 라인차트 추세 비교

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 카테고리 (칼럼) 분류기

# COMMAND ----------

# 카테고리 분류기

def category_classifier(data, category):
    col_list = []
    for i in range(len(data.columns)):
        if category == 'total':
            return category
        elif data.columns[i].split('_')[0] == category:
            col_list.append(data.columns[i])
        else :
            pass
    return col_list

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 카테고리별 라인차트 생성기

# COMMAND ----------

import plotly.express as px

def lineC(data, category):
    # 입력 유효성 체크
    if category in ['total', 'all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility']:
        # 카테고리 분류기 호출
        col_list = category_classifier(temp, category)
        # 라인차트 정의
        for j in range(len(col_list)):
            fig = px.line(data[col_list])     
        fig.layout = dict(title= f'{category}카테고리별 피처 *추세* 비교')
        fig.show()    
        
    else : 
        print("카테고리를 입력하세요, ['total', 'all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 카테고리별 누적영역차트 생성기

# COMMAND ----------

# 누적 영역 차트 함수 생성
import plotly.express as px
import plotly.graph_objects as go

def stackareaC(data, category):
    # 입력 유효성 체크
    if category in ['total', 'all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility']:
        # 카테고리 분류기 호출
        col_list = category_classifier(data, category)
        # 누적영역차트 정의
        fig = go.Figure()
        for j in range(len(col_list)):
            fig.add_trace(go.Scatter(
                x = data[col_list[j]].index,
                y = data[col_list[j]].values,
                hoverinfo='x+y',
                mode='lines',
                line=dict(width = 0.5),
                stackgroup='one',
                groupnorm='percent',
                name = col_list[j]
            ))
        fig.layout = dict(title= f'{category}카테고리별 피처 *비중* 비교')
        fig.update_layout(showlegend=True, yaxis=dict(range=[1, 100], ticksuffix='%'))
        fig.show()
    else : 
        print("카테고리를 입력하세요, ['total', 'all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### all 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'all')

# COMMAND ----------

stackareaC(totalQ_avg, 'all')

# COMMAND ----------

# MAGIC %md
# MAGIC ### collectible 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'collectible')

# COMMAND ----------

stackareaC(totalQ_avg, 'collectible')

# COMMAND ----------

# MAGIC %md
# MAGIC ### art 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'art')

# COMMAND ----------

stackareaC(totalQ_avg, 'art')

# COMMAND ----------

# MAGIC %md
# MAGIC ### metaverse 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'metaverse')

# COMMAND ----------

stackareaC(totalQ_avg, 'metaverse')

# COMMAND ----------

# MAGIC %md
# MAGIC ### game 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'game')

# COMMAND ----------

stackareaC(totalQ_avg, 'game')

# COMMAND ----------

# MAGIC %md
# MAGIC ### utility 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'utility')

# COMMAND ----------

stackareaC(totalQ_avg, 'utility')

# COMMAND ----------

# MAGIC %md
# MAGIC ### DeFi 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'defi')

# COMMAND ----------

stackareaC(totalQ_avg, 'defi')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature : 라인차트 추세비교
# MAGIC - active_market_wallets, average_usd, number_of_sales, primary_sales, primary_sales_usd, sales_usd, secondary_sales, secondary_sales_usd, unique_buyers, unique_sellers

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 피처 칼럼 분류기

# COMMAND ----------

# 카테고리 분류기
def feature_classifier(data, feature):
    col_list = []
    for i in range(len(data.columns)):
        split_col = data.columns[i].split('_', maxsplit=1)[1]
        if split_col == feature:       
            col_list.append(data.columns[i])
        elif split_col == 'all_sales_usd' and feature == 'sales_usd' : #콜렉터블만 sales_usd앞에 all이붙어서 따로 처리해줌
            col_list.append('collectible_all_sales_usd')
        else :
            pass
    return col_list

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 피처별 라인차트 생성기

# COMMAND ----------

def lineF(data, feature):
    # 피처 입력값 유효성 체크
    if feature in ['active_market_wallets', 'average_usd', 'number_of_sales', 'primary_sales', 'primary_sales_usd', 'sales_usd', 'secondary_sales', 'secondary_sales_usd', 'unique_buyers', 'unique_sellers']:
        # 피처 분류기 호출
        col_list = feature_classifier(data, feature)
        # 라인차트 정의
        for j in range(len(col_list)):
            fig = px.line(data[col_list])     
        fig.layout = dict(title= f'{feature} 피처별 *추세* 비교')
        fig.show()    
        
    else : 
        print("피처칼럼을 입력하세요, ['active_market_wallets', 'average_usd', 'number_of_sales', 'primary_sales', 'primary_sales_usd', 'sales_usd', 'secondary_sales', 'secondary_sales_usd', 'unique_buyers', 'unique_sellers']")

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 피처별 누적영역차트 생성기

# COMMAND ----------

# 누적 영역 차트 함수 생성
import plotly.express as px
import plotly.graph_objects as go

def stackareaF(data, feature):
    # 입력 유효성 체크
    if feature in ['active_market_wallets', 'average_usd', 'number_of_sales', 'primary_sales', 'primary_sales_usd', 'sales_usd', 'secondary_sales', 'secondary_sales_usd', 'unique_buyers', 'unique_sellers']:
        # 피처 분류기 호출
        col_list = feature_classifier(data, feature)
        # 누적영역차트 정의
        fig = go.Figure()
        for j in range(len(col_list)):
            fig.add_trace(go.Scatter(
                x = data[col_list[j]].index,
                y = data[col_list[j]].values,
                hoverinfo='x+y',
                mode='lines',
                line=dict(width = 0.5),
                stackgroup='one',
                groupnorm='percent',
                name = col_list[j]
            ))
        fig.layout = dict(title= f'{feature}피처별 *비중* 비교')
        fig.update_layout(showlegend=True, yaxis=dict(range=[1, 100], ticksuffix='%'))
        fig.show()
    else : 
        print("피처칼럼을 입력하세요, ['active_market_wallets', 'average_usd', 'number_of_sales', 'primary_sales', 'primary_sales_usd', 'sales_usd', 'secondary_sales', 'secondary_sales_usd', 'unique_buyers', 'unique_sellers']")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of Sales 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'number_of_sales')

# COMMAND ----------

stackareaF(totalQ_avg, 'number_of_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sales USD 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'sales_usd')

# COMMAND ----------

stackareaF(totalQ_avg, 'sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Average USD 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'average_usd')

# COMMAND ----------

stackareaF(totalQ_avg, 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Active Market Wallets 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'active_market_wallets')

# COMMAND ----------

stackareaF(totalQ_avg, 'active_market_wallets')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Primary Sales 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'primary_sales')

# COMMAND ----------

stackareaF(totalQ_avg, 'primary_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Secondary Sales 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'secondary_sales')

# COMMAND ----------

stackareaF(totalQ_avg, 'secondary_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Primary Sales USD 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'primary_sales_usd')

# COMMAND ----------

stackareaF(totalQ_avg, 'primary_sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Secondary Sales USD 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'secondary_sales_usd')

# COMMAND ----------

stackareaF(totalQ_avg, 'secondary_sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unique Buyers 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'unique_buyers')

# COMMAND ----------

stackareaF(totalQ_avg, 'unique_buyers')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unique Sellers 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'unique_sellers')

# COMMAND ----------

stackareaF(totalQ_avg, 'unique_sellers')
