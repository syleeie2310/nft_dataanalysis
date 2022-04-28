# Databricks notebook source
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf
import matplotlib.pyplot as plt
import seaborn as sb
from warnings import filterwarnings
filterwarnings("ignore")
plt.style.use("ggplot")
pd.options.display.float_format = '{:.2f}'.format

# COMMAND ----------

# MAGIC %md
# MAGIC # 데이터 로드

# COMMAND ----------

data = pd.read_csv('/dbfs/FileStore/nft/nft_market_cleaned/total_220222_cleaned.csv', index_col = "Date", parse_dates=True, thousands=',')

# COMMAND ----------

data.info()

# COMMAND ----------

data.head()

# COMMAND ----------

data.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross Correlation(상호상관분석)
# MAGIC - 위키피디아: https://en.wikipedia.org/wiki/Cross-correlation
# MAGIC - 1d 배열 : statsmodelCCF, numpy.correlate, matplotlib.pyplot.xcorr(numpy.correlate 기반)
# MAGIC   - https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.ccf.html
# MAGIC   - https://numpy.org/doc/stable/reference/generated/numpy.correlate.html#numpy.correlate
# MAGIC   - numpy.correlateFFT를 사용하여 컨볼루션을 계산하지 않기 때문에 큰 배열(즉, n = 1e5)에서 느리게 수행될 수 있습니다. 이 경우 scipy.signal.correlate바람직할 수 있습니다.
# MAGIC - 2d 배열 : scipy.signal.correlate2d , scipy.stsci.convolve.correlate2d 
# MAGIC - 교차 및 시차상관계수는 t기의 특정(기준)변수 x의 값(𝒙𝒕)과 t+k기에 관찰된 y값(𝒚𝒕+𝒌) 간의 상관관계의 정도를 나타냄
# MAGIC - k=0인 경우 즉, 𝜸𝟎인 경우를 교차상관계수(cross correlation coefficient)라고 하고, k≠0인 경우 를 시차상관계수(leads and lags correlation라고도 함
# MAGIC - 교차상관계수 해석
# MAGIC   - 𝜸𝟎> 0 : 두 변수들이 서로 같은 방향으로 변화(pro-cyclical:경기순응)
# MAGIC   - 𝜸𝟎< 0 : 두 변수들이 서로 반대 방향으로 변화(counter-cyclical:경기역행)
# MAGIC   - 𝜸𝟎 = 0 : 두 변수들이 서로 경기중립적
# MAGIC - 시차상관계수 해석
# MAGIC   - 𝜸𝒌의 값이 최대가 되는 시차 k가 양(+)이면 해당변수 𝒚𝒕는 𝒙𝒕의 후행지표
# MAGIC   - 𝜸𝒌의 값이 최대가 되는 시차 k가 음(-)이면 해당변수 𝒚𝒕는 𝒙𝒕의 선행지표
# MAGIC   - 𝜸𝒌의 값이 최대가 되는 시차 k가 0이면 해당변수 𝒚𝒕는 𝒙𝒕와 동행지표

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. CC 라이브러리 스터디

# COMMAND ----------

# MAGIC %md
# MAGIC ### 예제1 : statsmodel CCF
# MAGIC - adjusted (=unbiased): 참이면 교차 상관의 분모는 nk이고 그렇지 않으면 n입니다.
# MAGIC   - 편향되지 않은 것이 참이면 자기공분산의 분모가 조정되지만 자기상관은 편향되지 않은 추정량이 아닙니다.
# MAGIC - fft : True이면 FFT 컨볼루션을 사용합니다. 이 방법은 긴 시계열에 대해 선호되어야 합니다.

# COMMAND ----------

#define data 
marketing = np.array([3, 4, 5, 5, 7, 9, 13, 15, 12, 10, 8, 8])
revenue = np.array([21, 19, 22, 24, 25, 29, 30, 34, 37, 40, 35, 30]) 

# COMMAND ----------

import statsmodels.api as sm

#calculate cross correlation
sm.tsa.stattools.ccf(marketing, revenue, adjusted=False)

# COMMAND ----------

#  시차0에서 교차상관은 0.771, 시차1에서 교차상관은 0.462, 시차2에서 교차상관은 0.194, 시차3에서 교차상관은 -0.061
#  특정 월에서 마케팅비용을 지출하면 다음 2개월동안의 수익증가를 예측할 수 있다.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 예제2 : numpy.correlate

# COMMAND ----------

import numpy

myArray = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
myArray = numpy.array(myArray)
result = numpy.correlate(myArray, myArray, mode = 'full')
print(result)
print(result.size)
result = result[result.size // 2 :] # 완전히 겹치는 지점인 중간 이후부터 봐야함
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ccf와 correlate 와의 차이
# MAGIC - https://stackoverflow.com/questions/24616671/numpy-and-statsmodels-give-different-values-when-calculating-correlations-how-t
# MAGIC - ccf는 np.correlate 베이스이지만, 통계적의미에서 상관관계를 위한 추가 작업을 수행함
# MAGIC - numpy가 표준편차의 곱으로 공분산을 정규화 하지 않음, 값이 너무 큼
# MAGIC - ccf는 합성곱전에 신호의 평균을 빼고 결과를 첫번째 신호의 길이로 나누어 통계에서와 같은 상관관계 정의에 도달함
# MAGIC - 통계 및 시계열 분석에서는 교차상관함수를 정규화하여 시간종속 상관계수를 얻는 것이 일반적이다.
# MAGIC - 자기 상관을 상관 관계로 해석하면 통계적 의존도 의 척도가 없는 측정값이 제공 되고 정규화는 추정된 자기 상관의 통계적 속성에 영향을 미치기 때문에 정규화가 중요합니다.
# MAGIC - <아래 소스코드 참고>
# MAGIC 
# MAGIC ```
# MAGIC def ccovf(x, y, unbiased=True, demean=True):
# MAGIC     n = len(x)
# MAGIC     if demean:
# MAGIC         xo = x - x.mean()
# MAGIC         yo = y - y.mean()
# MAGIC     else:
# MAGIC         xo = x
# MAGIC         yo = y
# MAGIC     if unbiased:
# MAGIC         xi = np.ones(n)
# MAGIC         d = np.correlate(xi, xi, 'full')
# MAGIC     else:
# MAGIC         d = n
# MAGIC     return (np.correlate(xo, yo, 'full') / d)[n - 1:]
# MAGIC 
# MAGIC def ccf(x, y, unbiased=True):
# MAGIC     cvf = ccovf(x, y, unbiased=unbiased, demean=True)
# MAGIC     return cvf / (np.std(x) * np.std(y))
# MAGIC ```

# COMMAND ----------

# 카테고리별 피처 분류기
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

avgusd = data[feature_classifier(data, 'average_usd')]
avgusd.head()

# COMMAND ----------

avgusd_game = avgusd['game_average_usd']
avgusd_game.head()

# COMMAND ----------

# raw
plt.plot(avgusd_game)

# COMMAND ----------

result = np.correlate(avgusd_game, avgusd_game, 'full')
print(result[result.size // 2 :])

# COMMAND ----------

# numpy.correlate
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import ccf

#Calculate correlation using numpy.correlate
def corr(x,y):
    result = numpy.correlate(x, y, mode='full')
    return result[result.size//2:]

#Using numpy i get this
plt.plot(corr(avgusd_game,avgusd_game))

# COMMAND ----------

# statsmodel.ccf
plt.plot(ccf(avgusd_game, avgusd_game, adjusted=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 교차상관계수 시각화
# MAGIC - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xcorr.html
# MAGIC   - 이건 어떻게 커스텀을 못하겠다..

# COMMAND ----------

# numply.correlate
result = numpy.correlate(avgusd_game, avgusd_game, mode='full')
npcorr = result[result.size//2:]

nlags = len(npcorr)
leng = len(avgusd_game)

# /* Compute the Significance level */
conf_level = 2 / np.sqrt(nlags)
print('conf_level= ', conf_level)

# /* Draw Plot */
plt.figure(figsize=(30,10), dpi=80)
plt.hlines(0, xmin=0, xmax=leng, color='gray')  # 0 axis
plt.hlines(conf_level, xmin=0, xmax=leng, color='gray')
plt.hlines(-conf_level, xmin=0, xmax=leng, color='gray')

plt.bar(x=np.arange(len(npcorr)), height=npcorr, width=.3)
# plt.bar(x=avgusd_game.index, height=ccs, width=.3) # x 길이는 같은데..  안됨..ccs값과 인덱스와 매핑이 안되는 듯

# /* Decoration */
plt.title('Cross Correlation Plot <numpy.correlate>', fontsize=22)
plt.xlim(0,len(npcorr))
plt.show()

# COMMAND ----------

# ccf
ccs = ccf(avgusd_game, avgusd_game, adjusted=False)
nlags = len(ccs)
leng = len(avgusd_game)

# /* Compute the Significance level */
conf_level = 2 / np.sqrt(nlags)
print('conf_level= ', conf_level)

# /* Draw Plot */
plt.figure(figsize=(30,10), dpi=80)

plt.hlines(0, xmin=0, xmax=leng, color='gray')  # 0 axis
plt.hlines(conf_level, xmin=0, xmax=leng, color='gray')
plt.hlines(-conf_level, xmin=0, xmax=leng, color='gray')

plt.bar(x=np.arange(len(ccs)), height=ccs, width=.3)
# plt.bar(x=avgusd_game.index, height=ccs, width=.3) # 안되네..

# /* Decoration */
plt.title('Cross Correlation Plot <statsmodels.CCF>', fontsize=22)
plt.xlim(0,len(ccs))
plt.show()

# COMMAND ----------

# 약 250일까지 두변수 들이 서로 같은 방향으로 변화(pro-cyclical:경기순응)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. CCF-CC 교차 상관계수(Cross Correlation)
# MAGIC - avgusd 카테고리별 비교, 시가총액과 비교
# MAGIC - 변수간 동행성(comovement) 측정
# MAGIC - 경기순응적(pro-cyclical) / 경기중립적(a-cyclical) / 경기역행적(counter-cyclical)

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 교차상관계수 차트 생성기

# COMMAND ----------

# ccf 계수 생성기
def ccf_data(data):
    ccfdata = ccf(data, data, adjusted=False)
    return ccfdata

# COMMAND ----------

## ccf 차트 생성기
def ccfcc_plot(data, feature):
    # 칼럼 리스트함수 호출
    col_list = feature_classifier(data, feature)
    
    # ccf 계수 함수 호출
    for col in col_list:
        ccfdata = ccf_data(data[col])
    
        # /* Compute the Significance level */
        nlags = len(ccfdata)
        conf_level = 2 / np.sqrt(nlags)
#         print('conf_level= ', conf_level)
        print('교차상관계수가 0에 가까운 지점 = ', min(np.where(ccfdata < 0)[0])-1)
        
        # /* Draw Plot */
        plt.figure(figsize=(30,10), dpi=80)

        plt.hlines(0, xmin=0, xmax=nlags, color='gray')  # 0 axis
        plt.hlines(conf_level, xmin=0, xmax=nlags, color='gray')
        plt.hlines(-conf_level, xmin=0, xmax=nlags, color='gray')

        plt.bar(x=np.arange(nlags), height=ccfdata, width=.3)
        # plt.bar(x=avgusd_game.index, height=ccs, width=.3) # 안되네..

        # /* Decoration */
        plt.title(f'Cross Correlation Plot <{col}>', fontsize=22)
        plt.xlim(0,nlags)
        plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 자기교차상관
# MAGIC - 전체카테고리별 인덱스204~366 (약6개월에서 1년주기)까지 동행성이 있음
# MAGIC - acf와 동일함

# COMMAND ----------

 # 전체 카테고리- 자기교차상관 시각화
ccfcc_plot(data, 'average_usd')

# COMMAND ----------

# acf와 동일한듯, 비교해보자. pacf는 50%이하 길이로만 가능
# 절반만 봐도 acf는 비슷하네.

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def autoCorrelationF1(data, feature):
    
        # 피처 분류기 호출
    col_list = feature_classifier(data, feature)
    
    for col in col_list:
        series = data[col]

        acf_array = acf(series.dropna(), alpha=0.05, nlags=850) 
        pacf_array = pacf(series.dropna(), alpha=0.05, nlags=850) # 50% 이하 길이까지만 가능
        array_list = [acf_array, pacf_array]

        fig = make_subplots(rows=1, cols=2)

        for i in range(len(array_list)) :
            corr_array = array_list[i]
            lower_y = corr_array[1][:,0] - corr_array[0]
            upper_y = corr_array[1][:,1] - corr_array[0]
        
            [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f', row=1, col=i+1)
             for x in range(len(corr_array[0]))]

            fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4', marker_size=12, row=1, col=i+1)

            fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)', row=1, col=i+1)

            fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
                fill='tonexty', line_color='rgba(255,255,255,0)', row=1, col=i+1)

# 
            fig.update_traces(showlegend=False)
#             fig.update_xaxes(range=[-1,42])
            fig.update_yaxes(zerolinecolor='#000000')

        fig.update_layout(title= f'<b>[{col}] Autocorrelation (ACF)                                 [{col}] Partial Autocorrelation (PACF)<b>', 
                         title_x=0.5)
        fig.show()

# COMMAND ----------

autoCorrelationF1(data, 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 상호교차상관
# MAGIC - 카테고리가 너무 많다. 4개만 교차해서 보자 collectible_avgusd, game_avgusd, all_avgusd, all_sales_usd
# MAGIC - 인덱스265~315 (약9개월에서 10개월주기)까지 동행성이 있음

# COMMAND ----------

## ccf 차트 생성기
def ccfcc_plot1(data):

    col_list = ['collectible_average_usd', 'game_average_usd','all_average_usd', 'all_sales_usd']
    x1col_list = []
    x2col_list = []
    ccfdata_list = []
    
    for i in range(len(col_list)-1):
        for j in range(1, len(col_list)):
            x1col_list.append(col_list[i])
            x2col_list.append(col_list[j])
            ccfdata_list.append(ccf(data[col_list[i]], data[col_list[j]], adjusted=False))
            
    plt.figure(figsize=(30,30), dpi=80)
    plt.suptitle("Cross Correlation Plot", fontsize=40)

    for i in range(len(ccfdata_list)):   
        ccfdata = ccfdata_list[i]
        # /* Compute the Significance level */
        nlags = len(ccfdata)
        conf_level = 2 / np.sqrt(nlags)

        # /* Draw Plot */
        plt.subplot(3, 3, i+1)   
        plt.title(f'<{x1col_list[i]} X {x2col_list[i]}, {min(np.where(ccfdata < 0)[0])-1} >', fontsize=22)
        plt.bar(x=np.arange(nlags), height=ccfdata, width=.3)
        plt.xlim(0,nlags)        

        plt.hlines(0, xmin=0, xmax=nlags, color='gray')  # 0 axis
        plt.hlines(conf_level, xmin=0, xmax=nlags, color='gray')
        plt.hlines(-conf_level, xmin=0, xmax=nlags, color='gray')
      
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# COMMAND ----------

#  2~3열이 상호교차상관 그래프
ccfcc_plot1(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. CCF-LC 시차 상관계수(leads and lags correlation)
# MAGIC - 시차 상호 상관(TLCC) https://dive-into-ds.tistory.com/96
# MAGIC - 선행적(leading) / 동행적(coincident) / 후행적(lagging)

# COMMAND ----------

# # lag 데이터프레임 생성기
# def lag_df(data, num):
#     col = data.columns
#     for i in range(1,num+1):
#         data[i] = data[col].shift(i)
#     return data

# COMMAND ----------

#  시차상관계수 계산함수
def TLCC(X1, X2, lag):
    result=[]
    for i in range(lag):
        result.append(X1.corr(X2.shift(i)))
    return np.round(result, 4)
#     print(f'시차상관계수가 가장 높은 lag = <{np.argmax(result)}>')

# COMMAND ----------

TLCC(data['game_average_usd'], data['collectible_average_usd'], 14)

# COMMAND ----------

TLCC(data['all_average_usd'], data['all_sales_usd'], 14)

# COMMAND ----------

TLCC(data['all_average_usd'], data['all_number_of_sales'], 100)

# COMMAND ----------

# MAGIC %md
# MAGIC ### avg_usd피처, 카테고리별 시차상관분석

# COMMAND ----------

# 카테고리별 피처 분류기
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

# defi는 21-01-16에 들어옴, 총 1704중ㅇ에 400개, 1/6도 안되므로 제외한다
# data[['defi_average_usd']]['2021-01-15':]
avgusd_col_list = feature_classifier(data, 'average_usd')
avgusd_col_list.remove('defi_average_usd')
# avgusd_col_list.remove('all_average_usd')
print(len(avgusd_col_list), avgusd_col_list ) 

# COMMAND ----------

## TLCC 차트 생성기
def TLCC_plot(data, col_list, nlag):

    x1col_list = []
    x2col_list = []
    TLCC_list = []

    for i in range(len(col_list)):
        for j in range(len(col_list)):
            if col_list[i] == col_list[j]:
                pass
            else:
                x1col_list.append(col_list[i])
                x2col_list.append(col_list[j])
                tlccdata =TLCC(data[col_list[i]], data[col_list[j]], nlag)
                TLCC_list.append(tlccdata)

    plt.figure(figsize=(30,40))
    plt.suptitle("TLCC Plot", fontsize=40)
    
    ncols = 3
    nrows = len(x1col_list)//3+1
    
    for i in range(len(TLCC_list)): 
        tlccdata = TLCC_list[i]
        plt.subplot(nrows, ncols, i+1)   
        plt.title(f'<{x1col_list[i]} X {x2col_list[i]}, {np.argmax(tlccdata)} >', fontsize=22)
        plt.plot(np.arange(len(tlccdata)), tlccdata)
        plt.xlim(-1,len(tlccdata)+1)        
        plt.vlines(np.argmax(tlccdata), ymin=min(tlccdata), ymax=max(tlccdata) , color='blue',linestyle='--',label='Peak synchrony')
#         plt.hlines(0, xmin=0, xmax=nlags, color='gray')  # 0 axis

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# COMMAND ----------

# 그래프 너무 많다. 보기 힘드니까 생략하자
# TLCC_plot(data, avgusd_col_list, 14)

# COMMAND ----------

## TLCC table 생성기
def TLCC_table(data, col_list, nlag):

    x1col_list = []
    x2col_list = []
    TLCC_list = []
    TLCC_max_idx_list = []
    TLCC_max_list = []
    havetomoreX1 = []
    havetomoreX2 = []
    result = []

    for i in range(len(col_list)):
        for j in range(len(col_list)):
            if col_list[i] == col_list[j]:
                pass
            else:
                x1col_list.append(col_list[i])
                x2col_list.append(col_list[j])
                tlccdata = TLCC(data[col_list[i]], data[col_list[j]], nlag)
                TLCC_list.append(tlccdata)
                
                TLCC_max_idx= np.argmax(tlccdata)
                TLCC_max_idx_list.append(TLCC_max_idx)
                if TLCC_max_idx == nlag-1:
                    havetomoreX1.append(col_list[i])
                    havetomoreX2.append(col_list[j])
    
                TLCC_max = max(tlccdata)
                TLCC_max_list.append(TLCC_max)
                if TLCC_max >= 0.9:
                    result.append('*****')  # 아주 높은 상관관계
                elif TLCC_max >= 0.7 and TLCC_max < 0.9: 
                    result.append('****')# 높은 상관관계가 있음
                elif TLCC_max >= 0.4 and TLCC_max < 0.7:
                    result.append('***')# 다소 상관관계가 있음
                elif TLCC_max >= 0.2 and TLCC_max < 0.4:
                    result.append('**')# 약한 상관관계
                elif TLCC_max < 0.2:
                    result.append('*')# 상관관계 거의 없음
                else :
                    print('분기 체크 필요')
                    
    # 결과 테이블 생성
    result_df = pd.DataFrame(data=list(zip(x1col_list, x2col_list, TLCC_max_idx_list, TLCC_max_list, result)), columns=['Lead(x1)', 'Lag(x2)', 'TLCC_max_idx', 'TLCC_max', 'result'])
    
    # max_tlcc_idx가 최대lag와 동일한 칼럼 반환                
    return havetomoreX1, havetomoreX2, result_df

# COMMAND ----------

# game이 후행인 경우는 모두 가장 높은 lag가 값이 높다. 더 올려보자
# utility는 다른카테고리와 거의 시차상관성이 없다.
havetomoreX1, havetomoreX2, result_df = TLCC_table(data, avgusd_col_list, 14)
result_df

# COMMAND ----------

print(havetomoreX1)
print(havetomoreX2)

# COMMAND ----------

for i in range(len(havetomoreX1)):
    tlccdata = TLCC(data[havetomoreX1[i]], data[havetomoreX2[i]], 150)
    print(havetomoreX1[i], havetomoreX2[i], np.argmax(tlccdata), np.round(max(tlccdata),4))

# COMMAND ----------

# 최대 lag값으로 다시 확인해보자
havetomoreX1, havetomoreX2, result_df = TLCC_table(data, avgusd_col_list, 150)
result_df

# COMMAND ----------

# 선행/후행을 쌍으로 재정렬하는 함수
def TLCC_table_filtered(data):
    result_x1x2_list = []
    result_after_x1 = []
    result_after_x2 = []
    for i in range(len(data)):
        result_x1x2_list.append(list(data.iloc[i, :2].values))

    for i in range(len(result_x1x2_list)):
        for j in range(len(result_x1x2_list)):
            if result_x1x2_list[i][0] == result_x1x2_list[j][1]  and result_x1x2_list[i][1] == result_x1x2_list[j][0]:
                result_after_x1.append(result_x1x2_list[i][0])
                result_after_x2.append(result_x1x2_list[i][1])
                result_after_x1.append(result_x1x2_list[j][0])
                result_after_x2.append(result_x1x2_list[j][1])


    result_x1x2_df = pd.DataFrame(data=list(zip(result_after_x1, result_after_x2)), columns=['after_x1','after_x2']) # 'x1->x2, x2->x1 쌍변수 리스트
    result_x1x2_df.drop_duplicates(inplace=True) # 중복 제거
    result_x1x2_df.reset_index(inplace=True) # 인덱스 리셋
    
    after_x1 = []
    after_x2 = []
    TLCC_max_idx = []
    TLCC_max = []
    result = []
    print('<<TLCC 데이터프레임에서 쌍변수순으로 필터링>>')
    for i in range(len(result_x1x2_df)):
        xrow = data[data['Lead(x1)']==result_x1x2_df['after_x1'][i]]
        x1x2row = xrow[xrow['Lag(x2)']==result_x1x2_df['after_x2'][i]]
        after_x1.append(x1x2row.values[0][0])
        after_x2.append(x1x2row.values[0][1])
        TLCC_max_idx.append(x1x2row.values[0][2])
        TLCC_max.append(x1x2row.values[0][3])
        result.append(x1x2row.values[0][4])

    result_df_filtered = pd.DataFrame(data=list(zip(after_x1, after_x2, TLCC_max_idx, TLCC_max, result)), columns=['Lead(x1)', 'Lag(x2)', 'TLCC_max_idx', 'TLCC_max', 'result'])
    return result_df_filtered

# COMMAND ----------

# 재정렬된 데이터프레임, 총 30개 행
result_df_filtered = TLCC_table_filtered(result_df)
print(len(result_df_filtered))
result_df_filtered

# COMMAND ----------

# 높은 상관관계만 추려보자(0.5 이상) 20개
good = result_df_filtered[result_df_filtered['TLCC_max'] >= 0.5] #  0.7이상이 18개
print(len(good))
good
# all->art(22), collectible/metaverse->all(54), all->game(44), art<->collectible(0), art->game(32), metaverse->art(99), collectible->game(58), meta->collec(95), meta->game(143)

# COMMAND ----------

# 보통/낮은 상관관계만 추려보자(0.5 이하) 10개
bad = result_df_filtered[result_df_filtered['TLCC_max'] <= 0.5]
print(len(bad))
bad

# COMMAND ----------

# 최근 한달 중앙값
data[avgusd_col_list][-30:].median()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [실험 결과] avg_usd 카테고리별 시차상관분석
# MAGIC ### 상관관계가 낮은 케이스
# MAGIC ####  - utility
# MAGIC ---
# MAGIC ### 상관관계가 높은 케이스
# MAGIC ####  - 동행 : art-collectible/metaverse, collectible-metaverse, game-collectible
# MAGIC     - 특이사항) art/collectible/metaverse는 모두 평단가가 높은 카테고리이다. 추정유저군인 전문투자자들은 즉각 반응 하나봄
# MAGIC       - art시장 가격거품이 빠지면 다른 시장도 영향을 받는 것임
# MAGIC     - 특이사항) game-collectible은 유일하게 전체 상관분석에도 높았었는데..아무래도 유저군이 겹치는 것으로 추정됨(이유는 아래 계속)
# MAGIC ####  - 지연 : art/collectible/metaverse-game(32,58,143), metaverse-collectible(95)
# MAGIC     - 특이사항) game이 선행인 지연케이스는 없음, 즉 게임평균가는 다른 카테고리를 리드하지 않는다.
# MAGIC       - 유입/활동이 제일 많아 nft마켓의 비중은 높으나 "마켓의 가격형성"에 영향을 주지 않는것으로 보아, 유저 오디언스 특징이 다른 것으로 추정. 라이트유저(게임하며 돈벌기) vs 헤비유저(전문투자자)
# MAGIC       - 투자관점에서 게임카테고리는 투자해봤자 돈이 안되겠네..
# MAGIC       - 게임만 다른카테고리와 분명하게 다른 경향을 보이는 이유
# MAGIC         - 평균가 범위 갭이 매우 큼, 최근 한달 중앙값 게임 193 vs 3514, 1384, 2402
# MAGIC         - game 평균가는 엄청 작은데 판매수 매우 많아서 시장가치(sales usd) 비중이 꽤 높음, 22년1월 중앙값 게임 25% vs 14.2%, 55.4%, 5.3% 
# MAGIC ---
# MAGIC ### 의문점1 : 왜 극단적으로 동행성(0) vs 1~5달지연(34-149)으로 나뉠까? 
# MAGIC ####  - 반응이 너무 느리다. 일정기간이 넘으면 무의미한것 아닐까..? 그것을 알기 위해, all을 같이 봐야겠다.
# MAGIC     - 동행 : all-collectible, art/game-all
# MAGIC     - 지연 : all-art/game(22, 44), collectible/metaverse-all(54, 54),
# MAGIC       - 의문점) all은 포괄인데 왜 art/game보다 선행하나? 재귀적 영향관계??
# MAGIC       - 의문점) 시장가치 비중 14%인 art가 all과 동행하고, 나머지 2개는 54일이다. 왜일까? 외부 요인이 있을 것으로 추정(언론이슈)
# MAGIC     - 종합 : 전체 평균가와 가장 높은 지연인 54를 기준으로 참고할 수 있을까? 아니면 매우 긴 지연도 유의미 하는걸까?(재귀적 영향관계로?) -
# MAGIC ---
# MAGIC ### 의문점2 : 선행/후행의 결과가 같거나 다를 경우 해석은 어떻게??
# MAGIC ####  - 상호 동행 : 거의 특징이 동일하다고 봐도 될듯
# MAGIC     - art<->collectible(0)
# MAGIC ####  - 편 지연(편동행 생략) : a는 b에 바로 반응이 갈 정도로 영향이 크지만, b는 상대적으로 낮다?
# MAGIC     - metaverse -> art/collectible(99,55) -> game(32,58), meta->game(14),  collectible/metaverse->all(54), 3) all->art/game(22,44)
# MAGIC       - 인과에 따라서 메타버스가 game에 영향을 주는 거라면 143이 유의미할 수도 있을 듯
# MAGIC       - all이 art/game에 재귀적으로 영향을 주는 거라면 all피처가 유의미할 수도 있을 듯
# MAGIC ####  - 상호 지연 : 즉시 반응을 줄정도의 영향력은 없는, 상대적으로 서로에게 낮은 영향력을 가졌나?
# MAGIC     - 없음, 이 케이스가 합리적인 명제인지도 모르겠음 헷갈림
# MAGIC ---
# MAGIC 위 의문을 해소하기 위한 인과검정이 필요하다.
# MAGIC ---
# MAGIC #### 케이스 셀렉션
# MAGIC - 공적분 검정용 케이스 : 일단..대표 지연케이스로 collectible->game(59)를 공적분 검증해보자

# COMMAND ----------

# MAGIC %md
# MAGIC #### 대표 케이스 시차상관계수 비교 테이블

# COMMAND ----------

avgusd_col_list

# COMMAND ----------

# 월 중앙값 집계 데이터
dataM_median = data.resample('M').median()
dataM_median.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] 시차상관계수 차트 생성기

# COMMAND ----------

#  시차상관계수 계산함수
def TLCC_comparison(X1, X2, start_lag, end_lag):
    result=[]
    laglist = []
    for i in range(start_lag, end_lag+1):
        result.append(X1.corr(X2.shift(i)))
        laglist.append(i)
    return laglist, np.round(result, 4)

# COMMAND ----------

# 차트 함수
def TLCC_comparison_table(data, x1, x2, startlag, endlag): # 데이터, 기준변수, 비교변수, startlag, endlag
    x2list = x2.copy()
    x2list.remove(x1)  # 입력한 변수에서 삭제되기때문에 사전 카피필요
    x2_list = [x1, *x2list]
    x1_list = []
    tlcc_list = []
    lag_var_list= []
    lvar_tlcc_list=[]
    sd_list = []
    rsd_list = []
    
    # x2별 lag, tlcc값 받아오기
    for i in range(len(x2_list)): 
        x2data = data[x2_list[i]]
        lag_list,  result = TLCC_comparison(data[x1], x2data, startlag, endlag) 
        tlcc_list.append(result)
        sd_list.append(np.std(x2data))   # =stdev(범위)
        rsd_list.append(np.std(x2data)/np.mean(x2data)*100)  # RSD = stdev(범위)/average(범위)*100, 
        # RSD(상대표준편차) or CV(변동계수) : 똑같은 방법으로 얻은 데이터들이 서로 얼마나 잘 일치하느냐 하는 정도를 가리키는 정밀도를 나타내는 성능계수, 값이 작을 수록 정밀하다.
        x1_list.append(x1)
        
    # 데이터프레임용 데이터 만들기
    temp = tlcc_list.copy()
    dfdata = list(zip(x1_list, x2_list, sd_list, rsd_list, *list(zip(*temp)))) # temp..array를 zip할수 있도록 풀어줘야함..
    
    # 데이터프레임용 칼럼명 리스트 만들기
    column_list = ['X1변수', 'X2변수', 'X2표준편차', 'X2상대표준편차', *lag_list]  

    result_df = pd.DataFrame(data=dfdata, columns= column_list,)

    return result_df

# COMMAND ----------

# 판다스 스타일의 천의자리 구분은 1.3 부터 지원함
# pd.__version__ #  pip install --upgrade pandas==1.3  # import pandas as pd

# 데이터프레임 비주얼라이제이션 함수
def visualDF(dataframe):
#     pd.set_option('display.precision', 2) # 소수점 글로벌 설정
    pd.set_option('display.float_format',  '{:.2f}'.format)
    dataframe = dataframe.style.bar(subset=['X2표준편차','X2상대표준편차'])\
    .background_gradient(subset=[*result_df.columns[4:]], cmap='Blues', vmin = 0.5, vmax = 0.9)\
    .set_caption(f"<b><<< X1변수({result_df['X1변수'][0]})기준 X2의 시차상관계수'>>><b>")\
    .format(thousands=',')\
    .set_properties(
        **{'border': '1px black solid !important'})
    return dataframe

# COMMAND ----------

# 월 중앙값 기준      # collectible에 대한 교차시차상관분석
print(f"<<<X1기준 X2의 변동폭 및 시차상관계수 테이블>>>")
result_df = TLCC_comparison_table(dataM_median, 'collectible_average_usd', avgusd_col_list, -6, 6)
result_df

# COMMAND ----------

# 월중앙값 전체기간
visualDF(result_df) 

# COMMAND ----------

# gmae이 생각보다 상관이 낮게 나왔다. game데이터는 2017년 데이터 없으므로, 2018년 이후 데이터로 다시 해보자

# COMMAND ----------

# 월 중앙값 기준 "2018년 이후 (game데이터는 2017년 데이터 없음)"
print(f"<<<X1기준 X2의 변동폭 및 시차상관계수 테이블>>>")
result_df = TLCC_comparison_table(dataM_median['2018':], 'collectible_average_usd', avgusd_col_list, -6, 6)
result_df

# COMMAND ----------

# 월중앙값 2018년 이후
visualDF(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### [결론] 월 중앙값 기준 시차상관분석(collectible_avgusd 기준)
# MAGIC - 2018년이후 데이터로 분석하니, 모든 카테고리 상관성이 높아졌다.(특히 과거 시차관련)
# MAGIC - collectible의 자기상관도는 매우 높으나 RSD 정밀도가 낮다.
# MAGIC - RSD(상대표준편차)는 metaverse가 상대적으로 정밀도가 높고, art와 all의 정밀도가 낮다.
# MAGIC - utility는 상관성이 없다.
# MAGIC - metaverse는 y변수가 음수 일 때 상관성이 매우 높으므로 X가 후행한다. metaverse -> collectible  "매우 명확"
# MAGIC - all, art, game은 y변수가 양수일 때 상관성이 음수일 보다 상대적으로 더 높다.
# MAGIC   - 그런데 -2음수일때도 높은 것으로 보다 상호지연관계가 있으면서, 동시에 X의 선행 영향력이 더 크다. collectible <->> all/art/game(단 게임은 비교적 짧다)

# COMMAND ----------

# MAGIC %md
# MAGIC ### all카테고리, 피처별 시차상관분석

# COMMAND ----------

all_col_list = ['all_active_market_wallets','all_number_of_sales','all_average_usd','all_primary_sales','all_primary_sales_usd','all_sales_usd','all_secondary_sales','all_secondary_sales_usd','all_unique_buyers','all_unique_sellers']
print(len(all_col_list), all_col_list) # 총 10개 카테고리

# COMMAND ----------

# 그래프 너무 많다. 보기 힘드니까 생략하자
#  TLCC_plot(data, all_col_list, 14)

# COMMAND ----------

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# COMMAND ----------

# avgusd가 후행인경우 lag값이 가장 높다. 더 올려보자
havetomoreX1, havetomoreX2, result_df = TLCC_table(data, all_col_list, 14)
result_df

# COMMAND ----------

print(havetomoreX1)
print(havetomoreX2)

# COMMAND ----------

for i in range(len(havetomoreX1)):
    tlccdata = TLCC(data[havetomoreX1[i]], data[havetomoreX2[i]], 150)
    print(havetomoreX1[i], havetomoreX2[i], np.argmax(tlccdata), np.round(max(tlccdata),4))

# COMMAND ----------

# 최대 lag값으로 다시 확인해보자
havetomoreX1, havetomoreX2, result_df = TLCC_table(data, all_col_list, 150)
result_df

# COMMAND ----------

# 재정렬된 데이터프레임, 총 90개행
result_df_filtered = TLCC_table_filtered(result_df)
print(len(result_df_filtered))
result_df_filtered

# COMMAND ----------

# 높은 상관관계만 추려보자(0.75 이상) 총 93개행
good = result_df_filtered[result_df_filtered['TLCC_max'] >= 0.75]
print(len(good))
good
# 동행성-동행
# 총지갑수<->총판매수/1차판매수/2차판매수/2차매출/구매자수/판매자수,  총판매수<->1차판매수/2차판매수/2차매출/구매자수/판매자수, 1차판매수<->2차판매수/2차매출/구매자수, 총매출<->2차매출
# 2차판매수<->구매자수/판매자수, 2차매출<->구매자수, 구매자수<->판매자수

# 동행-지연
# 총지갑수->총매출(124), 총판매수->1차매출(132)/총매출(123), 평균가->1차매출(98), 총매출->평균가(98), 1차판매수->1차매출(119)/총매출(117)/판매자수(143), 총매출->1차매출(98), 2차매출->1차매출(118)
# 2차판매수->총매출(127), 구매자수->총매출(123), 판매자수->총매출(130), 2차판매수->2차매출(125)
# 판매자수->2차매출(127)

# 지연-지연
#  총지갑수<->평평균가(100<->70),1차매출(132<->56)  , 총판매수<->평균가(100,66), 평균가<->1차판매수(66,100),2차판매수(65, 100),2차매출(33,98),구매자수(71,100),판매자수(67,100),  1차매출<->2차판매수(56,134)
# 1차매출<->구매자수(56,132),판매자수(56,135)

# COMMAND ----------

# 보통/낮은 상관관계만 추려보자(0.75 이하) 없음 7개
bad = result_df_filtered[result_df_filtered['TLCC_max'] <= 0.75]
print(len(bad))
bad

# COMMAND ----------

# MAGIC %md
# MAGIC #### [실험 결과] all카테고리 피처별 시차상관분석 
# MAGIC 
# MAGIC ### 상관관계가 낮은 케이스
# MAGIC ####  - 대체로 높다. 1차매출이 리드하는 경우가 상대적으로 0.6~7로 낮은편
# MAGIC ---
# MAGIC ### 상관관계가 높은 케이스
# MAGIC - 항목이 많아 분석이 어렵다. 구체적인 과제에 맞춰 분석하는게 좋을 듯
# MAGIC - **평균가 기준) 총지갑수/총판매수/1차판매수/2차판매수/2차매출/구매자수/판매자수는 평균가와 대체로 2~3달의 상호지연관계이고, 총매출과 평균가 그리고 1차매출은 약 3달의 편지연관계를 갖는다.**
# MAGIC - **특이사항) 시차지연의 경우, 위 평균가 피처별분석와 동일하거나 상대적으로 높은 편이고(33~143), "상호지연관계" 많다.**
# MAGIC - **의문점 ) "평균가와 구매자수의 지연관계가 2~3달이면 생각보다 너무 길다"**  
# MAGIC 
# MAGIC #### 상호 동행
# MAGIC - 총지갑수<->총판매수/1차판매수/2차판매수/2차매출/구매자수/판매자수,  총판매수<->1차판매수/2차판매수/2차매출/구매자수/판매자수, 
# MAGIC - 1차판매수<->2차판매수/2차매출/구매자수, 총매출<->2차매출, 2차판매수<->구매자수/판매자수, 2차매출<->구매자수, 구매자수<->판매자수 
# MAGIC 
# MAGIC #### 편 지연(편동행 생략)
# MAGIC - 총지갑수->총매출(124), 총판매수->1차매출(132)/총매출(123), 평균가->1차매출(98), 총매출->평균가(98), 1차판매수->1차매출(119)/총매출(117)/판매자수(143)
# MAGIC - 총매출->1차매출(98), 2차매출->1차매출(118), 2차판매수->총매출(127), 구매자수->총매출(123), 판매자수->총매출(130), 2차판매수->2차매출(125), 판매자수->2차매출(127)
# MAGIC 
# MAGIC #### 상호 지연
# MAGIC - 총지갑수<->평균가(100,70)/1차매출(132,56), 총판매수<->평균가(100,66), 평균가<->1차판매수(66,100)/2차판매수(65, 100)/2차매출(33,98)/구매자수(71,100)/판매자수(67,100)
# MAGIC - 1차매출<->2차판매수(56,134), 1차매출<->구매자수(56,132),판매자수(56,135)
# MAGIC 
# MAGIC ---
# MAGIC #### 케이스 셀렉션
# MAGIC - 공적분 검정용 케이스 : 일단..대표 지연케이스로 avgusd->buyer(71)을 공적분 검증해보자(avg_usd를 예측했으니까..)**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 대표 케이스 시차상관계수 비교 테이블

# COMMAND ----------

all_col_list

# COMMAND ----------

# 월 중앙값 집계 데이터
dataM_median.head()

# COMMAND ----------

# 월 중앙값 기준
print(f"<<<X1기준 X2의 변동폭 및 시차상관계수 테이블>>>")
result_df = TLCC_comparison_table(dataM_median, 'all_average_usd', all_col_list, -6, 6)
result_df

# COMMAND ----------

# 월중앙값 기준
visualDF(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### [결론] 월 중앙값 기준 시차상관분석(all_avgusd 기준)
# MAGIC - all_avgusd의 자기상은 한달 전후가 매우 높음
# MAGIC - RSD는 1차판매수의 정밀도가 상대적으로 높은편이다.
# MAGIC - 대체로 상관성이 매우 높은데 X2가 음수일 때 상관성이 상대적으로 더 높으므로 X1가 후행한다. X2 -> 평균가
# MAGIC - 특이점은 일부(가격류)를 제외하고 2달 내외부터 상관성이 높아진다는 것. 즉 가격류는 상호 동행하고 그외는 약2달의 지연 관계가 있다.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 시각화(pass)
# MAGIC - 예제 해석을 못하겠어서 pass

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 예제1: line
# MAGIC - 어떻게 해석을 해야할지 모르겠다

# COMMAND ----------

def crosscorr(datax1, datax2, lag=0, wrap=False):
# """ Lag-N cross correlation. 
# Shifted data filled with NaNs 

# Parameters
# ----------
# lag : int, default 0
# datax, datay : pandas.Series objects of equal length
# wrap : NaN 채우는 것. shift 하면서 사라진 값으로 다시 채우기. 값이 순환되게 된다. wrap=False 이면 NaN은 drop하고 correlation 구한다.
# Returns
# ----------
# crosscorr : float
# """
    if wrap:
        shiftedx2 = datax2.shift(lag)
        shiftedx2.iloc[:lag] = datax2.iloc[-lag:].values
        return datax1.corr(shiftedx2)
    else: 
        return datax1.corr(datay.shift(lag))

# COMMAND ----------

#  해석어케함.. offset 양수이므로 s2가 선행한다? 59일차?
s1 = data['game_average_usd']
s2 = data['collectible_average_usd']

rs = [crosscorr(s1,s2, lag) for lag in range(-300, 300)]
offset = np.floor(len(rs)/2)-np.argmax(rs) # 최대 correlation 값 가지는 offset 계산

f,ax=plt.subplots(figsize=(30,5))
# print(rs)
ax.plot(rs)
ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Offset = {offset} \nS1 leads <> S2 leads', xlabel='Offset',ylabel='Pearson r')
# ax.set_xticks(range(-300, 300))
ax.set_xticklabels([-300, -150, -50, 0, 50, 150, 300])
plt.legend()

# Offset이 왼쪽에 있으면, S1이 리드하과 S2가 따라오는 것
# shift(-150)이 d2에 대해서 적용되고, d2의 미래와 d1의 현재간에 correlation 계산 하는 것. 즉, offset이 음수이면 d1이 선행한다는 뜻
# 이것도 결국 global level로 correlation 측정하는 것. 시차 두면서.

# COMMAND ----------

#  해석어케함.. offset 양수이므로 s2가 선행한다? 이건 이상한데.. 평균가보다 마켓이 선행한다고?. 세일즈랑 비교해봐야하나.
s1 = data['all_average_usd']
s2 = data['all_sales_usd']

rs = [crosscorr(s1,s2, lag) for lag in range(-300, 300)]
offset = np.floor(len(rs)/2)-np.argmax(rs) # 최대 correlation 값 가지는 offset 계산

f,ax=plt.subplots(figsize=(30,5))
# print(rs)
ax.plot(rs)
ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Offset = {offset} \nS1 leads <> S2 leads', xlabel='Offset',ylabel='Pearson r')
# ax.set_xticks(range(-300, 300))
ax.set_xticklabels([-300, -150, -50, 0, 50, 150, 300])
plt.legend()

# Offset이 왼쪽에 있으면, S1이 리드하과 S2가 따라오는 것
# shift(-150)이 d2에 대해서 적용되고, d2의 미래와 d1의 현재간에 correlation 계산 하는 것. 즉, offset이 음수이면 d1이 선행한다는 뜻
# 이것도 결국 global level로 correlation 측정하는 것. 시차 두면서.

# COMMAND ----------

#  해석어케함.. 모르겠다.
s1 = data['all_average_usd']
s2 = data['all_number_of_sales']

rs = [crosscorr(s1,s2, lag) for lag in range(-300, 300)]
offset = np.floor(len(rs)/2)-np.argmax(rs) # 최대 correlation 값 가지는 offset 계산

f,ax=plt.subplots(figsize=(30,5))
# print(rs)
ax.plot(rs)
ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Offset = {offset} \nS1 leads <> S2 leads', xlabel='Offset',ylabel='Pearson r')
# ax.set_xticks(range(-300, 300))
ax.set_xticklabels([-300, -150, -50, 0, 50, 150, 300])
plt.legend()

# Offset이 왼쪽에 있으면, S1이 리드하과 S2가 따라오는 것
# shift(-150)이 d2에 대해서 적용되고, d2의 미래와 d1의 현재간에 correlation 계산 하는 것. 즉, offset이 음수이면 d1이 선행한다는 뜻
# 이것도 결국 global level로 correlation 측정하는 것. 시차 두면서.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 예제2: heatmap
# MAGIC - 이것도 뭘어떻게 봐야할지 모르겠다.

# COMMAND ----------

data.shape[0]//20

# COMMAND ----------

no_splits = 30
samples_per_split = data.shape[0]//no_splits
rss=[]

for t in range(0, no_splits):
    d1 = data['game_average_usd'].iloc[(t)*samples_per_split:(t+1)*samples_per_split]
    d2 = data['collectible_average_usd'].iloc[(t)*samples_per_split:(t+1)*samples_per_split]
    rs = [crosscorr(d1,d2, lag) for lag in range(-300,300)]
    rss.append(rs)
rss = pd.DataFrame(rss)
f,ax = plt.subplots(figsize=(30,10))
sb.heatmap(rss, cmap='RdBu_r',ax=ax)
ax.set(title=f'Windowed Time Lagged Cross Correlation', xlabel='Offset',ylabel='Window epochs')
# ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
# ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);

# Rolling window time lagged cross correlation
window_size = 300 #samples
t_start = 0
t_end = t_start + window_size
step_size = 30
rss=[]
while t_end < 1704:
    d1 = data['game_average_usd'].iloc[t_start:t_end]
    d2 = data['collectible_average_usd'].iloc[t_start:t_end]
    rs = [crosscorr(d1,d2, lag, wrap=False) for lag in range(-300,300)]
    rss.append(rs)
    t_start = t_start + step_size
    t_end = t_end + step_size
rss = pd.DataFrame(rss)

f,ax = plt.subplots(figsize=(30,10))
sb.heatmap(rss,cmap='RdBu_r',ax=ax)
ax.set(title=f'Rolling Windowed Time Lagged Cross Correlation',xlabel='Offset',ylabel='Epochs')
# ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
# ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 공적분 검정(Cointegration Test)
# MAGIC ####- 개념 : 시계열 Y와 시계열 X 모두 <<일차 차분이 안정적인 시계열(I(1)과정, 1차적분과정)이고>> 두시계열 사이에 안정적인 선형결합이 존재한다면 두시계열간에 공적분이 존재한다고 정의한다.
# MAGIC   - 즉 X와 Y가 공적분관계에 있다면 아무리 X와 Y가 불안정적 확률변수(I(1))라고 해도 두변수에 대한 회귀식을 세워도 허구적인 회귀현상이 나타나지 않는다.
# MAGIC     - 공적분 관계는, 단기적으로 서로 다른 패턴을 보이지만 장기적으로 볼 때 일정한 관계가 있음을 의미함, 
# MAGIC 
# MAGIC ####- 검정 방법 : 대표적으로 요한슨 검정을 많이 함
# MAGIC   - (단변량) engel & granget 검정 : ADF단위근검정 아이디어
# MAGIC   - (다변량) johansen 검정 : ADF단위근검정을 다변량의 경우로 확장하여 최우추정을 통해 검정 수행

# COMMAND ----------

# MAGIC %md
# MAGIC ### (단변량)Engle-Granger 2step OLS Test
# MAGIC - statsmodels.coint 는 engle-granger 기반, [signatrue](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.coint.html)
# MAGIC - 회귀분석 결과의 잔차항에 대해 검정
# MAGIC   - OLS추정량을 이용하여 회귀모형 Y=bX+z을 추정하여 잔차항 zhat을 구한다. 그리고 잔차항 zhat에 대한 DF검정을 수행한다.
# MAGIC   - 일반 DF임계값과는 다른 임계갑을 사용해야 한다.('공적분 검정 임계값 분포표')
# MAGIC 
# MAGIC - 귀무가설 : 비정상 시계열 간의 조합에 따른 오차항에 단위근이 존재한다. 즉, 서로 공적분 관계가 존재하지 않는다.
# MAGIC   - p-value값이 5%보다 작을 때 귀무가설을 기각하여 공적분관계가 있음을 알 수 있다.
# MAGIC 
# MAGIC - 앵글&그레인저 검정의 한계로 일반적으로 요한슨을 많이 사용한다.
# MAGIC   - 시계열 사이에 1개의 공적분 관계만 판별할 수 있음, 즉3개 이상의 시계열사이의 공적분 검정 불가
# MAGIC   - 회귀 모형으로 장기균형관계를 판단할 때 표본의 크기가 무한하지 않으면 종속변수로 어떤 시계열을 선택하는지에 따라 검정결과가 달라지는 문제가 있고, 시계열 수가 증가하면 더 심해짐
# MAGIC   
# MAGIC - [앵글&그레인저 공적분 검정 예제1](https://mkjjo.github.io/finance/2019/01/25/pair_trading.html)
# MAGIC - [앵글&그레인저 공적분 검정 예제2](https://lsjsj92.tistory.com/584)

# COMMAND ----------

import statsmodels.tsa.stattools as ts
# pd.set_option('display.precision', 2) 
# pd.options.display.float_format = '{:.2f}'.format

# COMMAND ----------

# 공적분 관계 시각화 (두변수간의 비율이 평균을 중심으로달라지는지 확인) -> 어떻게 보는거지? 장기적으로 편차가 적어지면 장기적 관계가 있다??
import statsmodels.tsa.stattools as ts
x1 = data['collectible_average_usd']['2018':]
x2 = data['game_average_usd']['2018':]

# 디폴트 : raw데이터(로그변환/스케일링등 정규화하면 안됨, 특징 사라짐), augmented engle&granger(default), maxlag(none), trend='c'
score, pvalue, _ = ts.coint(x1,x2)
print(f'추세 상수 only //  ADF score={np.round(score, 4)} // coint test p-value={np.round(pvalue, 4)}')
score, pvalue, _ = ts.coint(x1,x2, trend='ct')
print(f'추세 상수&기울기 //  ADF score={np.round(score, 4)} // coint test p-value={np.round(pvalue, 4)}')
score, pvalue, _ = ts.coint(x1,x2, trend='ctt')
print(f'추세 상수&기울기(2차) //  ADF score={np.round(score, 4)} // coint test p-value={np.round(pvalue, 4)}')
score, pvalue, _ = ts.coint(x1,x2, trend='nc')
print(f'추세 없음 //  ADF score={np.round(score, 4)} // coint test p-value={np.round(pvalue, 4)}')

(x2/x1).plot(figsize=(30,10))
plt.axhline((x2/x1).mean(), color='red', linestyle='--')
plt.xlabel('Time')
plt.title('collectible / game Ratio')
plt.legend(['collectible / game Ratio', 'Mean'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [EG결과] collectible avgusd vs game avgusd
# MAGIC - 추세 상수&기울기(2차) 케이스 : p-value값이 0.85로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음, VAR모형 채택**
# MAGIC - 추세 없음 케이스 : p-value값이 0.33로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음, VAR모형 채택**

# COMMAND ----------

# 공적분 관계 시각화 -> 관계가 있는거야뭐야?
import statsmodels.tsa.stattools as ts
x1 = data['all_average_usd']
x2 = data['all_unique_buyers']

# 디폴트 : raw데이터(로그변환/스케일링등 정규화하면 안됨, 특징 사라짐), augmented engle&granger(default), maxlag(none), trend='c'
score, pvalue, _ = ts.coint(x1,x2)
print(f'추세 상수 only //  ADF score={np.round(score, 4)} // coint test p-value={np.round(pvalue, 4)}')
score, pvalue, _ = ts.coint(x1,x2, trend='ct')
print(f'추세 상수&기울기 //  ADF score={np.round(score, 4)} // coint test p-value={np.round(pvalue, 4)}')
score, pvalue, _ = ts.coint(x1,x2, trend='ctt')
print(f'추세 상수&기울기(2차) //  ADF score={np.round(score, 4)} // coint test p-value={np.round(pvalue, 4)}')
score, pvalue, _ = ts.coint(x1,x2, trend='nc')
print(f'추세 없음 //  ADF score={np.round(score, 4)} // coint test p-value={np.round(pvalue, 4)}')

(x2/x1).plot(figsize=(30,10))
plt.axhline((x2/x1).mean(), color='red', linestyle='--')
plt.xlabel('Time')
plt.title('all_buyers / all_avg_usd Ratio')
plt.legend(['all_buyers / all_avg_usd Ratio', 'Mean'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [EG결과] all_avgusd vs all_buyers
# MAGIC - 추세 상수&기울기(2차) 케이스 : p-value값이 0.55로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음, VAR모형 채택**
# MAGIC - 추세 없음 케이스 : p-value값이 0.13로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음, VAR모형 채택**
