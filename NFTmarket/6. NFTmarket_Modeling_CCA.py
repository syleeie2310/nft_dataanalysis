# Databricks notebook source
import numpy as np
import pandas as pd

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



# COMMAND ----------

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
    xcol_list = []
    ycol_list = []
    ccfdata_list = []
    
    for i in range(len(col_list)-1):
        for j in range(1, len(col_list)):
            xcol_list.append(col_list[i])
            ycol_list.append(col_list[j])
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
        plt.title(f'<{xcol_list[i]} X {ycol_list[i]}, {min(np.where(ccfdata < 0)[0])-1} >', fontsize=22)
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
def TLCC(X, Y, lag):
    result=[]
    print(lag)
    for i in range(lag):
        print(i)
        result.append(X.corr(Y.shift(i)))
        print(result)
    return np.round(result, 4)
#         print(i, np.round(result[i], 4))
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

# defi는 21-01-16에 들어옴, 총 1704중ㅇ에 400개, 1/6도 안되므로 제외한다
# data[['defi_average_usd']]['2021-01-15':]
avgusd_col_list = feature_classifier(data, 'average_usd')
avgusd_col_list.remove('defi_average_usd')
# avgusd_col_list.remove('all_average_usd')
print(len(avgusd_col_list), avgusd_col_list ) 

# COMMAND ----------

## TLCC 차트 생성기
def TLCC_plot(data, col_list, nlag):

    xcol_list = []
    ycol_list = []
    TLCC_list = []

    for i in range(len(col_list)):
        for j in range(len(col_list)):
            if col_list[i] == col_list[j]:
                pass
            else:
                xcol_list.append(col_list[i])
                ycol_list.append(col_list[j])
                tlccdata =TLCC(data[col_list[i]], data[col_list[j]], nlag)
                TLCC_list.append(tlccdata)

    plt.figure(figsize=(30,40))
    plt.suptitle("TLCC Plot", fontsize=40)
    
    ncols = 3
    nrows = len(xcol_list)//3+1
    
    for i in range(len(TLCC_list)): 
        tlccdata = TLCC_list[i]
        plt.subplot(nrows, ncols, i+1)   
        plt.title(f'<{xcol_list[i]} X {ycol_list[i]}, {np.argmax(tlccdata)} >', fontsize=22)
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

    xcol_list = []
    ycol_list = []
    TLCC_list = []
    TLCC_max_idx_list = []
    TLCC_max_list = []
    havetomoreX = []
    havetomoreY = []
    result = []

    for i in range(len(col_list)):
        for j in range(len(col_list)):
            if col_list[i] == col_list[j]:
                pass
            else:
                xcol_list.append(col_list[i])
                ycol_list.append(col_list[j])
                tlccdata = TLCC(data[col_list[i]], data[col_list[j]], nlag)
                TLCC_list.append(tlccdata)
                
                TLCC_max_idx= np.argmax(tlccdata)
                TLCC_max_idx_list.append(TLCC_max_idx)
                if TLCC_max_idx == nlag-1:
                    havetomoreX.append(col_list[i])
                    havetomoreY.append(col_list[j])
    
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
    result_df = pd.DataFrame(data=list(zip(xcol_list, ycol_list, TLCC_max_idx_list, TLCC_max_list, result)), columns=['Lead(X)', 'Lag(Y)', 'TLCC_max_idx', 'TLCC_max', 'result'])
    
    # max_tlcc_idx가 최대lag와 동일한 칼럼 반환                
    return havetomoreX, havetomoreY, result_df

# COMMAND ----------

# game이 후행인 경우는 모두 가장 높은 lag가 값이 높다. 더 올려보자
# utility는 다른카테고리와 거의 시차상관성이 없다.
havetomoreX, havetomoreY, result_df = TLCC_table(data, avgusd_col_list, 14)
result_df

# COMMAND ----------

print(havetomoreX)
print(havetomoreY)

# COMMAND ----------

for i in range(len(havetomoreX)):
    tlccdata = TLCC(data[havetomoreX[i]], data[havetomoreY[i]], 150)
    print(havetomoreX[i], havetomoreY[i], np.argmax(tlccdata), np.round(max(tlccdata),4))

# COMMAND ----------

# 최대 lag값으로 다시 확인해보자
havetomoreX, havetomoreY, result_df = TLCC_table(data, avgusd_col_list, 150)
result_df

# COMMAND ----------

# 선행/후행을 쌍으로 재정렬하는 함수
def TLCC_table_filtered(data):
    result_xy_list = []
    result_after_x = []
    result_after_y = []
    for i in range(len(data)):
        result_xy_list.append(list(data.iloc[i, :2].values))

    for i in range(len(result_xy_list)):
        for j in range(len(result_xy_list)):
            if result_xy_list[i][0] == result_xy_list[j][1]  and result_xy_list[i][1] == result_xy_list[j][0]:
                result_after_x.append(result_xy_list[i][0])
                result_after_y.append(result_xy_list[i][1])
                result_after_x.append(result_xy_list[j][0])
                result_after_y.append(result_xy_list[j][1])


    result_XY_df = pd.DataFrame(data=list(zip(result_after_x, result_after_y)), columns=['after_X','after_Y']) # 'x->y, y->x 쌍변수 리스트
    result_XY_df.drop_duplicates(inplace=True) # 중복 제거
    result_XY_df.reset_index(inplace=True) # 인덱스 리셋
    
    after_X = []
    after_Y = []
    TLCC_max_idx = []
    TLCC_max = []
    result = []
    print('<<TLCC 데이터프레임에서 쌍변수순으로 필터링>>')
    for i in range(len(result_XY_df)):
        xrow = data[data['Lead(X)']==result_XY_df['after_X'][i]]
        xyrow = xrow[xrow['Lag(Y)']==result_XY_df['after_Y'][i]]
        after_X.append(xyrow.values[0][0])
        after_Y.append(xyrow.values[0][1])
        TLCC_max_idx.append(xyrow.values[0][2])
        TLCC_max.append(xyrow.values[0][3])
        result.append(xyrow.values[0][4])

    result_df_filtered = pd.DataFrame(data=list(zip(after_X, after_Y, TLCC_max_idx, TLCC_max, result)), columns=['Lead(X)', 'Lag(Y)', 'TLCC_max_idx', 'TLCC_max', 'result'])
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
def TLCC_comparison(X, Y, start_lag, end_lag):
    result=[]
    laglist = []
    for i in range(start_lag, end_lag+1):
        result.append(X.corr(Y.shift(i)))
        laglist.append(i)
    return laglist, np.round(result, 4)

# COMMAND ----------

# 차트 함수
def TLCC_comparison_table(data, X, Y, startlag, endlag): # 데이터, 기준변수, 비교변수, startlag, endlag
    Ylist = Y.copy()
    Ylist.remove(X)  # 입력한 변수에서 삭제되기때문에 사전 카피필요
    Yindex_list = [X, *Ylist]
    tlcc_list = []
    lag_var_list= []
    lvar_tlcc_list=[]
    sd_list = []
    rsd_list = []
    
    # y별 lag, tlcc값 받아오기
    for i in range(len(Yindex_list)): 
        ydata = data[Yindex_list[i]]
        lag_list,  result = TLCC_comparison(data[X], ydata, startlag, endlag) 
        tlcc_list.append(result)
        sd_list.append(np.std(ydata))   # =stdev(범위)
        rsd_list.append(np.std(ydata)/np.mean(ydata)*100)  # stdev(범위)/average(범위)*100

#     # lag별 tlcc값 바인딩 변수 만들기(=칼럼)
#     for i in range(len(lag_list)):
#         lag_var_list.append([]) #  lag별 tlcc값을 바인딩할 그릇 생성
#         for j in range(len(tlcc_list)):
#              lag_var_list[i].append(tlcc_list[j][i])

    # 데이터프레임용 데이터 만들기
    temp = tlcc_list.copy()
    dfdata = list(zip(Yindex_list, sd_list, rsd_list, *list(zip(*temp)))) # temp..array를 zip할수 있도록 풀어줘야함..
    
    # 데이터프레임용 칼럼명 리스트 만들기
    column_list = ['Y변수', '표준편차', '상대표준편차', *lag_list]  

    result_df = pd.DataFrame(data=dfdata, columns= column_list,)
#     result_df = pd.DataFrame(data=dfdata, index = Yindex_list, columns= column_list)
#     result_df.index.name = f"X={X}" #  인덱스 이름 변경

    return result_df

# COMMAND ----------

# 월 중앙값 기준      # collectible에 대한 교차시차상관분석
print(f"<<<X기준 Y의 변동폭 및 시차상관계수 테이블>>>")
result_df = TLCC_comparison_table(dataM_median, 'collectible_average_usd', avgusd_col_list, -6, 6)
result_df

# COMMAND ----------

## 데이터프레임 스타일
# result_df.style.set_precision(2)
pd.set_option('display.precision', 2) # 소수점 글로벌 설정
result_df.style.background_gradient(cmap='Blues').set_caption(f"<b><<<'X(0)기준 Y의 변동폭 및 시차상관계수'>>><b>")
# df.style.applymap(lambda i: 'background-color: red' if i > 3 else '')

# COMMAND ----------

# gmae이 생각보다 상관이 낮게 나왔다. game데이터는 2017년 데이터 없으므로, 2018년 이후 데이터로 다시 해보자

# COMMAND ----------

# 월 중앙값 기준 "2018년 이후 (game데이터는 2017년 데이터 없음)"
print(f"<<<X기준 Y의 변동폭 및 시차상관계수 테이블>>>")
result_df = TLCC_comparison_table(dataM_median['2018':], 'collectible_average_usd', avgusd_col_list, -6, 6)
result_df

# COMMAND ----------

## 데이터프레임 스타일 "2018년 이후 (game데이터는 2017년 데이터 없음)"
# result_df.style.set_precision(2)
pd.set_option('display.precision', 2) # 소수점 글로벌 설정
result_df.style.background_gradient(cmap='Blues').set_caption(f"<b><<<'X(0)기준 Y의 변동폭 및 시차상관계수'>>><b>")
# df.style.applymap(lambda i: 'background-color: red' if i > 3 else '')

# COMMAND ----------

# MAGIC %md
# MAGIC #### [결론] 월 중앙값 기준 시차상관분석(collectible_avgusd 기준)
# MAGIC - 2018년이후 데이터로 분석하니, 모든 카테고리 상관성이 높아졌다.(특히 과거 시차관련)
# MAGIC - utility는 상관관계 없음
# MAGIC - metaverse는 -lag가 관계가 있고 +lag는 관계가 떨어지는 것으로 보아, meta -> collec 관계로 보임
# MAGIC - art, game 모두 +lag관계가 높은 것으로 보아, collec->meta관계로 보임, art는 6개월차에 관계가 높아짐
# MAGIC - 표준편차/ 상대표준편차 값이 너무 커서 판단이 어렵다. 평균을 함께 봐야하나?

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
havetomoreX, havetomoreY, result_df = TLCC_table(data, all_col_list, 14)
result_df

# COMMAND ----------

print(havetomoreX)
print(havetomoreY)

# COMMAND ----------

for i in range(len(havetomoreX)):
    tlccdata = TLCC(data[havetomoreX[i]], data[havetomoreY[i]], 150)
    print(havetomoreX[i], havetomoreY[i], np.argmax(tlccdata), np.round(max(tlccdata),4))

# COMMAND ----------

# 최대 lag값으로 다시 확인해보자
havetomoreX, havetomoreY, result_df = TLCC_table(data, all_col_list, 150)
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
print(f"<<<X기준 Y의 변동폭 및 시차상관계수 테이블>>>")
result_df = TLCC_comparison_table(dataM_median, 'all_average_usd', all_col_list, -6, 6)
result_df

# COMMAND ----------

## 데이터프레임 스타일
# result_df.style.set_precision(2) #안되네..
pd.set_option('display.precision', 2) # 소수점 글로벌 설정
# pd.set_option("styler.format.thousands", ",")#안되네..
# result_df.style.format(thousands=",") # 안됨
result_df.style.background_gradient(cmap='Blues').set_caption(f"<b><<<'X(0)기준 Y의 변동폭 및 시차상관계수'>>><b>")

# df.style.applymap(lambda i: 'background-color: red' if i > 3 else '')

# COMMAND ----------

# MAGIC %md
# MAGIC #### [결론] 월 중앙값 기준 시차상관분석(all_avgusd 기준)
# MAGIC - 자기상관 : 한달 전후만 있음
# MAGIC - 상호지연관계 : 지갑수, 판매수, 1차판매수, 2차판매수, 구매자수, 판매자수
# MAGIC - 상호동행관계 : 1차매출
# MAGIC - 편지연관계 : 총매출과 2차매출이 평균가에 영향을 줌
# MAGIC - 표준편차/ 상대표준편차 값이 너무 커서 판단이 어렵다. 평균을 함께 봐야하나?

# COMMAND ----------

# MAGIC %md
# MAGIC ### 시각화(pass)
# MAGIC - 예제 해석을 못하겠어서 pass

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 예제1: line
# MAGIC - 어떻게 해석을 해야할지 모르겠다

# COMMAND ----------

def crosscorr(datax, datay, lag=0, wrap=False):
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
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

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
# MAGIC   - engel & granget 검정 : ADF단위근검정 아이디어
# MAGIC   - johansen 검정 : ADF단위근검정을 다변량의 경우로 확장하여 최우추정을 통해 검정 수행

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
X = data['collectible_average_usd']['2018':]
Y = data['game_average_usd']['2018':]

# 디폴트 : raw데이터(로그변환/스케일링등 정규화하면 안됨, 특징 사라짐), augmented engle&granger(default), maxlag(none), trend='c'
score, pvalue, _ = ts.coint(X,Y)
print('Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('ADF score: ' + str( np.round(score, 4) ))
print('Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 상수&기울기')
score, pvalue, _ = ts.coint(X,Y, trend='ct')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 상수&기울기(2차)')
score, pvalue, _ = ts.coint(X,Y, trend='ctt')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 없음')
score, pvalue, _ = ts.coint(X,Y, trend='nc')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))

(Y/X).plot(figsize=(30,10))
plt.axhline((Y/X).mean(), color='red', linestyle='--')
plt.xlabel('Time')
plt.title('collectible / game Ratio')
plt.legend(['collectible / game Ratio', 'Mean'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [EG결과] collectible avgusd vs game avgusd
# MAGIC - 추세 상수&기울기(2차) 케이스 : p-value값이 0.85로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음**
# MAGIC - 추세 없음 케이스 : p-value값이 0.33로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음**

# COMMAND ----------

# 공적분 관계 시각화 -> 관계가 있는거야뭐야?
import statsmodels.tsa.stattools as ts
X = data['all_average_usd']
Y = data['all_unique_buyers']

# 디폴트 : raw데이터(로그변환/스케일링등 정규화하면 안됨, 특징 사라짐), augmented engle&granger(default), maxlag(none), trend='c'
score, pvalue, _ = ts.coint(X,Y)
print('Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('ADF score: ' + str( np.round(score, 4) ))
print('Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 상수&기울기')
score, pvalue, _ = ts.coint(X,Y, trend='ct')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 상수&기울기(2차)')
score, pvalue, _ = ts.coint(X,Y, trend='ctt')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 없음')
score, pvalue, _ = ts.coint(X,Y, trend='nc')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))

(Y/X).plot(figsize=(30,10))
plt.axhline((Y/X).mean(), color='red', linestyle='--')
plt.xlabel('Time')
plt.title('총구매자수/총평균가 Ratio')
plt.legend(['총구매자수/총평균가 Ratio', 'Mean'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [EG결과] all_avgusd vs all_buyers
# MAGIC - 추세 상수&기울기(2차) 케이스 : p-value값이 0.55로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음**
# MAGIC - 추세 없음 케이스 : p-value값이 0.13로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음**

# COMMAND ----------

# MAGIC %md
# MAGIC ### (다변량)Johansen Test
# MAGIC #### 다변량 시계열 분석에 포함, 
# MAGIC - VAR모형을 기반으로 가설검정을 통해 적분계열간 안정적인 장기균형관계가 존재하는지 점검하는 방법
# MAGIC - 3개 이상의 불안정 시계열 사이의 공적분 검정에 한계를 갖는 앵글&그렌저 검정 방법을 개선하여 다변량에도 공적분 검정을 할 수 있음
# MAGIC - statsmodels.tsa.vector_ar.vecm. coint_johansen 
# MAGIC   - VAR(VECM)의 공적분 순위를 결정하기 위함
# MAGIC   - [signature](https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html)

# COMMAND ----------

#

# COMMAND ----------

from statsmodels.tsa.vector_ar.vecm import coint_johansen

# COMMAND ----------

X = data[avgusd_col_list]
X.head()

# COMMAND ----------

jresult = coint_johansen(X, det_order=0, k_ar_diff=1)
jresult.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 그레인저 인과검정(Granger Causality)
# MAGIC - 개념 : 동일한 시간축의 범위를 가진 두 데이터가 있을 때 한 데이터를 다른 한쪽의 데이터의 특정한 시간간격에 대해서 선형회귀를 할 수 있다면 그래인저 인과가 있다고 하는 것이다.
# MAGIC   - A lags + B lags로 B의 데이터를 선형회귀한 것의 예측력 > B lags로만 B의 데이터를 선형회귀한 것의 예측력
# MAGIC - 유의 : 인과의 오해석 경계 필요. (인과관계의 여지가 있다정도로 해석)
# MAGIC   - 달걍의 개체수 증가와 미래의 닭의 개체 수 증가에 인과 영향 결과가 있다고 해서 반드시 닭의 수의 요인은 달걀의 개체수라고 말하기엔 무리가 있음. 단순한 확대해석이기 때문, 그래서 "일반적인 인과관계"를 말하는 것이 아니므로 사람들이 생각하는 추상적인 인과관계를 명확하게 밝혀주는 것이 아니다. 
# MAGIC   - 그레인저 인과관계는 상관관계처럼 결과를 해석할 때 논리적으로 결함이 없는지 고찰하고 해석할 떄 주의해야함.
# MAGIC - **전제조건**
# MAGIC   - 입력파라미터 : 선행시계열, 후행시계열, 시차(지연)
# MAGIC   - 시계열 데이터 정상성
# MAGIC     - KPSS테스트를 통해 정상성을 만족하는 시차를 찾아낸다.
# MAGIC     - 5.TSA에서 단위근검정을 통해 1차 차분의 정상성을 확인했으므로 생략한다.
# MAGIC   - 테스트 방향 : 변수 A, B의 양방향으로 2회 검정 세트 수행이 일반적이며, 결과에 따라 해석이 달라지는 어려움이 있음
# MAGIC - 귀무가설
# MAGIC   - 유의 수준을 0.05(5%)로 설정하였고 테스트를 통해서 검정값(p-value)가 0.05이하로 나오면 귀무가설을 기각할 수 있다. 귀무가설은 “Granger Causality를 따르지 않는다” 이다.
# MAGIC - [클라이브 그레인저 위키](https://ko.wikipedia.org/wiki/%ED%81%B4%EB%9D%BC%EC%9D%B4%EB%B8%8C_%EA%B7%B8%EB%A0%88%EC%9D%B8%EC%A0%80)
# MAGIC - [그레인저 인과관계](https://intothedata.com/02.scholar_category/timeseries_analysis/granger_causality/)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 정상성 시차 찾기
# MAGIC - 통계적 가설 검정(Unit root test:단위근검정)
# MAGIC - 단위근 : 단위근이란 확률론의 데이터 검정에서 쓰이는 개념으로 시계열 데이터는 시간에 따라 일정한 규칙을 가짐을 가정한다
# MAGIC 
# MAGIC #### 1. Augmented Dickey-Fuller("ADF") Test
# MAGIC - 시계열에 단위근이 존재하는지 검정,단위근이 존재하면 정상성 시계열이 아님.
# MAGIC - 귀무가설이 단위근이 존재한다.
# MAGIC - 검증 조건 ( p-value : 5%이내면 reject으로 대체가설 선택됨 )
# MAGIC - 귀무가설(H0): non-stationary. 대체가설 (H1): stationary.
# MAGIC - adf 작을 수록 귀무가설을 기각시킬 확률이 높다
# MAGIC #### 2. Kwiatkowski-Phillips-Schmidt-Shin (“KPSS”) Test
# MAGIC - KPSS 검정은 1종 오류의 발생가능성을 제거한 단위근 검정 방법이다.
# MAGIC - DF 검정, ADF 검정과 PP 검정의 귀무가설은 단위근이 존재한다는 것이나, KPSS 검정의 귀무가설은 정상 과정 (stationary process)으로 검정 결과의 해석 시 유의할 필요가 있다.
# MAGIC   - 귀무가설이 단위근이 존재하지 않는다.
# MAGIC - 단위근 검정과 정상성 검정을 모두 수행함으로서 정상 시계열, 단위근 시계열, 또 확실히 식별하기 어려운 시계열을 구분하였다.
# MAGIC - KPSS 검정은 단위근의 부재가 정상성 여부에 대한 근거가 되지 못하며 대립가설이 채택되면 그 시계열은 trend-stationarity(추세를 제거하면 정상성이 되는 시계열)을 가진다고 할 수 있습니다.
# MAGIC - 때문에 KPSS 검정은 단위근을 가지지 않고 Trend- stationary인 시계열은 비정상 시계열이라고 판단할 수 있습니다.

# COMMAND ----------

#  시차상관계수 계산함수 (coint는 앵글&그레인저 기반으로 pvalue는 adf단위근 검정)
def adf_lag(X, Y, start_lag, end_lag):
    pvalue=[]
    for i in range(start_lag, end_lag+1):
        _, p, _ = ts.coint(X,Y.shift(i, fill_value=0))
        pvalue.append(p)
    return pvalue

def kpss_lag(X, Y, start_lag, end_lag):
    pvalue=[]
    for i in range(start_lag, end_lag+1):
        stats, p, lag, _ = kpss(X, regression="ct", nlags=i)
        pvalue.append(p)
    return pvalue

# COMMAND ----------

# MAGIC %md
# MAGIC ##### collectible_avgusd & game_avgusd
# MAGIC - 그레인저인과검정과 비교할 수 있도록 lag를 15로 잡자
# MAGIC - KPSS기준 최대 11개월까지 상호지연관계 "정상성" 있음, cg 11, gc12

# COMMAND ----------

xcol = 'collectible_average_usd'
ycol = 'game_average_usd'
X = dataM_median[xcol]['2018':]
Y = dataM_median[ycol]['2018':]

startlag = -15
endlag = 15
# adf_pval = adf_lag(X,Y, startlag, endlag)
# kpss_pval = kpss_lag(X,Y, startlag, endlag)

fig = plt.figure(figsize=(30,10))
plt.suptitle("lag difference sationary check <ADF & KPSS>", fontsize=30)

plt.subplot(2, 1, 1)   
plt.title('<ADF pvalue>', fontsize=22)
plt.plot(range(startlag, endlag+1), adf_lag(X,Y, startlag, endlag), label = f'{xcol} -> {ycol}')
plt.plot(range(startlag, endlag+1), adf_lag(Y,X, startlag, endlag), label = f'{ycol} -> {xcol}')
plt.legend(loc='center left')
plt.hlines(0.05, xmin=startlag, xmax=endlag, color='blue')

plt.subplot(2, 1, 2)
plt.title('<KPSS pvalue>', fontsize=22)
plt.plot(range(startlag, endlag+1), kpss_lag(X,Y, startlag, endlag), label = f'{xcol} -> {ycol}')
plt.plot(range(startlag, endlag+1), kpss_lag(Y,X, startlag, endlag), label = f'{ycol} -> {xcol}')
plt.legend(loc='center left')
plt.hlines(0.05, xmin=startlag, xmax=endlag, color='blue')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### all_avgusd & all_buyers
# MAGIC - 그레인저인과검정과 비교할 수 있도록 lag를 15로 잡자
# MAGIC - KPSS기준 최대 12개월까지 상호지연관계 "정상성" 있음, ub12, bu15

# COMMAND ----------

xcol = 'all_average_usd'
ycol = 'all_unique_buyers'
X = dataM_median[xcol]
Y = dataM_median[ycol]

startlag = -15
endlag = 15
# adf_pval = adf_lag(X,Y, startlag, endlag)
# kpss_pval = kpss_lag(X,Y, startlag, endlag)

fig = plt.figure(figsize=(30,10))
plt.suptitle("lag difference sationary check <ADF & KPSS>", fontsize=30)

plt.subplot(2, 1, 1)   
plt.title('<ADF pvalue>', fontsize=22)
plt.plot(range(startlag, endlag+1), adf_lag(X,Y, startlag, endlag), label = f'{xcol} -> {ycol}')
plt.plot(range(startlag, endlag+1), adf_lag(Y,X, startlag, endlag), label = f'{ycol} -> {xcol}')
plt.legend(loc='center left')
plt.hlines(0.05, xmin=startlag, xmax=endlag, color='blue')

plt.subplot(2, 1, 2)
plt.title('<KPSS pvalue>', fontsize=22)
plt.plot(range(startlag, endlag+1), kpss_lag(X,Y, startlag, endlag), label = f'{xcol} -> {ycol}')
plt.plot(range(startlag, endlag+1), kpss_lag(Y,X, startlag, endlag), label = f'{ycol} -> {xcol}')
plt.legend(loc='center left')
plt.hlines(0.05, xmin=startlag, xmax=endlag, color='blue')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 그레인저 인과분석
# MAGIC - 딕셔너리 언패킹을 못해서 시각화못함
# MAGIC - **정상성시차 : 최대 cg11, gc12 ub12, bu15**
# MAGIC - from statsmodels.tsa.stattools import grangercausalitytests [signature](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html)
# MAGIC   - 2개 시계열의 그랜저 비인과성에 대한 4가지 테스트.
# MAGIC   - 2번째 시계열이 1번째 시계열을 유발하는지 테스트(2->1)
# MAGIC   - maxlag = 15가 최대

# COMMAND ----------

from statsmodels.tsa.stattools import grangercausalitytests

# COMMAND ----------

# collectible -> game, 6~15까지 귀무가설 기각하여 collectible로 game을 예측 할 수 있음
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(data[['game_average_usd', 'collectible_average_usd']]['2018':], maxlag=15)

# COMMAND ----------

# collectible -> game, 6~15까지 귀무가설 기각하여 collectible로 game을 예측 할 수 있음
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(dataM_median[['game_average_usd', 'collectible_average_usd']]['2018':], maxlag=15)

# COMMAND ----------

#  game -> collectible, 1~10까지 귀무가설기각하여 game으로 collectible을 예측할 수 있음
grangercausalitytests(dataM_median[['collectible_average_usd', 'game_average_usd']]['2018':], maxlag=15)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### collectible_avgusd & game_avgusd
# MAGIC - ***정상성 시차 : 최대 cg11, gc12***
# MAGIC - ***그레인저인과 시차 : cg 6 ~ 15, gc1 ~ 10***
# MAGIC - ***pvalue 기준***
# MAGIC   - collectible이 game을 6 ~ 11개월 선행한다. 
# MAGIC   - game이 collectible을 1 ~ 10개월 선행한다. 
# MAGIC - ***f stats 기준 : gc가 더 높으므로, c가 먼저 g를 리드하고 이후 반대로 다시 영향을 받는다.***
# MAGIC   - c -> g, lag 6, 4.3468  
# MAGIC   - g -> c, lag 6, 39.8356
# MAGIC ---
# MAGIC - ***종합 해석***
# MAGIC   - 상호인과관계이나 g가 c에게 더 빠른 영향를 준다.(1달만에)
# MAGIC   - 그러나 상호인과관계가 있는 6개월 기준으로 보았을 때, c가 g를 더 리드한다.
# MAGIC   - 상호인과관계가 성립되므로 제 3의 외부 변수 영향 가능성이 높다.(ex 외부언론, 홍보 등), 이 경우 var모형을 사용해야한다.

# COMMAND ----------

# buyer -> avgusd, 2~15까지 귀무가설 기각하여 buyer로 avgusd를 예측 할 수 있음
grangercausalitytests(dataM_median[['all_average_usd', 'all_unique_buyers']], maxlag=15)

# COMMAND ----------

# avgusd -> buyer, 1~15까지 귀무가설 기각하여 avgusd로 buyer를 예측 할 수 있음
grangercausalitytests(dataM_median[['all_unique_buyers', 'all_average_usd']], maxlag=15)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### all_avgusd & all_buyers
# MAGIC - **정상성 시차 : 최대 ub 12, bu 15**
# MAGIC - **그레인저인과 시차 : ub1 ~ 15 , bu 2 ~ 15**
# MAGIC - **pvalue 기준**
# MAGIC   - avgusd가  buyer를 1 ~ 12개월 선행한다. 
# MAGIC   - buyer가 avgusd를 2 ~ 15개월 선행한다.
# MAGIC 
# MAGIC - **f stats 기준 : bu가 더 높으므로, u가 먼저 b를 리드하고 이후 반대로 다시 영향을 받는다.**
# MAGIC   - u -> b, lag 2, 40.0170 
# MAGIC   - b -> u, lag 2, 59.8666
# MAGIC ---
# MAGIC - **종합 해석**
# MAGIC   - b는 거의 동행성을 보인다.
# MAGIC   - 상호인과관계이나, u가 b에게 더 빠른 영향를 준다.(1달만에, 근데 비슷함)
# MAGIC   - 그러나 상호인과관계가 있는 2개월 기준으로 보았을 때, u가 b를 더 리드한다.
# MAGIC   - 상호인과관계가 성립되므로 제 3의 외부 변수 영향 가능성이 높다.(ex 외부언론, 홍보 등), 이 경우 var모형을 사용해야한다.

# COMMAND ----------

# collectible -> all, 2~13 귀무가설 기각
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(dataM_median[['all_average_usd', 'collectible_average_usd']]['2018':], maxlag=15)

# COMMAND ----------

# all -> collectible, 3~11 귀무가설 기각
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(dataM_median[['collectible_average_usd', 'all_average_usd']]['2018':], maxlag=15)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### all_avgusd & collectible_avgusd
# MAGIC - all -> collectible : 3~11 귀무가설 기각, 3기준 fstats 16.1708
# MAGIC - collectible -> all : 2~13 귀무가설 기각, 3기준 fstats 75.9002

# COMMAND ----------

# collectible -> buyers 1~15 귀무가설 기각
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(dataM_median[['all_unique_buyers', 'collectible_average_usd']]['2018':], maxlag=15)

# COMMAND ----------

# buyers -> collectible 5~11 귀무가설 기각
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(dataM_median[['collectible_average_usd', 'all_unique_buyers']]['2018':], maxlag=15)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### all_buyers & collectible_avgusd
# MAGIC - buyers -> collectible 5~11 귀무가설 기각, 5기준 fstats 13.7463
# MAGIC - collectible -> buyers 1~15 귀무가설 기각, 5기준 fstats 35.7845

# COMMAND ----------

# game -> all 1~8, 15 귀무가설 기각
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(dataM_median[['all_average_usd', 'game_average_usd']]['2018':], maxlag=15)

# COMMAND ----------

# all -> game 5~15 귀무가설 기각
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(dataM_median[['game_average_usd', 'all_average_usd']]['2018':], maxlag=15)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### all_avgusd & game_avgusd
# MAGIC - all -> game 5~15 귀무가설 기각, 5기준 fstats 16.0765
# MAGIC - game -> all 1~8, 15 귀무가설 기각, 5기준 fstats 29.9136

# COMMAND ----------

# game -> buyers 4~15 귀무가설 기각
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(dataM_median[['all_unique_buyers', 'game_average_usd']]['2018':], maxlag=15)

# COMMAND ----------

# buyers -> game  6~14 귀무가설 기각
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(dataM_median[['game_average_usd', 'all_unique_buyers']]['2018':], maxlag=15)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### all_buyers & game_avgusd
# MAGIC - buyers -> game  6~14 귀무가설 기각, 6기준 fstats 4.3648
# MAGIC - game -> buyers 4~15 귀무가설 기각, 6기준 fstats 39.6156

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 제3의 외부 변수 추가
# MAGIC - 가격 형성 요인으로 외부 이슈(언론, 홍보, 커뮤니티) 요인으로 추정됨
# MAGIC - 커뮤니티 데이터(ex: nft tweet)를 구하지 못해 포털 검색 데이터(rate, per week)를 대안으로 분석해보자

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 미니 EDA
# MAGIC - 주단위 수치형 "비율" 데이터
# MAGIC - 1%미만은 1으로 사전에 변경

# COMMAND ----------

gtkwd_data = pd.read_csv('/dbfs/FileStore/nft/google_trend/nft_googletrend_w_170423_220423.csv', index_col = "Date", parse_dates=True, thousands=',')

# COMMAND ----------

gtkwd_data.info()

# COMMAND ----------

gtkwd_data.rename(columns={'nft':'nft_gt'}, inplace=True)
gtkwd_data.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 미니 시각화
# MAGIC - 분포 : 1이 77%
# MAGIC - 추세 : 2021년 1월부터 급등해서 6월까라 급락했다가 22년1월까지 다시 급등 이후 하락세
# MAGIC - 범위 : 21년도 이후 iqr범위는 10~40, 중위값은 약25, 최대 약 85, 

# COMMAND ----------

gtkwd_dataM_median = gtkwd_data.resample('M').median()
gtkwd_dataM_median.tail()

# COMMAND ----------

plt.figure(figsize=(30,10))

plt.subplot(2, 2, 1)   
plt.title('<weekly_raw>', fontsize=22)
plt.hist(gtkwd_data)

plt.subplot(2, 2, 2)
plt.title('<monthly_median>', fontsize=22)
plt.hist(gtkwd_dataM_median)

plt.show()

# COMMAND ----------

plt.figure(figsize=(30,10))

plt.subplot(2, 2, 1)   
plt.title('<weekly_raw>', fontsize=22)
plt.plot(gtkwd_data)

plt.subplot(2, 2, 2)
plt.title('<monthly_median>', fontsize=22)
plt.plot(gtkwd_dataM_median)

plt.show()

# COMMAND ----------

plt.figure(figsize=(30,10))

plt.subplot(2, 2, 1)   
plt.title('<weekly_raw>', fontsize=22)
plt.boxplot(gtkwd_data['2021':])

plt.subplot(2, 2, 2)
plt.title('<monthly_median>', fontsize=22)
plt.boxplot(gtkwd_dataM_median['2021':])

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 데이터 통합

# COMMAND ----------

marketdataM = data['2018':'2022-01'].resample('M').median()
marketdataM.tail()

# COMMAND ----------

# 월간 통합
total = pd.merge(marketdataM, gtkwd_dataM, left_index=True, right_index=True, how='left')
total.tail()

# COMMAND ----------

# 주간 통합
marketdataW = data['2018':'2022-01'].resample('W').median()
totalW = pd.merge(marketdataW, gtkwd_data, left_index=True, right_index=True, how='left')
totalW.tail()

# COMMAND ----------

# 주간 통합
total = pd.merge(marketdata, gtkwd_data, left_index=True, right_index=True, how='left')
total.tail()

# COMMAND ----------

# 정규화
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
total_scaled = total.copy()
total_scaled.iloc[:,:] = minmax_scaler.fit_transform(total_scaled)
total_scaled.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 미니 상관분석
# MAGIC - 확인결과 스케일링 정규화랑 차이 없음, raw데이터로 보면됨, 월간과 주간 차이 없음

# COMMAND ----------

# [함수] 카테고리별 히트맵 생성기
import plotly.figure_factory as ff

# 카테고리 분류기
def category_classifier(data, category):
    col_list = []
    for i in range(len(data.columns)):
        if data.columns[i].split('_')[0] == category:
            col_list.append(data.columns[i])
        else :
            pass
    return col_list

def heatmapC(data, category):
    # 카테고리 분류기 호출
    col_list = category_classifier(data, category)
    col_list.append('nft_gt')
    
    # 삼각행렬 데이터 및 mask 생성
    corr = round(data[col_list].corr(), 2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # 상부 삼각행렬 생성(np.tilu()은 하부), np.ones_like(bool)와 함께 사용하여 값이 있는 하부삼각행렬은 1(true)를 반환한다.
    # 하부를 만들면 우측기준으로 생성되기 때문에 왼쪽기준으로 생성되는 상부를 반전한다.
    df_mask = corr.mask(mask)

    
    fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
        x=df_mask.columns.tolist(),
        y=df_mask.columns.tolist(),
        colorscale='Blues',
        hoverinfo="none", #Shows hoverinfo for null values
        showscale=True,
        xgap=3, ygap=3, # margin
        zmin = 0, zmax=1     
        )
    
    fig.update_xaxes(side="bottom") # x축타이틀을 하단으로 이동

    fig.update_layout(
        title_text='<b>Correlation Matrix (ALL 카테고리 피처간 상관관계)<b>', 
        title_x=0.5, 
#         width=1000, height=1000,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed', # 하단 삼각형으로 변경
        template='plotly_white'
    )

    # NaN 값은 출력안되도록 숨기기
    for i in range(len(fig.layout.annotations)):
        if fig.layout.annotations[i].text == 'nan':
            fig.layout.annotations[i].text = ""

    fig.show()
    

# COMMAND ----------

# [함수] 피처별 히트맵 생성기
import plotly.figure_factory as ff

def heatmapF(data, feature):
    # 피처 분류기 호출
    col_list = feature_classifier(data, feature)
    col_list.append('nft_gt')
     # all 카테고리 제외
#     new_col_list = []
#     for col in col_list:
#         if col.split('_')[0] != 'all':
#             new_col_list.append(col)
#         else: pass
    
    corr = round(data[col_list].corr(), 2)
        
    # 삼각행렬 데이터 및 mask 생성
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # 상부 삼각행렬 생성(np.tilu()은 하부), np.ones_like(bool)와 함께 사용하여 값이 있는 하부삼각행렬은 1(true)를 반환한다.
    # 하부를 만들면 우측기준으로 생성되기 때문에 왼쪽기준으로 생성되는 상부를 반전한다.
   
    df_mask = corr.mask(mask)

    
    fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
        x=df_mask.columns.tolist(),
        y=df_mask.columns.tolist(),
        colorscale='Blues',
        hoverinfo="none", #Shows hoverinfo for null values
        showscale=True,
        xgap=3, ygap=3, # margin
        zmin = 0, zmax=1     
        )
    
    fig.update_xaxes(side="bottom") # x축타이틀을 하단으로 이동

    fig.update_layout(
        title_text='<b>Correlation Matrix ("average USD"피처, 카테고리간 상관관계)<b>', 
        title_x=0.5, 
#         width=1000, height=1000,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed', # 하단 삼각형으로 변경
        template='plotly_white'
    )

    # NaN 값은 출력안되도록 숨기기
    for i in range(len(fig.layout.annotations)):
        if fig.layout.annotations[i].text == 'nan':
            fig.layout.annotations[i].text = ""

    fig.show()
    

# COMMAND ----------

# nft_gt와 대체로 상관관계가 높음, utility제외(collectible이 가장 높음)
heatmapC(total, 'all')

# COMMAND ----------

# 대체로 상관관계가 높다. 상대적으로 avgusd가 낮지만 그래도 높은편이니 인과검정 할만 한듯(아무리 생각해도 nft가격은...커뮤니티영향이 클 것 같은데.. nft tweet 데이터가 없어서 아쉽다.)
heatmapC(total['2021':], 'all')

# COMMAND ----------

# nft_gt와 대체로 상관관계가 높음, utility제외(collectible이 가장 높음) 하지만, 2018~2020까지 모두 1이라서 판단하기 어려움
heatmapF(total, 'average_usd')

# COMMAND ----------

# 본격적으로 검색량이 많아진 21년도부터 차이가 확연하다.
# all기준 검색량과 상관관계가 높은편, metaverse, collectible, art가 가장 높고, defi는 낮은 수준. collectible, game과 인과검정해보자
heatmapF(total['2021':], 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 시차상관분석

# COMMAND ----------

nftgt_list = ['nft_gt', 'collectible_average_usd', 'game_average_usd', 'all_average_usd', 'all_unique_buyers']

# COMMAND ----------

# 월 중앙값, 2021년도 이후
print(f"<<<X기준 Y의 변동폭 및 시차상관계수 테이블>>>")
result_df = TLCC_comparison_table(total['2021':], 'nft_gt', nftgt_list, -6, 6)
result_df

# COMMAND ----------

## 데이터프레임 스타일(색 구간 설정 해야함, 볼 때 유의)
pd.set_option('display.precision', 2) # 소수점 글로벌 설정
result_df.style.background_gradient(cmap='Blues').set_caption(f"<b><<<'X(0)기준 Y의 변동폭 및 시차상관계수'>>><b>")

# COMMAND ----------

# 주간 기준, 2021년도 이후
print(f"<<<X기준 Y의 변동폭 및 시차상관계수 테이블>>>")
result_df = TLCC_comparison_table(totalW['2021':], 'nft_gt', nftgt_list, -12, 12)
result_df

# COMMAND ----------

## 데이터프레임 스타일(색 구간 설정 해야함, 볼 때 유의)
pd.set_option('display.precision', 2) # 소수점 글로벌 설정
result_df.style.background_gradient(cmap='Blues').set_caption(f"<b><<<'X(0)기준 Y의 변동폭 및 시차상관계수'>>><b>")

# COMMAND ----------

# MAGIC %md
# MAGIC ###### ㄴ nft&cau&gau 시차상관분석(2021년 이후 월중앙값 & 주간)
# MAGIC - nftgt & nftgt : +- 3개월 정도 상관성이 있음
# MAGIC - nftgt & cau : -2개월부터 상관성이 높고, +는 매우 높음, nftgt -> cau관계로 추정
# MAGIC - nftgt & gau : 1부터 상관성이 높음으나 cau에 상대적으로 낮음 nftgt -> cau관계로 추정
# MAGIC - nftgt & au : 0부터 높음, nftgt -> au관계로 추정
# MAGIC - nftgt & ub : -2 ~ 0높았은데, 1~2에 잠시 하락했다가 급등, ub->nftgt 관계인가? 뭐지??

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 공적분 검정
# MAGIC - 앵글&그레인저, 주간데이터
# MAGIC - nftgt 와 시차상관성이 높은 cau와 aub만 대표로 보자

# COMMAND ----------

# 공적분 관계 시각화
X = totalW['nft_gt']['2021':]
Y = totalW['collectible_average_usd']['2021':]

# 디폴트 : raw데이터(로그변환/스케일링등 정규화하면 안됨, 특징 사라짐), augmented engle&granger(default), maxlag(none), trend='c'
import statsmodels.tsa.stattools as ts
score, pvalue, _ = ts.coint(X,Y)
print('Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('ADF score: ' + str( np.round(score, 4) ))
print('Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 상수&기울기')
score, pvalue, _ = ts.coint(X,Y, trend='ct')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 상수&기울기(2차)')
score, pvalue, _ = ts.coint(X,Y, trend='ctt')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 없음')
score, pvalue, _ = ts.coint(X,Y, trend='nc')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))

(Y/X).plot(figsize=(30,10))
plt.axhline((Y/X).mean(), color='red', linestyle='--')
plt.xlabel('Time')
plt.title('collectible_avgusd / nft_gt Ratio')
plt.legend(['collectible_avgusd / nft_gt Ratio', 'Mean'])
plt.show()

# COMMAND ----------

# 공적분 관계 시각화
X = totalW['nft_gt']['2021':]
Y = totalW['all_unique_buyers']['2021':]

# 디폴트 : raw데이터(로그변환/스케일링등 정규화하면 안됨, 특징 사라짐), augmented engle&granger(default), maxlag(none), trend='c'
import statsmodels.tsa.stattools as ts
score, pvalue, _ = ts.coint(X,Y)
print('Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('ADF score: ' + str( np.round(score, 4) ))
print('Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 상수&기울기')
score, pvalue, _ = ts.coint(X,Y, trend='ct')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 상수&기울기(2차)')
score, pvalue, _ = ts.coint(X,Y, trend='ctt')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))
print('='*50)

print('추세 없음')
score, pvalue, _ = ts.coint(X,Y, trend='nc')
print('Rawdata Correlation: ' + str( np.round(X.corr(Y), 4) ))
print('Rawdata ADF score: ' + str( np.round(score, 4) ))
print('Rawdata Cointegration test p-value: ' + str( np.round(pvalue, 4) ))

(Y/X).plot(figsize=(30,10))
plt.axhline((Y/X).mean(), color='red', linestyle='--')
plt.xlabel('Time')
plt.title('all_buyers / nft_gt Ratio')
plt.legend(['all_buyers / nft_gt', 'Mean'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### ㄴ 앵글&그레인저 검정 결과(주간)
# MAGIC - nftgt & cau : ctt기준 pval  0.3798로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음**
# MAGIC - nftgt & ub : ctt기준 pval 0.4232 로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음** 

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 그레인저 인과검정
# MAGIC - nftgt 와 시차상관성이 높은 cau와 aub만 대표로 보자
# MAGIC - 월중앙값으로 보면 인과검정 실패하여, 주간으로 다시 봄
# MAGIC - 

# COMMAND ----------

# MAGIC %md
# MAGIC ###### ㄴ nft_gt & cau(주간)

# COMMAND ----------

# nft_gt -> cau, 주간
# f검정 pval이 0.05초과하여 귀무가설 채택, 인과관계 없음, 그나마 2가 0.06으로 가까운편
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(totalW[['collectible_average_usd', 'nft_gt']]['2021':], maxlag=12)

# COMMAND ----------

# cau -> nft_gt, 주간
# 1~2가 f검정 pval이 0.05미만으로 귀무가설 기각, 그레인저 인과검정 통과
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(totalW[['nft_gt', 'collectible_average_usd']]['2021':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### ㄴ nft_gt & gau(주간)

# COMMAND ----------

# nft_gt -> gau, 주간
# 1주 f검정 pval이 0.05미만으로 귀무가설 기각, 인과검정 통과
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(totalW[['game_average_usd', 'nft_gt']]['2021':], maxlag=12)

# COMMAND ----------

# gau -> nft_gt, 주간
# 3주 f검정 pval이 0.05미만으로 귀무가설 기각, 인과검정 통과
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(totalW[['nft_gt', 'game_average_usd']]['2021':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### ㄴ nft_gt & aau(주간)

# COMMAND ----------

# nft_gt -> aau, 주간
# f검정 pval이 0.05초과하여 귀무가설 채택, 인과관계 없음, 그나마 1가 0.09으로 가까운편
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(totalW[['all_average_usd', 'nft_gt']]['2021':], maxlag=12)

# COMMAND ----------

# aau -> nft_gt, 주간
# 1,2,3,12 f검정 pval이 0.05미만으로 귀무가설 기각, 인과검정 통과 없음
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(totalW[['nft_gt', 'all_average_usd']]['2021':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### ㄴ nft_gt & ub(주간)

# COMMAND ----------

# nft_gt -> aub
# 1~2,7 주 가 f검정 pval이 0.05미만으로 귀무가설 기각하여 인과검정 통과
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(totalW[['all_unique_buyers', 'nft_gt']]['2021':], maxlag=12)

# COMMAND ----------

# aub -> nft_gt
# f검정 pval이 0.05초과로 귀무가설 채택하여 인과검정 불통
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(totalW[['nft_gt', 'all_unique_buyers']]['2021':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### ㄴ외부변수 인과검정 결과
# MAGIC - 월간으로 상관성 및 시차상관성은 높았음에도 인과검정 시 모두 인과성 없었음
# MAGIC - 해석이 어려웠는데, 데이터 정보 손실 문제(비율을 월간중앙값으로 가공) 또는 제 3의 요인으로 추정(커뮤니티 데이터) 
# MAGIC - 최대한 nft_gt데이터 정보를 살리기 위해 주간으로 다시 검정 결과
# MAGIC   - nft_gt -> cau : 인과영향 없음, 그나마 2가 0.06으로 가까운편
# MAGIC   - cau -> nft_gt : 1, 2 인과영향 있음
# MAGIC   - nft_gt -> gau : 1 인과영향 있음
# MAGIC   - gau -> nft_gt : 3 인과영향 있음
# MAGIC   - nft_gt -> aau : 인과영향 없음, 그나마 1가 0.09으로 가까운편
# MAGIC   - aau -> nft_gt : 1,2,3,12 인과영향 있음
# MAGIC   - nft_gt -> aub : 1,2,7 인과영향 있음
# MAGIC   - aub -> nft_gt : 인과영향 없음

# COMMAND ----------

# MAGIC %md
# MAGIC ##### <검정결과종합>
# MAGIC - [도표 문서](https://docs.google.com/presentation/d/1_XOsoLV95qqUwJI8kxFXS_7NUIQbp872UHT_cQ162Us/edit#slide=id.g122453ac673_0_0)
# MAGIC - 1) game → buyers/collectible
# MAGIC - 2) buyers → collectible
# MAGIC - 3) collectible →all
# MAGIC - 4) all → buyers 
# MAGIC - 결과적으로 다변량 시계열분석은.. 어떤 변수로 무엇을 예측해야할까?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 다변량 시계열 분석
# MAGIC - 공적분 미존재시 VAR -> 요한슨검정 -> 공적분 존재시 VECM

# COMMAND ----------

# MAGIC %md
# MAGIC ### 공적분 미존재시 VAR(벡터자기회귀모형)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (다변량)Johansen Test
# MAGIC - VAR모형에 대한 가설검정을 통해 적분계열간 안정적인 장기균형관계가 존재하는지 점검하는 방법
# MAGIC - 3개 이상의 불안정 시계열 사이의 공적분 검정에 한계를 갖는 앵글&그렌저 검정 방법을 개선하여 다변량에도 공적분 검정을 할 수 있음
# MAGIC - statsmodels.tsa.vector_ar.vecm. coint_johansen 
# MAGIC   - VECM의 공적분 순위에 대한 요한센 공적분 검정
# MAGIC   - [signature](https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html)

# COMMAND ----------

from statsmodels.tsa.vector_ar.vecm import coint_johansen

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 공적분 존재시 VECM(벡터오차수정모형)
# MAGIC - 불안정시계열X와 Y를 1차 차분한 변수를 이용하여 회귀분석을 수행함으로써 전통적 방법의 사용으로 인해 야기되는 문제점들을 어느정도 해결할 수 있으나, 두 변수 같의 장기적 관계에 대한 소중한 정보를 상실하게 된다.
# MAGIC - 이 경우 만일 두 변수 간에 공적분이 존재한다면 오차수정모형(error correction model)을 통해 변수들의 단기적 변동뿐만 아니라 장기균형관계에 대한 특성을 알 수 있게 된다.
# MAGIC - VECM은 오차수정모형(ECM)에 벡터자기회귀모형(VAR)과 같은 다인자 모형 개념을 추가 한 것
# MAGIC - [VECM 예제](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=gush14&logNo=120145414589)
# MAGIC - [파이썬 예제](http://incredible.ai/trading/2021/07/01/Pair-Trading/)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 충격반응분석

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# ## 공적분 검증기(카테고리)
# def coint_table(data, col_list, nlag):

#     xcol_list = []
#     ycol_list = []
#     pval_list = []
# #     havetomoreX = []
# #     havetomoreY = []
    
#     for i in range(len(col_list)):
#         for j in range(1, len(col_list)):
#             if col_list[i] == col_list[j]:
#                 pass
#             else:
#                 xcol_list.append(col_list[i])
#                 ycol_list.append(col_list[j])
#                 _, pval = coint_lag(data[col_list[i]], data[col_list[j]], nlag)
#                 pval_list.append(pval)
                
#                 print(col_list[i], '|', col_list[j] )
#                 print(pval)

# COMMAND ----------

# coint_table(data, avgusd_col_list, 14)

# COMMAND ----------

# # avgusd피처의 카테고리간 공적분 검증

# avgusd_col_list
# for 






# COMMAND ----------

# all카테고리 피처간 공적분 검증

# COMMAND ----------

# ## TLCC table 생성기
# def TLCC_table(data, col_list, nlag):

#     xcol_list = []
#     ycol_list = []
#     TLCC_list = []
#     havetomoreX = []
#     havetomoreY = []

#     for i in range(len(col_list)):
#         for j in range(1, len(col_list)):
#             if col_list[i] == col_list[j]:
#                 pass
#             else:
#                 xcol_list.append(col_list[i])
#                 ycol_list.append(col_list[j])
#                 tlccdata = TLCC(data[col_list[i]], data[col_list[j]], nlag)
#                 TLCC_list.append(tlccdata)
# #                 print(col_list[i], col_list[j])
# #                 print(tlccdata)
# #                 print(np.argmax(tlccdata))
# #                 print(np.argmax(TLCC_list[i]))
#                 max_TLCC_idx = np.argmax(tlccdata)
#                 max_TLCC = np.round(max(tlccdata),4)
#                 if max_TLCC >= 0.7:
#                     result = '높음'
#                 elif max_TLCC > 0.3 and max_TLCC < 0.7:
#                     result = '보통'
#                 else :
#                     result = '낮음'
#                 print(col_list[i], '|', col_list[j], '|', max_TLCC_idx, '|', max_TLCC, '|', result)
            
                
#                 if max_TLCC_idx == nlag-1:
#                     havetomoreX.append(col_list[i])
#                     havetomoreY.append(col_list[j])

#     return havetomoreX, havetomoreY
