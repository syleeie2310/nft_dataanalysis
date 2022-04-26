# Databricks notebook source
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # 클린 데이터 로드

# COMMAND ----------

data = pd.read_csv('/dbfs/FileStore/nft/nft_market_cleaned/total_220222_cleaned.csv', index_col = "Date", parse_dates=True, thousands=',')

# COMMAND ----------

data.info()

# COMMAND ----------

data.tail()

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
# MAGIC ### [함수] ACF/PACF 차트 생성

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go

def autoCorrelation_stack(series):
    acf_array = acf(series.dropna(), alpha=0.05) 
    pacf_array = pacf(series.dropna(), alpha=0.05)
    
    array_list = [acf_array, pacf_array]
    for i in range(len(array_list)) :
        corr_array = array_list[i]
        lower_y = corr_array[1][:,0] - corr_array[0]
        upper_y = corr_array[1][:,1] - corr_array[0]

        fig = go.Figure()
        [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f') 
         for x in range(len(corr_array[0]))]
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                       marker_size=12)
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
                fill='tonexty', line_color='rgba(255,255,255,0)')
        fig.update_traces(showlegend=False)
        fig.update_xaxes(range=[-1,42])
        fig.update_yaxes(zerolinecolor='#000000')
        
        title= 'Autocorrelation (ACF)' if i == 0 else 'Partial Autocorrelation (PACF)' 
        fig.update_layout(title=title)
        fig.show()

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def autoCorrelationF(data, feature):
    
        # 피처 분류기 호출
    col_list = feature_classifier(data, feature)
    
    for col in col_list:
        series = data[col]

        acf_array = acf(series.dropna(), alpha=0.05) 
        pacf_array = pacf(series.dropna(), alpha=0.05)

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


            fig.update_traces(showlegend=False)
            fig.update_xaxes(range=[-1,42])
            fig.update_yaxes(zerolinecolor='#000000')

        fig.update_layout(title= f'<b>[{col}] Autocorrelation (ACF)                                 [{col}] Partial Autocorrelation (PACF)<b>', 
                         title_x=0.5)
        fig.show()

# COMMAND ----------

import plotly.express as px
from plotly.subplots import make_subplots

def diff_plot(data, feature, plot):

    # 피처 분류기 호출
    col_list = feature_classifier(data, feature)
    
    diff_data = data[col_list].diff(periods=1).dropna() # dropna()는 diff를 통해 생긴 데이터 공백제거
    
    if plot == 'line':
        # 라인 차트 생성 
        for col in col_list:
#             series = data[col]
            # 데이터 차분
#             diff_series = series.diff(periods=1).dropna() 
            fig = px.line(diff_data[col], title= f'<b>[{col}] 차분 시각화<b>') 
            fig.update_layout(showlegend=False, title_x=0.5)
            fig.update_xaxes(None)
            fig.update_yaxes(None)
            fig.show()
    elif plot == 'acf':
        autoCorrelationF(diff_data, feature)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3) 통계적 가설 검정(Unit root test:단위근검정)
# MAGIC 
# MAGIC #### raw+차분과 log+차분을 정상성 테스트 해보자
# MAGIC - 검증 조건 ( p-value : 5%이내면 reject으로 대체가설 선택됨 )
# MAGIC - 귀무가설(H0): non-stationary.
# MAGIC - 대체가설 (H1): stationary.
# MAGIC - 단위근 : 단위근이란 확률론의 데이터 검정에서 쓰이는 개념으로 시계열 데이터는 시간에 따라 일정한 규칙을 가짐을 가정한다
# MAGIC 
# MAGIC #### 1. Augmented Dickey-Fuller("ADF") Test
# MAGIC - 시계열에 단위근이 존재하는지 검정,단위근이 존재하면 정상성 시계열이 아님.
# MAGIC - 귀무가설이 단위근이 존재한다.
# MAGIC - adf 작을 수록 귀무가설을 기각시킬 확률이 높다
# MAGIC #### 2. Kwiatkowski-Phillips-Schmidt-Shin (“KPSS”) Test
# MAGIC - 1종 오류를 범할 문제를 제거한 안정성 검정 방법
# MAGIC - 귀무가설이 단위근이 존재하지 않는다.

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] ADF 검정

# COMMAND ----------

# adf 검정
from statsmodels.tsa.stattools import adfuller

def adf_test(data, feature):

    # 피처 분류기 호출
    col_list = feature_classifier(data, feature)
    
    for col in col_list:
        result = adfuller(data[col].values)
        print(f'[{col}] ADF Statistics: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        print('='*50)

# COMMAND ----------

# adf 검정
from statsmodels.tsa.stattools import adfuller

def adf_test1(data):
#     print("Results of ADF Test")
    result = adfuller(data)
#     print('ADF Statistics: %f' % result[0])
#     print('p-value: %f' % result[1])
    return result
#     print('Critical values:')
#     for key, value in result[4].items():
#         print('\t%s: %.3f' % (key, value))

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] KPSS 검정

# COMMAND ----------

# KPSS 검정
from statsmodels.tsa.stattools import kpss

def kpss_test(data, feature):
    print("Results of KPSS Test:")
    
    # 피처 분류기 호출
    col_list = feature_classifier(data, feature)
    
    for col in col_list:
        result = kpss(data[col].values, regression="c", nlags="auto")
        print(f'<<{col}>>')
        kpss_output = pd.Series(
            result[0:3], index=["KPSS Statistic", "p-value", "Lags Used"] )
        for key, value in result[3].items():
            kpss_output["Critical Value (%s)" % key] = value
        print(kpss_output)
        print('='*50)

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
TLCC_plot(data, avgusd_col_list, 14)

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
def TLCC_comparison_plot1(data, X, Y, startlag, endlag): # 데이터, 기준변수, 비교변수, startlag, endlag
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
        sd_list.append(numpy.std(ydata))   # =stdev(범위)
        rsd_list.append(numpy.std(ydata)/numpy.mean(ydata)*100)  # stdev(범위)/average(범위)*100

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

# 월 중앙값 기준
print(f"<<<X기준 Y의 변동폭 및 시차상관계수 테이블>>>")
result_df = TLCC_comparison_plot1(dataM_median, 'collectible_average_usd', avgusd_col_list, -6, 6)
result_df

# COMMAND ----------

## 데이터프레임 스타일
# result_df.style.set_precision(2)
pd.set_option('display.precision', 2) # 소수점 글로벌 설정
result_df.style.background_gradient(cmap='Blues').set_caption(f"<b><<<'X(0)기준 Y의 변동폭 및 시차상관계수'>>><b>")
# df.style.applymap(lambda i: 'background-color: red' if i > 3 else '')

# COMMAND ----------

# MAGIC %md
# MAGIC #### [결론] 월 중앙값 기준 시차상관분석(collectible_avgusd 기준)
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
result_df = TLCC_comparison_plot1(dataM_median, 'all_average_usd', all_col_list, -6, 6)
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
# MAGIC - 상관관계와 유사한 공적분은 두 변수간의 비율이 평균을 중심으로 달라짐을 의미한다
# MAGIC - 공적분 관계는, 단기적으로 서로 다른 패턴을 보이지만 장기적으로 볼 때 일정한 관계가 있음을 의미함
# MAGIC - 귀무가설 : 비정상 시계열 간의 조합에 따른 오차항에 단위근이 존재한다. 즉, 서로 공적분 관계가 존재하지 않는다.
# MAGIC   -  p-value값이 5%보다 작을 때 귀무가설을 기각하여 공적분관계가 있음을 알 수 있다.
# MAGIC - 크게 2가지 검정 방법이 있는데, 대표적으로 요한슨 검정을 많이 함
# MAGIC   - engel & granget 검정 ,  johansen 검정

# COMMAND ----------

# MAGIC %md
# MAGIC ### Engle-Granger Test
# MAGIC - statsmodels.coint 는 engle-granger 기반
# MAGIC - 회귀분석 결과의 잔차항에 대해 검정
# MAGIC - N개의 비정상시계열 사이에는 일반적으로 N-1개까지의 공적분 관계가 존재할 수 있다
# MAGIC - EG 공적분 검정은 세 개 이상의 비정상시계열 사이의 공적분 검정부터 한계 가짐
# MAGIC 
# MAGIC - [앵글&그레인저 공적분 검정 예제1](https://mkjjo.github.io/finance/2019/01/25/pair_trading.html)
# MAGIC - [앵글&그레인저 공적분 검정 예제2](https://lsjsj92.tistory.com/584)

# COMMAND ----------

import statsmodels.tsa.stattools as ts

# COMMAND ----------

# 공적분 계산
X = data['game_average_usd']
Y = data['collectible_average_usd']

(Y/X).plot(figsize=(15,7))
plt.axhline((Y/X).mean(), color='red', linestyle='--')
plt.xlabel('Time')
plt.title('collectible / game Ratio')
plt.legend(['collectible / game Ratio', 'Mean'])
plt.show()

# COMMAND ----------

# raw데이터
score, pvalue, _ = ts.coint(X,Y)
print('Rawdata Correlation: ' + str(X.corr(Y)))
print('Rawdata Cointegration test p-value: ' + str(pvalue))
# # log데이터
# X = np.log1p(X)
# Y = np.log1p(Y)
# score, pvalue, _ = ts.coint(X,Y)
# print('Log data Correlation: ' + str(X.corr(Y)))
# print('Log data Cointegration test p-value: ' + str(pvalue))

# COMMAND ----------

# 게임평균가 와 콜렉터블 평균가
# 공적분의 pvalue가 0.05를 초과하여 귀무가설을 채택한다. 서로 공적분관계 없음
# 로그변환하니까 관계성이 더 낮아진다. 안보는게 맞는 듯

# COMMAND ----------

# 공적분 계산
X = data['all_average_usd']
Y = data['all_sales_usd']
(Y/X).plot(figsize=(15,7))
plt.axhline((Y/X).mean(), color='red', linestyle='--')
plt.xlabel('Time')
plt.title('all_sales_usd / all_avgusd  Ratio')
plt.legend(['all_sales_usd / all_avgusd  Ratio', 'Mean'])
plt.show()

# COMMAND ----------

# raw데이터
score, pvalue, _ = ts.coint(X,Y)
print('Rawdata Correlation: ' + str(X.corr(Y)))
print('Rawdata Cointegration test p-value: ' + str(pvalue))
# # log데이터
# X = np.log1p(X)
# Y = np.log1p(Y)
# score, pvalue, _ = ts.coint(X,Y)
# print('Log data Correlation: ' + str(X.corr(Y)))
# print('Log data Cointegration test p-value: ' + str(pvalue))

# COMMAND ----------

# 전체평균가 와 전체판매가
# 공적분의 pvalue가 0.05를 초과하여 귀무가설을 채택한다. 서로 공적분관계 없음

# COMMAND ----------

# 공적분 계산
X = data['all_average_usd']
Y = data['all_number_of_sales']
(Y/X).plot(figsize=(15,7))
plt.axhline((Y/X).mean(), color='red', linestyle='--')
plt.xlabel('Time')
plt.title('all_sales/ all_avgusd Ratio')
plt.legend(['all_sales / all_avgusd Ratio', 'Mean'])
plt.show()

# COMMAND ----------

# raw데이터
score, pvalue, _ = ts.coint(X,Y)
print('Rawdata Correlation: ' + str(X.corr(Y)))
print('Rawdata Cointegration test p-value: ' + str(pvalue))
# # log데이터
# X = np.log1p(X)
# Y = np.log1p(Y)
# score, pvalue, _ = ts.coint(X,Y)
# print('Log data Correlation: ' + str(X.corr(Y)))
# print('Log data Cointegration test p-value: ' + str(pvalue))

# COMMAND ----------

# raw데이터
score, pvalue, _ = ts.coint(X,Y, trend='ct')
print('Rawdata Correlation: ' + str(X.corr(Y)))
print('Rawdata Cointegration test p-value: ' + str(pvalue))
# log데이터
X = np.log1p(X)
Y = np.log1p(Y)
score, pvalue, _ = ts.coint(X,Y)
print('Log data Correlation: ' + str(X.corr(Y)))
print('Log data Cointegration test p-value: ' + str(pvalue))

# COMMAND ----------

# 전체평균가 와 전체판매수
# 공적분의 pvalue가 0.05를 초과하여 귀무가설을 채택한다. 서로 공적분관계 없음

# COMMAND ----------

# MAGIC %md
# MAGIC #### (pass)lag별 공적분 검증
# MAGIC -> 본래 lag별로 공적분 검증을 하지는 않음, 공적분 여부에 따라 다변량분석 기법이 달라지기 때문
# MAGIC - all카테고리 대표 : avgusd-buyer 71, 공적분 검증 성공(min=94)
# MAGIC   - 종합 : 전체 평균가와 전체 구매자수는 장기관점에서 인과성이 있다. 평균가가 71~94일 정도 선행한다.
# MAGIC - avgusd피처 대표 : collectible-game 59, 공적분 검증 성공(min=66), 
# MAGIC   - 종합 : 콜렉터블 평균가와 게임 평균가는 장기적으로 인과성이 있다. 콜렉터들이 2달 정도 선행한다.

# COMMAND ----------

#  시차상관계수 계산함수
def coint_lag(X, Y, nlag):
#     score=[]
    pvalue=[]
    for i in range(nlag):
        _, p, _ = ts.coint(X,Y.shift(i, fill_value=0))
#         score.append(s)
        pvalue.append(p)
    return pvalue

# COMMAND ----------

X = data['collectible_average_usd']
Y = data['game_average_usd']
pval = coint_lag(X,Y, 100)
print(np.argmin(pval), '|', min(pval))
print(pval)

# COMMAND ----------

# 40 이후 pval 0.05 미만으로 귀무가설 기각하여 공적분 검증완료, 장기 관계 있음
plt.figure(figsize=(30,10))
plt.plot(range(len(pval)), pval)
plt.hlines(0.05, xmin=0, xmax=len(pval), color='blue')
plt.show()

# COMMAND ----------

X = data['all_average_usd']
Y = data['all_unique_buyers']
# print(Y.shift(i, fill_value=0))
pval = coint_lag(X,Y, 100)
print(np.argmin(pval), '|', min(pval))
print(pval)
# s, p, _ = ts.coint(X,Y.shift(i))

# COMMAND ----------

# 20전후, 75 이후가 pval 0.05 미만으로 귀무가설 기각하여 공적분 검증완료, 장기 관계 있음
# 
plt.figure(figsize=(30,10))
plt.plot(range(len(pval)), pval)
plt.hlines(0.05, xmin=0, xmax=len(pval), color='blue')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Johansen Test
# MAGIC - 벡터 형태로 검정, EG 공적분 검정의 한계 없음

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 그레인저 인과검정(Granger Causality)
# MAGIC - [클라이브 그레인저 위키](https://ko.wikipedia.org/wiki/%ED%81%B4%EB%9D%BC%EC%9D%B4%EB%B8%8C_%EA%B7%B8%EB%A0%88%EC%9D%B8%EC%A0%80)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 다변량 시계열 분석
# MAGIC -  공적분 미존재시 VAR
# MAGIC - 공적분 존재시 VECM

# COMMAND ----------

# MAGIC %md
# MAGIC ### 공적분 미존재시 VAR

# COMMAND ----------

# MAGIC %md
# MAGIC ### 공적분 존재시 VECM
# MAGIC  
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
