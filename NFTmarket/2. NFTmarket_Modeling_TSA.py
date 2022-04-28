# Databricks notebook source
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # 정제 데이터 로드

# COMMAND ----------

data = pd.read_csv('/dbfs/FileStore/nft/nft_market_cleaned/total_220222_cleaned.csv', index_col = "Date", parse_dates=True, thousands=',')

# COMMAND ----------

data.info()

# COMMAND ----------

data.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC # 시계열 특성 분석

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 정상성 판단
# MAGIC ### 자기 상관 함수(ACF)
# MAGIC - 잔차들이 시간의 흐름에서 독립적인지를 확인하기 위함(acf에 0에 가까우면 독립적)
# MAGIC - 시차가 클수록 0에 가까워지며, 정상 시계열은 상대적으로 빠르게 0에 수렴한다. 
# MAGIC - ACF는 보통 시계열에서 과거의 종속변수(Y)와의 비교를 통해 계절성 판단을 주로 한다.
# MAGIC - 보통 시계열 분석에서 많이 사용이 되며, 현재의 Y값과 과거의 Y값의 상관성을 비교한다. 왜냐하면, 각각의 Y값이 독립적이어야 결과 분석이 더 잘되기 때문이다.(Y를 정상화시키면 분석이 더 잘된다는 개념과 같다.) 
# MAGIC 
# MAGIC ### 편자기 상관 함수(PACF)
# MAGIC - 시차에 따른 일련의 편자기상관이며, 시차가 다른 두 시계열 데이터간의 순수한 상호 연관성
# MAGIC 
# MAGIC ### 그래프 해석
# MAGIC - AR(p) 특성: ACF는 천천히 감소하고, PACF는 처음 시차를 제외하고 급격히 감소
# MAGIC - MA(q) 특성: ACF가 급격히 감소하고, ACF는 천천히 감소
# MAGIC - 각각 급격히 감소하는 시차를 모수로 사용한다. AR -> PACF,    MA -> ACF
# MAGIC <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpcuWC%2Fbtq5CACTt5C%2FX3UFPPkwhZpjV59WygsV30%2Fimg.png">

# COMMAND ----------

from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
plt.style.use("ggplot")

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

# MAGIC %md
# MAGIC ### raw 데이터 시각화
# MAGIC - 평균이 일정하지 않음, 대체로 MA특징을 가짐 (PACF), 차분 필요
# MAGIC - 2개의 경향으로 나눠짐
# MAGIC   - all, collectible, art, metaverse 
# MAGIC   - defi, game, utility

# COMMAND ----------

autoCorrelationF(data, 'average_usd') #raw df, feature

# COMMAND ----------

# MAGIC %md
# MAGIC ### log변환 데이터 시각화
# MAGIC - raw데이터와 유사함

# COMMAND ----------

autoCorrelationF(np.log1p(data), 'average_usd') #raw df, feature

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) 차분
# MAGIC - 대체로 정상성을 보여 1차로 충분해보임, 그러나 특정 변동폭이 너무 커서 전체적인 패턴을 여전히 보기 어려움
# MAGIC - 큰 변동폭이 있음 미 all, collectible, art, metaverse 
# MAGIC - all은 21.11.15 에 큰 변동폭
# MAGIC - collectible, art, metaverse 가 22.1.8에 큰 변동이 있음
# MAGIC - defi, game, utility는 모두 다름

# COMMAND ----------

# MAGIC %md
# MAGIC #### raw데이터+차분
# MAGIC - 차분을 통해 정상성을 갖는다.

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

diff_plot(data, 'average_usd', 'line') #raw df, feature

# COMMAND ----------

diff_plot(data, 'average_usd', 'acf') #raw df, feature

# COMMAND ----------

# MAGIC %md
# MAGIC #### log변환+차분
# MAGIC - 여전히 갭이 커서 보기 어렵다. 적절한지 모르겠다.

# COMMAND ----------

diff_plot(np.log1p(data), 'average_usd', 'line') #raw df, feature

# COMMAND ----------

# 첫번째는 자기자신과의 상관관계이므로 1이 나올수밖에 없다.
diff_plot(np.log1p(data), 'average_usd', 'acf') #raw df, feature

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
# MAGIC - KPSS 검정은 1종 오류의 발생가능성을 제거한 단위근 검정 방법이다.
# MAGIC - DF 검정, ADF 검정과 PP 검정의 귀무가설은 단위근이 존재한다는 것이나, KPSS 검정의 귀무가설은 정상 과정 (stationary process)으로 검정 결과의 해석 시 유의할 필요가 있다.
# MAGIC   - 귀무가설이 단위근이 존재하지 않는다.
# MAGIC - 단위근 검정과 정상성 검정을 모두 수행함으로서 정상 시계열, 단위근 시계열, 또 확실히 식별하기 어려운 시계열을 구분하였다.
# MAGIC - KPSS 검정은 단위근의 부재가 정상성 여부에 대한 근거가 되지 못하며 대립가설이 채택되면 그 시계열은 trend-stationarity(추세를 제거하면 정상성이 되는 시계열)을 가진다고 할 수 있습니다.
# MAGIC - 때문에 KPSS 검정은 단위근을 가지지 않고 Trend- stationary인 시계열은 비정상 시계열이라고 판단할 수 있습니다.

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

# KPSS 검정
from statsmodels.tsa.stattools import kpss

def kpss_test1(data):
#     print("Results of KPSS Test")
    result = kpss(data, regression="c", nlags="auto")
    kpss_output = pd.Series(
        result[0:3], index=["KPSS Statistic", "p-value", "Lags Used"] )
#     for key, value in result[3].items():
#         kpss_output["Critical Value (%s)" % key] = value
#     print(kpss_output[:1])   
    
#     print('KPSS Statistics: %f' % kpss_output[0])
#     print('p-value: %f' % kpss_output[1])
    return kpss_output


# COMMAND ----------

# MAGIC %md 
# MAGIC #### [함수] 단위근검정 실행기

# COMMAND ----------

pd.options.display.float_format = '{: .4f}'.format

def URT(data, feature) :
    # 피처 분류기 호출
    col_list = feature_classifier(data, feature)
    
    adf_stats = []
    adf_Pval = []
    kpss_stats = []
    kpss_Pval = []
    total_list = []
    
    for col in col_list:
#         print(f'<<<<{col}>>>>')
        col_data = data[col]
        
        # ADF검정기 호출
        adf_result = adf_test1(col_data) 
        adf_stats.append(adf_result[0])
        adf_Pval.append(adf_result[1])
        
        # KPSS검정기 호출
        kpss_result = kpss_test1(col_data)
        kpss_stats.append(kpss_result[0])
        kpss_Pval.append(kpss_result[1])
        
        # 종합
        if adf_result[1] <= 0.05 and kpss_result[1] >= 0.05:
            total_list.append('ALL Pass')
        elif adf_result[1] <= 0.05 or kpss_result[1] >= 0.05:
            total_list.append('One Pass')
        else :
            total_list.append('fail')
        
    # 테이블 생성
#     col_list.append('total')
    result_df = pd.DataFrame(list(zip(adf_stats, adf_Pval, kpss_stats, kpss_Pval, total_list)), index = col_list, columns=['adf_stats', 'adf_Pval', 'KPSS_stats', 'KPSS_Pval', 'total'])
    
#     # adf stats가 낮은 순으로 정렬
#     result_df.sort_values(sort, inplace=True)
    
    return result_df             

# COMMAND ----------

# MAGIC %md
# MAGIC #### Raw+차분 검정(ADF, KPSS)

# COMMAND ----------

# 전체 기간 : art제외하고 모두 정상성을 가짐
URT(data.diff(periods=1).dropna(), 'average_usd')

# COMMAND ----------

# 2018년 이후 :
URT(data['2018':].diff(periods=1).dropna(), 'average_usd')

# COMMAND ----------

# 2018년 ~ 2021년 : all, defi, utility만 통과
URT(data['2018':'2021'].diff(periods=1).dropna(), 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Log+차분 검정(ADF, KPSS)

# COMMAND ----------

# 전체 기간 : utility는 조금 약함, 
URT(np.log1p(data).diff(periods=1).dropna(), 'average_usd')

# COMMAND ----------

# 전체기간 : art와 defi만 모두 통과
URT(np.log1p(data['2018':]).diff(periods=1).dropna(), 'average_usd')

# COMMAND ----------

# 2018~2021 : art와 defi만 모두 통과
URT(np.log1p(data['2018':'2021']).diff(periods=1).dropna(), 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4) [종합요약] "average_usd"피처의 카테고리별 정상성 분석
# MAGIC 
# MAGIC - 차분은 1회면 충분하다. MA값은 raw는 1, log는 0으로 확인됨, (P=?, D=1, Q=1)
# MAGIC   - acf/pacf 그래프에서  p와 q값을 선정하는 것은 권장하지 않음, 정확하지 않고 해석하기 어려움
# MAGIC   - 전체 행 길이의 log변환 값을 최대치로, ar을 실험하는 가이드가 있으나 정확하지 않음, 값이 변하지 않는지 더 체크해봐야함
# MAGIC - 통계적 가설 검정
# MAGIC   - 카테고리별, raw/log별, 기간별 결과가 모두 달라서 혼란스럽다..
# MAGIC   - raw+차분와 log+차분, 중에 무엇을 골라야하나?
# MAGIC   - 카테고리는 어떻게 골라야 하나?

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 시계열 분해
# MAGIC - 시계열 성분 : 추세, 계절/순환, 불규칙(나머지)
# MAGIC - statsmodels.tsa.seasonal.STL : LOESS를 사용한 계절 추세 분해
# MAGIC - statsmodels.tsa.seasonal.seasonal_decompose : 가산 또는 곱셈 모델과 같은 선형 모델
# MAGIC   - (1) 시도표 (time series plot)를 보고 시계열의 주기적 반복/계절성이 있는지, 가법 모형(additive model, y = t + s + r)과 승법 모형(multiplicative model, y = t * s * r) 중 무엇이 더 적합할지 판단을 합니다. 
# MAGIC 
# MAGIC  
# MAGIC 
# MAGIC <가법 모형을 가정 시>
# MAGIC 
# MAGIC   - (2) 시계열 자료에서 추세(trend)를 뽑아내기 위해서 중심 이동 평균(centered moving average)을 이용합니다. 
# MAGIC 
# MAGIC  
# MAGIC 
# MAGIC   - (3) 원 자료에서 추세 분해값을 빼줍니다(detrend). 그러면 계절 요인과 불규칙 요인만 남게 됩니다. 
# MAGIC 
# MAGIC  
# MAGIC 
# MAGIC   - (4) 다음에 계절 주기 (seasonal period) 로 detrend 이후 남은 값의 합을 나누어주면 계절 평균(average seasonality)을 구할 수 있습니다. (예: 01월 계절 평균 = (2020-01 + 2021-01 + 2022-01 + 2023-01)/4, 02월 계절 평균 = (2020-02 + 2021-02 + 2022-02 + 2023-02)/4). 
# MAGIC 
# MAGIC  
# MAGIC 
# MAGIC   - (5) 원래의 값에서 추세와 계절성 분해값을 빼주면 불규칙 요인(random, irregular factor)이 남게 됩니다. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] 시각화

# COMMAND ----------

from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose

def plot_seasonal_decompose(result:DecomposeResult, dates:pd.Series=None, title:str="Seasonal Decomposition"):
    x_values = dates if dates is not None else np.arange(len(result.observed))
    return (
        make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.observed, mode="lines", name='Observed'),
            row=1,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.trend, mode="lines", name='Trend'),
            row=2,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.seasonal, mode="lines", name='Seasonal'),
            row=3,
            col=1,
        )
        .add_trace(
            go.Scatter(x=x_values, y=result.resid, mode="lines", name='Residual'),
            row=4,
            col=1,
        )
        .update_layout(
            height=900, title=f'<b>{title}</b>', margin={'t':100}, title_x=0.5, showlegend=False
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### 실험1 (미차분)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### raw 데이터

# COMMAND ----------

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
df = data['collectible_average_usd']
decomposition = seasonal_decompose(df, model='additive', period=365) 
# 일자데이터... 기간 어케함 ㅜ, 자동 달력변동이 안되고 덧셈분해만 가능
fig = plot_seasonal_decompose(decomposition, dates=df.index)
fig.show()

# COMMAND ----------

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
df = data['game_average_usd']
decomposition = seasonal_decompose(df, model='additive', period=365) 
# 일자데이터... 기간 어케함 ㅜ, 자동 달력변동이 안되고 덧셈분해만 가능
fig = plot_seasonal_decompose(decomposition, dates=df.index)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### log변환 데이터

# COMMAND ----------

df = np.log1p(data['collectible_average_usd'])
decomposition = seasonal_decompose(df, model='additive', period=365) # 일자데이터...
fig = plot_seasonal_decompose(decomposition, dates=df.index)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 집계 데이터

# COMMAND ----------

from statsmodels.tsa.seasonal import seasonal_decompose
dataM_median = data.resample('M').median() # 월 중앙값 데이터 생성
df = dataM_median['collectible_average_usd']

decomposition = seasonal_decompose(df, model='additive', period=12) # 일자데이터...
fig = plot_seasonal_decompose(decomposition, dates=df.index)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 실험2 (1차분)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### raw+차분

# COMMAND ----------

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
df = data['collectible_average_usd'].diff(periods=1).dropna()
decomposition = seasonal_decompose(df, model='additive', period=365) 
fig = plot_seasonal_decompose(decomposition, dates=df.index)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### log+차분

# COMMAND ----------

# 차분먼저하고 로그변환하면 오류남..
df = np.log1p(data['collectible_average_usd']).diff(periods=1).dropna()
decomposition = seasonal_decompose(df, model='additive', period=365) 
fig = plot_seasonal_decompose(decomposition, dates=df.index)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 집계+차분

# COMMAND ----------

# 차분-> 집계 와, 집계->차분의 그래프가 다름, 무엇이 정확할까?
from statsmodels.tsa.seasonal import seasonal_decompose
dataM_median = (data.resample('M').median()).diff(periods=1).dropna() 
# dataM_median = (data.diff(periods=1).dropna()).resample('M').median() 
df = dataM_median['collectible_average_usd']

decomposition = seasonal_decompose(df, model='additive', period=12) # 일자데이터...
fig = plot_seasonal_decompose(decomposition, dates=df.index)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [종합요약] : raw, log, 집계, 차분까지 모두 계절성을 보이고 불규칙에서도 패턴이 있다. SARIMA를 써보자.  다만 21년도 이후부터는 계절성이 없어 예측 우려
# MAGIC - collectible_average_usd는 계절성이 있다.  불규칙에서도 패턴을 보인다. -> 예측이 가능할 것 같지만
# MAGIC - [실험1] 20년까지 계절성와 불규칙(반복) 특징이 있음. 21년부터 업어 예측될지 의문, -> 지수평활을 해야할 것 같은데.. 
# MAGIC   - raw추세 : 18년 하락, 21년 급상승
# MAGIC   - raw계절성 : 1년 주기로 7월에급상승하고 이후 하락세
# MAGIC   - raw불규칙 : 20년 중반까지 1년간 상승하다 8월 하락 특징이 있었으나,  21년부터 하락 지속
# MAGIC   - log는 계절성이 18년 1월부터 뜀, 로그변환으로 관측값과 왜곡이 생겨 부적합
# MAGIC - [실험2] 해석 어려움, 유의미한지?
# MAGIC    -차분+집계는 실험1과 유사함 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Seasonal adjustment
# MAGIC - 계절성이 있긴하지만, 추세에 크게 영향을 주진 못함, 영향력이 작은 듯함, 트랜드 영향이 더 큰 데이터임. 최근 트랜드에 대한 가중치 고려가 필요함(ex 지수평활법)
# MAGIC - 계절성으로 조정된 데이터, (원데이터에 계절성을 뺌)
# MAGIC - 계절성으로 조정된 시계열에는 추세-주기 성분도 있고 나머지 성분도 있습니다.
# MAGIC - 그래서, 시계열이 “매끄럽지” 않고, “하락세”나 “상승세”라는 표현이 오해를 불러 일으킬 수 있습니다.
# MAGIC - 시계열에서 전환점을 살펴보는 것과 어떤 방향으로의 변화를 해석하려는 것이 목적이라면, 계절성으로 조정된 데이터보다는 추세-주기 성분을 사용하는 것이 더 낫습니다.

# COMMAND ----------

# 미차분 raw 데이터 계절성 조정
df = data['collectible_average_usd']
decomposition = seasonal_decompose(df, model='additive', period=365) 
# decomposition_trend = decomposition.trend
decomposition_seasonal = decomposition.seasonal
df_adjusted = (df - decomposition_seasonal).rename('seasonal adjusted')
df_adjusted

# COMMAND ----------

# 음수가 있네..;
df_adjusted.describe()

# COMMAND ----------

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = go.Figure([
    # 원 데이터-------------------------------------------------------
    go.Scatter(x = df.index, y = df, name = "raw", mode = 'lines')
    # 계절성 조정 데이터------------------------------------------------------
    , go.Scatter(x = df_adjusted.index, y = df_adjusted, name = "adjusted", mode = 'lines')
])
fig.update_layout(title = '<b>[collectible_average_usd] 계절성 조정 비교<b>', title_x=0.5, legend=dict(orientation="h", xanchor="right", x=1, y=1.1))
fig.update_yaxes(ticklabelposition="inside top", title=None)
fig.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 상호상관분석(Cross Correlation)

# COMMAND ----------



# COMMAND ----------


