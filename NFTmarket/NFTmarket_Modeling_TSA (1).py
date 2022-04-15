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
# MAGIC # 교차 및 시차 상관계수(Cross Correlation)
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
# MAGIC ## 예제1 : statsmodel CCF
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
# MAGIC ## 예제2 : numpy.correlate

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
# MAGIC ## ccf와 correlate 와의 차이
# MAGIC - https://stackoverflow.com/questions/24616671/numpy-and-statsmodels-give-different-values-when-calculating-correlations-how-t
# MAGIC - ccf는 np.correlate 베이스이지만, 통계적의미에서 상관관계를 위한 추가 작업을 수행함
# MAGIC - numpy가 표준편차의 곱으로 공분산을 정규화 하지 않음, 값이 너무 큼
# MAGIC - ccf는 합성곱전에 신호의 평균을 빼고 결과를 첫번째 신호의 길이로 나누어 통계에서와 같은 상관관계 정의에 도달함
# MAGIC - 통계 및 시계열 분석에서는 교차상관함수를 정규화하여 시간종속 상관계수를 얻는 것이 일반적이므로, 우리는 ccf를 사용한다.
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

col_list = feature_classifier(data, 'average_usd')

# COMMAND ----------

avgusd = data[col_list]
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
# MAGIC ### (교차)상관계수 시각화
# MAGIC - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xcorr.html
# MAGIC   - 이건 어떻게 커스텀을 못하겠다..

# COMMAND ----------

import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
plt.style.use("ggplot")

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
# MAGIC ## 함수 생성

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

## ccfcc table
def ccfcc_table(data, feature, lag):
    # 칼럼 리스트함수 호출
    col_list = feature_classifier(data, feature)
    
    # ccf 계수 함수 호출
    for col in col_list:
        ccfdata = ccf_data(data[col])
        print(col)
        print(ccfdata[:lag])
#         pd.DataFrame(ccfdata[lags], index=data[col].index, columns=col)

# COMMAND ----------

ccfcc_table(data, 'average_usd', 7)

# COMMAND ----------

# MAGIC %md
# MAGIC ## CCF-CC 교차 상관계수(Cross Correlation)
# MAGIC - avgusd 카테고리별 비교, 시가총액과 비교
# MAGIC - 변수간 동행성(comovement) 측정
# MAGIC - 경기순응적(pro-cyclical) / 경기중립적(a-cyclical) / 경기역행적(counter-cyclical)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 자기교차상관
# MAGIC - 전체카테고리별 인덱스204~366 (약6개월에서 1년주기)까지 동행성이 있음

# COMMAND ----------

 # 전체 카테고리- 자기교차상관 시각화
ccfcc_plot(data, 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 상호교차상관
# MAGIC - 카테고리가 너무 많다. 4개만 교차해서 보자 collectible_avgusd, game_avgusd, all_avgusd, all_sales_usd

# COMMAND ----------

col_list = ['collectible_average_usd', 'game_average_usd','all_average_usd', 'all_sales_usd']

# COMMAND ----------

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

# 1열은 자기교차상관, 2~3열이 상호교차상관 그래프
ccfcc_plot1(data)

# COMMAND ----------

교차 상관계수를 뽑아서 -> 피어슨을 돌려보자

# COMMAND ----------

# MAGIC %md
# MAGIC ## CCF-LC 시차 상관계수(leads and lags correlation)
# MAGIC - 경기선행적(leading) / 경기동행적(coincident) / 경기후행적(lagging)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 시각화

# COMMAND ----------



# COMMAND ----------


