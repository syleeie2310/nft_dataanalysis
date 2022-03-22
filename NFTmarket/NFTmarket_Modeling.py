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

data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # 8. 시계열 특성 분석

# COMMAND ----------

# MAGIC %md
# MAGIC ## 정상성 판단
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
# MAGIC - 평균이 일정하지 않음, 차분 필요
# MAGIC - 2개의 경향으로 나눠짐
# MAGIC   - all, collectible, art, metaverse 
# MAGIC   - defi, game, utility

# COMMAND ----------

autoCorrelationF(data, 'average_usd') #raw df, feature

# COMMAND ----------

# MAGIC %md
# MAGIC #### [종합요약] "average_usd"피처, 카테고리별 자기상관계수
# MAGIC 
# MAGIC | 카테고리 | ACF | PACF |
# MAGIC |:-------:|:----:|:----:|
# MAGIC |:all:|::|::|
# MAGIC |:collectible:|::|::|
# MAGIC |:art:|::|::|
# MAGIC |:metaverse:|::|::|
# MAGIC |:game:|::|::|
# MAGIC |:defi:|::|::|

# COMMAND ----------

# MAGIC %md
# MAGIC ### 차분

# COMMAND ----------

import plotly.express as px
from plotly.subplots import make_subplots

def diff_line(data, feature):

    # 피처 분류기 호출
    col_list = feature_classifier(data, feature)
    
    for col in col_list:
        series = data[col]
        # 데이터 차분
        diff_series = series.diff(periods=1).dropna() # dropna()는 diff를 통해 생긴 데이터 공백제거
        fig = px.line(diff_series, title= f'[{col}] 차분 시각화') 
#         fig = make_subplots(rows=1, cols=2)
        fig.show()

# COMMAND ----------

diff_line(data, 'average_usd') #raw df, feature

# COMMAND ----------

# 21년 11월 15~16일 급등락이 매우 큼, 그외에는 정상성을 보이므로 1차 차분으로 충분
# arima(0,1,2)하면 될듯
 
    

# COMMAND ----------

autoCorrelation_stack(diff_series)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 시계열 검증(adfuller)
# MAGIC - 검증 조건 ( p-value : 5%이내면 reject으로 대체가설 선택됨 )
# MAGIC - 귀무가설(H0): non-stationary.
# MAGIC - 대체가설 (H1): stationary.

# COMMAND ----------

from statsmodels.tsa.stattools import adfuller

# COMMAND ----------

# raw데이터 : p-value가 0.05보다 크므로 귀무가설 채택하여 "비정상성"
print(adfuller(series)[0]) # adf 작을 수록 귀무가설을 기각시킬 확률이 높다
print(adfuller(series)[1]) # p-value

# COMMAND ----------

# raw데이터 : p-value가 0.05보다 작으므로 귀무가설 기각하여 "정상성"
print(adfuller(diff_series)[0]) # adf 작을 수록 귀무가설을 기각시킬 확률이 높다
print(adfuller(diff_series)[1]) # p-value

# COMMAND ----------

# MAGIC %md
# MAGIC # 모델링

# COMMAND ----------

# MAGIC %md
# MAGIC ## ARIMA

# COMMAND ----------

order = (0, 1, 2)
model = ARIMA(train, order)
rfit = model.fit()
rfit.summary()

# COMMAND ----------

# (0,1,1)이 정말 최선일지, 모든 경우의 수의 aic값을 체크해보자
def arima_aic_check(data, order,sort = 'AIC'):
    order_list = []
    aic_list = []
    bic_lsit = []
    for p in range(order[0]):
        for d in range(order[1]):
            for q in range(order[2]):
                model = ARIMA(data, order=(p,d,q))
                try:
                    model_fit = model.fit()
                    c_order = f'p:{p} d:{d} q:{q}'
                    aic = model_fit.aic
                    bic = model_fit.bic
                    order_list.append(c_order)
                    aic_list.append(aic)
                    bic_list.append(bic)
                except:
                    pass
    result_df = pd.DataFrame(list(zip(order_list, aic_list)),columns=['order','AIC'])
    result_df.sort_values(sort, inplace=True)
    return result_df

# COMMAND ----------

# aic값이 가장 작은 모델을 선택, 12번 (1,1,2)의 aic값이 가장 작다
arima_aic_check(train, [3,3,3])

# COMMAND ----------

# P>z값이 일반적으로 학습의 적정성을 위해 호가인되는 T-검정값
# 즉, p-value 0.05수준에서 보면 MA(1)와 MA(2)의 값은 유효한데, 모형의 constant는 유효하지 않다. 따라서 모형의 model.fit()파라미터중trend='c'가 아니라 nc로 설정하는 것이 옳다
order = (1, 1, 2)
model = ARIMA(train, order)
rfit = model.fit()
rfit.summary()

# COMMAND ----------

# constraint가 없는 모형으로 fitting하니 t 검정값이 더 좋아짐?
order = (1, 1, 2)
model = ARIMA(train, order)
model_fit = model.fit(trend='nc')
model_fit.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ARIMA-예측

# COMMAND ----------

model_fit.plot_predict()

# COMMAND ----------

# model_fit.plot_predict('2017-06-23', '2022-02-28')
model_fit.plot_predict(1, 1720)

# COMMAND ----------

fore = model_fit.forecast(steps=1)
print(fore)
# 예측값, stderr, upperbound, lower bound
# 2월 1일을 573.9라고 예측, 비슷한 듯

# COMMAND ----------

series['2022-02-01']

# COMMAND ----------

# 과거 데이터로 테스트해보자
model_fit.predict(1, 10, typ='levels') # typ= default값이 linear, 예측할때 levels

# COMMAND ----------

# ... 뭥미?
train[1:10]

# COMMAND ----------

# 미래 예측하기 # 왜 안되지 ㅜㅜ
preds = model_fit.predict('2022-02-01', '2022-02-28', typ='levels')
preds

# COMMAND ----------

preds = model_fit.predict(1,1720, typ='levels')
preds

# COMMAND ----------

# MAGIC %md
# MAGIC ### 검증
# MAGIC - 예측값의 잔차 ACF를 그려 정상성을 체크한다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prophet
# MAGIC - https://facebook.github.io/prophet/docs/quick_start.html#python-api

# COMMAND ----------

# MAGIC %md
# MAGIC ### quick start

# COMMAND ----------

# !pip install Prophet

# COMMAND ----------

# py4j 로깅 숨기기
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

import pandas as pd
from prophet import Prophet

# COMMAND ----------

# 칼럼명 변경
df = series.reset_index()
df.columns = ['ds', 'y']
print(df)

# COMMAND ----------

m = Prophet()
m.fit(df)

# COMMAND ----------

future = m.make_future_dataframe(periods=14)
future.tail()

# COMMAND ----------

# yhat : 응답변수의 추정값
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# COMMAND ----------

from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)

# COMMAND ----------

1# Python
plot_components_plotly(m, forecast)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Forecasting Growth

# COMMAND ----------

df['cap'] = 8.5

# COMMAND ----------

mlg = Prophet(growth='logistic')
mln = Prophet(growth='linear')
mlg.fit(df)
mln.fit(df)


# COMMAND ----------

# 선형 예측
future = mln.make_future_dataframe(periods=365)
future['cap'] = 8.5
fcst = mln.predict(future)
fig = mln.plot(fcst)

# COMMAND ----------

# 로지스틱 예측
future = mlg.make_future_dataframe(periods=365)
future['cap'] = 8.5
fcst = mlg.predict(future)
fig = mlg.plot(fcst)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saturating Minimum

# COMMAND ----------

df['y'] = 10 - df['y']
df['cap'] = 6
df['floor'] = 1.5
future['cap'] = 6
future['floor'] = 1.5
m = Prophet(growth='logistic')
m.fit(df)
fcst = m.predict(future)
fig = m.plot(fcst)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trend changepoints

# COMMAND ----------

from prophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adjusting trend flexibility

# COMMAND ----------

m = Prophet(changepoint_prior_scale=0.5)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)

# COMMAND ----------

# 값을 줄여보자
m = Prophet(changepoint_prior_scale=0.001)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)

# COMMAND ----------

# 변경점 직접 지정
# Python
m = Prophet(changepoints=['2020-11-01'])
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)

# COMMAND ----------


