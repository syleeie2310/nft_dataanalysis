# Databricks notebook source
import numpy as np
import pandas as pd

# COMMAND ----------

data = pd.read_csv('/dbfs/FileStore/nft/nft_market_cleaned/total_220222_cleaned.csv', index_col = "Date", parse_dates=True, thousands=',')

# COMMAND ----------

data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # 모델링

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. 데이터 분리
# MAGIC - collectible_average_usd 피처를 대표로 진행해보자

# COMMAND ----------

# raw 데이터
train = data.loc[:'2021', 'collectible_average_usd']
test = data.loc['2022':, 'collectible_average_usd']
print(len(train), train.tail())
print(len(test), test.head())

# COMMAND ----------

import plotly.express as px
fig = px.line()
fig.add_scatter(x=train.index, y = train, mode="lines", name = "train")
fig.add_scatter(x=test.index, y = test, mode="lines", name = "test")
fig.update_layout(title = '<b>[collectible_average_usd] Raw data <b>', title_x=0.5, legend=dict(orientation="h", xanchor="right", x=1, y=1.1))
fig.update_yaxes(ticklabelposition="inside top", title=None)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prophet
# MAGIC - https://facebook.github.io/prophet/docs/quick_start.html#python-api

# COMMAND ----------

# MAGIC %md
# MAGIC ### quick start

# COMMAND ----------

# !pip install Prophet

# COMMAND ----------

# py4j 로깅 숨기기
from prophet import Prophet
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

# 칼럼명 변경
df = train.reset_index()
df.columns = ['ds', 'y']
print(df)

# COMMAND ----------

m = Prophet(growth='linear') # linear
m.fit(df)

# COMMAND ----------

# 예측 범위 (인덱스) 만들기
future = m.make_future_dataframe(periods=60)
future.tail()

# COMMAND ----------

# 예측하기 
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()#yhat : 응답변수의 추정값

# COMMAND ----------

from prophet.plot import plot_plotly, plot_components_plotly
plot_plotly(m, forecast, figsize=(1600,700))

# COMMAND ----------

# 시계열 분해
# trend : 21년부터 급등
# seasonal : 매년 8월부터 상승하고 1월에 하락, 토요일 상승
plot_components_plotly(m, forecast, figsize=(1600,300))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1) 포화 예측

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 성장 예측
# MAGIC - 성장예측시 일반적으로 달성할 수 있는 최대지점(규모)인 "수용능력"을 설정하며, 예측은 이 지점에서 포화되어야 한다.

# COMMAND ----------

# 시장 규모 설정
df['cap'] = 2000

# COMMAND ----------

# 선형 모형
mln = Prophet(growth='linear')
mln.fit(df)

# COMMAND ----------

future = mln.make_future_dataframe(periods=20)
future['cap'] = 2000
fcst = mln.predict(future)
# plot_plotly(mln, fcst, figsize=(1600,700))
fig = mln.plot(fcst, figsize=(20,6), xlabel='Date', ylabel='Value')
ax = fig.gca()
ax.set_title("[Collectible_average_usd] Linear Forecast", size=34)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 포화 최소
# MAGIC - 로지스틱 모델은 포화 최소값도 설정할 수 있음 floor 지정

# COMMAND ----------

# 로지스틱 모형 생성
mlg = Prophet(growth='logistic')
mlg.fit(df)

# COMMAND ----------

future = mlg.make_future_dataframe(periods=20)
future['cap'] = 2000 
future['floor'] = 0 # floor 디폴트값
fcst = mlg.predict(future)
# plot_plotly(mlg, fcst, figsize=(1600,700))
fig = mlg.plot(fcst, figsize=(20,6), xlabel='Date', ylabel='Value')
ax = fig.gca()
ax.set_title("[Collectible_average_usd] Logistic Forecast")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) 트렌드 변화점

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 자동 변화점 감지
# MAGIC - 최대 25개의 잠재 포인트를 찾음

# COMMAND ----------

from prophet.plot import add_changepoints_to_plot
fig = m.plot(forecast, figsize=(20,6), xlabel='Date', ylabel='Value')
a = add_changepoints_to_plot(fig.gca(), mln, forecast)
ax = fig.gca()
ax.set_title("[Collectible_average_usd] Linear Forecast", size=34)

# COMMAND ----------

from prophet.plot import add_changepoints_to_plot
fig = m.plot(forecast, figsize=(20,6), xlabel='Date', ylabel='Value')
a = add_changepoints_to_plot(fig.gca(), mlg, forecast)
ax = fig.gca()
ax.set_title("[Collectible_average_usd] Logistic Forecast", size=34)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 추세 유연성 조정
# MAGIC - 기본값 0.05
# MAGIC - 높이면 추세가 더 유연해진다.
# MAGIC - 추세가 과대 또는 과소적합해 보이는 경우에 필요에 따라 매개변수를 조정한다.
# MAGIC - 이 데이터는 추세를 빼면 될듯(추세 영향 거의 없음)

# COMMAND ----------

m_cp = Prophet(growth='linear', changepoint_prior_scale=0.5)
forecast = m_cp.fit(df).predict(future)
fig = m_cp.plot(forecast, figsize=(20,6), xlabel='Date', ylabel='Value')
ax = fig.gca()
ax.set_title("[Collectible_average_usd] Linear Forecast - CP 0.05(default) ", size=34)

# COMMAND ----------

m_cp05 = Prophet(growth='linear', changepoint_prior_scale=0.5)
forecast = m_cp05.fit(df).predict(future)
fig = m_cp05.plot(forecast, figsize=(20,6), xlabel='Date', ylabel='Value')
ax = fig.gca()
ax.set_title("[Collectible_average_usd] Linear Forecast - CP 0.5 ", size=34)

# COMMAND ----------

# 값을 줄여보자, 오히려 줄이니까 이상함. 불규칙 패턴 영향으로 보임
m_cp0001 = Prophet(growth='linear', changepoint_prior_scale=0.001)
forecast = m_cp0001.fit(df).predict(future)
fig = m_cp0001.plot(forecast, figsize=(20,6), xlabel='Date', ylabel='Value')
ax = fig.gca()
ax.set_title("[Collectible_average_usd] Linear Forecast - CP 0.001 ", size=34)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 변경점 수동 지정

# COMMAND ----------

# 변경점 직접 지정
# Python
m_cp191001 = Prophet(growth='linear', changepoints=['2019-10-01'])
forecast = m_cp191001.fit(df).predict(future)
fig = m_cp191001.plot(forecast, figsize=(20,6), xlabel='Date', ylabel='Value')
ax = fig.gca()
ax.set_title("[Collectible_average_usd] Linear Forecast - CP 191001 ", size=34)
