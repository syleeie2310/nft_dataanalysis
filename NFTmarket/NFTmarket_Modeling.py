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
# MAGIC ### 1) raw 데이터 시각화
# MAGIC - 평균이 일정하지 않음, 대체로 MA특징을 가짐 (PACF), 차분 필요
# MAGIC - 2개의 경향으로 나눠짐
# MAGIC   - all, collectible, art, metaverse 
# MAGIC   - defi, game, utility

# COMMAND ----------

autoCorrelationF(data, 'average_usd') #raw df, feature

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) 차분
# MAGIC - 대체로 정상성을 보여 1차로 충분해보임, 그러나 특정 변동폭이 너무 커서 전체적인 패턴을 여전히 보기 어려움
# MAGIC - 큰 변동폭이 있음 미 all, collectible, art, metaverse 
# MAGIC - all은 21.11.15 에 큰 변동폭
# MAGIC - collectible, art, metaverse 가 22.1.8에 큰 변동이 있음
# MAGIC - defi, game, utility는 모두 다름

# COMMAND ----------

import plotly.express as px
from plotly.subplots import make_subplots

def diff_plot(data, feature, plot):

    # 피처 분류기 호출
    col_list = feature_classifier(data, feature)
    
    diff_data = data.diff(periods=1).dropna() # dropna()는 diff를 통해 생긴 데이터 공백제거
    
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

diff_plot(data[1:], 'average_usd', 'acf') #raw df, feature

# COMMAND ----------

# MAGIC %md
# MAGIC #### log변환
# MAGIC - 여전히 갭이 커서 보기 어렵다. 적절한지 모르겠다.

# COMMAND ----------

diff_plot(np.log1p(data), 'average_usd', 'line') #raw df, feature

# COMMAND ----------

# 첫번째는 자기자신과의 상관관계이므로 1이 나올수밖에 없다.
diff_plot(np.log1p(data), 'average_usd', 'acf') #raw df, feature

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3) ADF 정상성 테스트(adfuller)
# MAGIC - 검증 조건 ( p-value : 5%이내면 reject으로 대체가설 선택됨 )
# MAGIC - 귀무가설(H0): non-stationary.
# MAGIC - 대체가설 (H1): stationary.

# COMMAND ----------

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

# raw데이터 : utlity외 전부 0.05보다 크므로 귀무가설 채택하여 "비정상성"
# adf 작을 수록 귀무가설을 기각시킬 확률이 높다
adf_test(data, 'average_usd')

# COMMAND ----------

# 1차 차분 데이터, 전부다 p-value가 0에 수렴하여 귀무가설 기각, "정상성"
adf_test(data.diff(periods=1).dropna() , 'average_usd')

# COMMAND ----------

# log변환 + 1차 차분 데이터, 위 1차차분결과와 비슷하다. 유의미한지 모르겠음.
adf_test(np.log1p(data).diff(periods=1).dropna() , 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4) [종합요약] "average_usd"피처, 카테고리별 자기상관계수
# MAGIC (P=?, D=1, Q=1)
# MAGIC - acf/pacf 그래프에서  p와 q값을 선정하는 것은 권장하지 않음, 정확하지 않고 해석하기 어려움
# MAGIC - 전체 행 길이의 log변환 값을 최대치로, ar을 실험하는 것을 권장

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 시계열 분해
# MAGIC - 시간 부족으로 Pass

# COMMAND ----------

# MAGIC %md
# MAGIC # 모델링

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. 데이터 분리
# MAGIC - collectible_average_usd 피처를 대표로 진행해보자

# COMMAND ----------

train = data.loc[:'2022-01', 'collectible_average_usd']
test = data.loc['2022-02':, 'collectible_average_usd']
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
# MAGIC ## 1. ARIMA
# MAGIC - arima의 핵심은 어떤 특징을 얼마나 포함하느냐 (전분기? 전년? 등)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 실험1

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1) 모수 설정

# COMMAND ----------

# 평가 지표 함수
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

def evaluation(y, y_preds) :
#     print(y)
#     print(y_preds)
    r2 = r2_score(y, y_preds)
    mae = mean_absolute_error(y, y_preds)
    mse = mean_squared_error(y, y_preds)
    rmse = np.sqrt(mean_squared_error(y, y_preds))
    rmsle = np.sqrt(mean_squared_log_error(y, y_preds))
    return r2, mae, mse, rmse, rmsle

# COMMAND ----------

# 최적 P값 찾는 함수
def arima_aic_check(data, pdq, sort = 'AIC'):
    order_list = []
    aic_list = []
    r2_list = []
    mae_list = []
    mse_list = []
    rmse_list = []
    rmsle_list = []
    eval_list_list = [r2_list, mae_list, mse_list, rmse_list, rmsle_list]
    p, d, q = pdq
    for i in range(p+1):
        model = ARIMA(data, order=(i,d,q))
        model_fit = model.fit()
        c_order = f'p:{i} d:{d} q:{q}'
        order_list.append(c_order)
        aic = model_fit.aic
        aic_list.append(aic)
        # 예측(과거 예측)
        preds= model_fit.forecast(steps=len(data.index))
        # 평가
        r2, mae, mse, rmse, rmsle = evaluation(data, preds[0])
        eval_list = [r2, mae, mse, rmse, rmsle]
        for i in range(len(eval_list_list)) :
            eval_list_list[i].append(eval_list[i])

    result_df = pd.DataFrame(list(zip(order_list, aic_list, r2_list, mae_list, mse_list, rmse_list, rmsle_list)),columns=['order','AIC','r2', 'mae', 'mse', 'rmse', 'rmsle'])
    result_df.sort_values(sort, inplace=True)
    return result_df

# COMMAND ----------

pd.options.display.float_format = '{: .4f}'.format

# COMMAND ----------

# 어떻게 선정해야하지?? 값이 전부다 너무 크다 ㅜㅜ
pdq = (round(np.log(len(data))), 1, 1)
arima_aic_check(train, pdq)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2) 모형 구축

# COMMAND ----------

# constant 0.05 이하, t검정 0.05 이하, 
order = (0, 1, 1)
model = ARIMA(data, order)
model_fit = model.fit()
# 모델저장
model_fit.save('/dbfs/FileStore/nft/nft_market_model/model.pkl')
model_fit.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3) 미래 예측
# MAGIC - 날짜로 예측 어케함?

# COMMAND ----------

from statsmodels.tsa.arima_model import ARIMAResults
# 한스텝씩 예측
def forecast_one_step(test):
      # 한 스텝씩!, 예측구간 출력
    loaded = ARIMAResults.load('/dbfs/FileStore/nft/nft_market_model/model.pkl')
    y_pred, stderr, interval  = loaded.forecast(steps=len(test.index))  # 예측값, 표준오차(stderr), 신뢰구간(upperbound, lower bound), 
    return (
        y_pred.tolist()[0],
        np.asarray(interval).tolist()[0]
    )

# COMMAND ----------

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def forecast_plot(train, test, y_preds, pred_upper, pred_lower):
    fig = go.Figure([
        # 훈련 데이터-------------------------------------------------------
        go.Scatter(x = train.index, y = train, name = "Train", mode = 'lines'
                  ,line=dict(color = 'royalblue'))
        # 테스트 데이터------------------------------------------------------
        , go.Scatter(x = test.index, y = test, name = "Test", mode = 'lines'
                    ,line = dict(color = 'rgba(0,0,30,0.5)'))
        # 예측값-----------------------------------------------------------
        , go.Scatter(x = test.index, y = y_preds, name = "Prediction", mode = 'lines'
                         ,line = dict(color = 'red', dash = 'dot', width=3))

        # 신뢰 구간---------------------------------------------------------
        , go.Scatter(x = test.index.tolist() + test.index[::-1].tolist() 
                    ,y = pred_upper + pred_lower[::-1] ## 상위 신뢰 구간 -> 하위 신뢰 구간 역순으로
                    ,fill='toself'
                    ,fillcolor='rgba(0,0,30,0.1)'
                    ,line=dict(color='rgba(0,0,0,0)')
                    ,hoverinfo="skip"
                    ,showlegend=False)
    ])
    fig.update_layout(title = '<b>[collectible_average_usd] Raw data ARIMA(0,1,1)<b>', title_x=0.5, legend=dict(orientation="h", xanchor="right", x=1, y=1.2))
    fig.update_yaxes(ticklabelposition="inside top", title=None)
    fig.show()

# COMMAND ----------

def forecast (train, test, update):
    y_preds = []
    pred_upper = []
    pred_lower = []
    temp = train.values
    

    for new_ob in test:
        y_pred, interval = forecast_one_step(test) 
        y_preds.append(y_pred)
        pred_upper.append(interval[1])
        pred_lower.append(interval[0])
        ## 모형 업데이트 !!
#         if update == True :
#             #model.update(new_ob) # 왜 업데이트 속성이 없는가 ㅜㅜ
#             temp = data[:]
#             model_temp = ARIMA(temp, (0, 1, 1))
#             model_temp_fit = model_temp.fit()
#             model_temp_fit.save('/dbfs/FileStore/nft/nft_market_model/model.pkl')
#         else :
#             pass
    forecast_plot(train, test, y_preds, pred_upper, pred_lower)

# COMMAND ----------

forecast(train, test, False)

# COMMAND ----------

forecast(train, test, True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4) 성능 평가
# MAGIC https://mizykk.tistory.com/102
# MAGIC - R2 : (1에 가까울 수록 좋음)분산기반 예측 성능 평가 
# MAGIC - MAE : (0에 가까울 수록 좋음)(Scale영향)
# MAGIC   - 오차들의 절대값 평균, MSE보다 이상치에 덜 민감
# MAGIC - MSE : (0에 가까울 수록 좋음)(Scale영향)
# MAGIC   - 예측값과 실체값의 차이인 "오차들의 제곱 평균", 이상치에 민감
# MAGIC - RMSE : (0에 가까울 수록 좋음)(Scale영향)
# MAGIC   - MSE의 루트값, 오류지표를 실제값과 유사한 단위로 변환하여 해석이 용이
# MAGIC - RMSLE : (Scale영향)
# MAGIC   - 오차를 구할 때 RMSE에 log 변환
# MAGIC   - 이상치에 덜 민감, 상대적 error값, under estimation에 큰 패널티
# MAGIC   
# MAGIC   
# MAGIC |방법|	Full|	잔차 계산|	이상치 영향|
# MAGIC |----|----|----|----|
# MAGIC |MAE|	Mean Absolute Error|	Absolute Value|	Yes|
# MAGIC |MSE|	Mean Squared Error|	Square	|No|
# MAGIC |RMSE|	Root Mean Squared Error|	Square	|No|
# MAGIC |MAPE|	Mean Absolute Percentage Error|	Absolute Value	|Yes|
# MAGIC |MPE|	Mean Percentage Error|	N/A	|Yes|

# COMMAND ----------

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

# COMMAND ----------

print(len(test), len(test_preds[0]))

# COMMAND ----------

y = test
# y_preds = test_preds[0]
print('r2_score : {:.3f}'.format(r2_score(y, y_preds)))
print('MAE : {:.6f}'.format(mean_absolute_error(y, y_preds))) # 이상치 영향 받음
print('MSE : {:.6f}'.format(mean_squared_error(y, y_preds))) # 특이 값이 커서  값이 너무 큼
print('RMSE : {:.6f}'.format(np.sqrt(mean_squared_error(y, y_preds))))
print('RMSLE : {:.6f}'.format(np.sqrt(mean_squared_log_error(y, y_preds))))

# COMMAND ----------

model_fit.plot_predict()

# COMMAND ----------

# model_fit.plot_predict('2017-06-23', '2022-02-28')
model_fit.plot_predict(1, 1334)

# COMMAND ----------

fore = model_fit.forecast(steps=1)
print(fore)
# 예측값, stderr, upperbound, lower bound
# 2월 1일을 37로 예측

# COMMAND ----------

test['all_average_usd']

# COMMAND ----------

# 과거 데이터로 테스트해보자
model_fit.predict(1, 10, typ='levels') # typ= default값이 linear, 예측할때 levels

# COMMAND ----------

# ... 뭥미?
train['all_average_usd'].head(10)

# COMMAND ----------

# 미래 예측하기 # 왜 안되지 ㅜㅜ, 인덱스가 없어서 그런듯
preds = model_fit.predict('2022-02-01', '2022-02-28', typ='levels')
preds

# COMMAND ----------

preds = model_fit.predict(1,1350, typ='levels')
preds

# COMMAND ----------

# MAGIC %md
# MAGIC ### 모형 성능 판단
# MAGIC - AIC, BIC 가 더 작은 모형

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
arima_aic_check(train['all_average_usd'], [3,1,3])

# COMMAND ----------

# P>z값이 일반적으로 학습의 적정성을 위해 호가인되는 T-검정값
# 즉, p-value 0.05수준에서 보면 MA(1)와 MA(2)의 값은 유효한데, 모형의 constant는 유효하지 않다. 따라서 모형의 model.fit()파라미터중trend='c'가 아니라 nc로 설정하는 것이 옳다
order = (1, 1, 2)
model = ARIMA(train['all_average_usd'], order)
rfit = model.fit()
rfit.summary()

# COMMAND ----------

# constraint가 없는 모형으로 fitting하니 t 검정값이 더 좋아짐?
order = (1, 1, 2)
model = ARIMA(train['all_average_usd'], order)
model_fit = model.fit(trend='nc')
model_fit.summary()

# COMMAND ----------

test 데이터 의 rmse 평가지표를 보는 것

# COMMAND ----------

# MAGIC %md
# MAGIC ### 모수 추정 -> pass
# MAGIC - 1.LSE방법, 2. MLE 방법

# COMMAND ----------

# MAGIC %md
# MAGIC ### 모형 진단 : 잔차 분석  -> pass
# MAGIC - 예측값 정상성 검증
# MAGIC - 예측값의 잔차 ACF를 그려 정상성을 체크한다.

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
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

import pandas as pd
from prophet import Prophet

# COMMAND ----------

# 칼럼명 변경
df = train['all_average_usd'].reset_index()
df.columns = ['ds', 'y']
print(df)

# COMMAND ----------

m = Prophet() # linear
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
# 외삽(미래예측 : 예측구간), 내삽에 따라 다름
# 예측구간과 신뢰구간은 다름
plot_plotly(m, forecast)

# COMMAND ----------

1# Python
plot_components_plotly(m, forecast)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Forecasting Growth

# COMMAND ----------

df['cap'] = 140 # 이걸 어떻게 정해야 하지?, 최대 140을 넣자

# COMMAND ----------

mlg = Prophet(growth='logistic')
mln = Prophet(growth='linear')
mlg.fit(df)
mln.fit(df)


# COMMAND ----------

# 선형 예측
future = mln.make_future_dataframe(periods=30)
future['cap'] = 140
fcst = mln.predict(future)
fig = mln.plot(fcst)

# COMMAND ----------

# 로지스틱 예측
future = mlg.make_future_dataframe(periods=30)
future['cap'] = 140 # floor 값 도 필요
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


