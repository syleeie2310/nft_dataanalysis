# Databricks notebook source
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
plt.style.use("ggplot")

# COMMAND ----------

# MAGIC %md
# MAGIC # 데이터 준비

# COMMAND ----------

# MAGIC %md
# MAGIC ## 정제 데이터 로드

# COMMAND ----------

data = pd.read_csv('/dbfs/FileStore/nft/nft_market_cleaned/total_220222_cleaned.csv', index_col = "Date", parse_dates=True, thousands=',')

# COMMAND ----------

data.info()

# COMMAND ----------

data.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 데이터 분리
# MAGIC - collectible_average_usd 피처를 대표로 진행해보자

# COMMAND ----------

# raw 데이터
train_raw = data.loc[:'2021', 'collectible_average_usd']
test_raw = data.loc['2022-01-10':, 'collectible_average_usd']
print(len(train_raw), train_raw.tail())
print(len(test_raw), test_raw.head())

# COMMAND ----------

import plotly.express as px
fig = px.line()
fig.add_scatter(x=train_raw.index, y = train_raw, mode="lines", name = "train")
fig.add_scatter(x=test_raw.index, y = test_raw, mode="lines", name = "test")
fig.update_layout(title = '<b>[collectible_average_usd] Raw data <b>', title_x=0.5, legend=dict(orientation="h", xanchor="right", x=1, y=1.1))
fig.update_yaxes(ticklabelposition="inside top", title=None)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 모델링

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. ARIMA
# MAGIC - arima의 핵심은 어떤 특징을 얼마나 포함하느냐 (전분기? 전년? 등)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1) 모수 설정

# COMMAND ----------

#진행이 오래걸리네.. 잘 되는지 체크해보자
# !pip install tqdm

# COMMAND ----------

from tqdm import tqdm
from tqdm import trange
pd.options.display.float_format = '{: .4f}'.format

# COMMAND ----------

# MAGIC %md
# MAGIC ##### [함수] 성능 평가지표 반환

# COMMAND ----------

# 평가 지표 함수
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

def evaluation(y, y_preds) :
    r2 = r2_score(y, y_preds) # 선형회귀 모델 설명력
    mae = mean_absolute_error(y, y_preds) # 평균 절대 오차
    mape = np.mean(np.abs((y - y_preds) / y)) * 100  # 평균 절대 비율 오차 : 시계열 주요 평가 지표 , # mape가 inf인 이유는 실제y값인 0으로 나눴기 때문, 
    mse = mean_squared_error(y, y_preds) # 평균 오차 제곱합
    rmse = np.sqrt(mean_squared_error(y, y_preds)) # 제곱근 평균 오차제곱합 : 시계열 주요 평가 지표, 작을수록 좋다.
    rmsle = np.sqrt(mean_squared_log_error(y, y_preds))

    return r2, mae, mse, rmse, rmsle, mape

# COMMAND ----------

# MAGIC %md
# MAGIC ##### [함수] aic check

# COMMAND ----------

# 최적 P와 q값 찾는 함수
from statsmodels.tsa.arima_model import ARIMA

def arima_aic_check(data, pdq, sort = 'AIC'):
    order_list = []
    aic_list = []
    bic_list = []
    r2_list = []
    mae_list = []
    mse_list = []
    rmse_list = []
    rmsle_list = []
    mape_list = []
    eval_list_list = [r2_list, mae_list, mse_list, rmse_list, rmsle_list, mape_list]
    p, d, q = pdq
    for i in tqdm(range(p+1)):
        for j in tqdm(range(q+1)): 
            model = ARIMA(data, order=(i,d,j))
            try:
                model_fit = model.fit()
                c_order = f'p:{i} d:{d} q:{j}'
                order_list.append(c_order)
                aic = model_fit.aic
                aic_list.append(aic)
                bic = model_fit.bic # 변수가 많을 때 패널티를 더 많이 줌
                bic_list.append(bic)

                 # 예측(과거 예측)
                preds= model_fit.forecast(steps=len(data.index))
                # 평가
                r2, mae, mse, rmse, rmsle, mape = evaluation(data, preds[0])
                eval_list = [r2, mae, mse, rmse, rmsle, mape]

                for i in range(len(eval_list_list)) :
                    eval_list_list[i].append(eval_list[i]) # 로그 역변환 안해도 될 듯
            except:
                pass
            
    result_df = pd.DataFrame(list(zip(order_list, aic_list, bic_list, r2_list, mae_list, mse_list, rmse_list, rmsle_list, mape_list)),columns=['order','AIC','BIC','r2', 'mae', 'mse', 'rmse', 'rmsle', 'mape'])
    result_df.sort_values(sort, inplace=True)
    return result_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Raw+차분
# MAGIC 차분1 고정, ma는 0과 1  로 최적의 p를 찾아보자

# COMMAND ----------

# q(ma)가 0일 때
# p가 늘어날 수록 aic 설명력은 좋아지지만,모델 정확도 rmse는 감소한다. bic는 p5가 가장 낮음
# 결론 : p1 이 aic/bic/rmse 지표가 최적임
pdq = (15, 1, 0)
arima_aic_check(train_raw, pdq)

# COMMAND ----------

# q(ma)가 1일 때
# p가 늘어날 수록 aic 설명력은 좋아지지만, 모델 정확도 ,rmse는 감소한다. bic는 p8가 가장 낮음
# 결론 : p1 이 aic/bic/rmse 지표가 최적임
pdq = (15, 1, 1)
arima_aic_check(train_raw, pdq)

# COMMAND ----------

pdq = (30, 1, 2)
arima_aic_check(train_raw, pdq)

# COMMAND ----------

# ar과 ma를 함께 올려보자.  6,1,3이 최적이다.  더 안올려도 될듯
guide = round(np.log1p(len(train_raw)))
pdq = (guide, 1, guide)
result = arima_aic_check(train_raw, pdq)
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### log+차분
# MAGIC - 로그데이터는 p 0이 가장 rmse 값이 낮음

# COMMAND ----------

# q가 0일 때
# raw와 유사한 결과,  p0이 가장 최적으로 보임
pdq = (15, 1, 0)
arima_aic_check(np.log1p(train_raw), pdq)

# COMMAND ----------

# q가 1일 때
# raw와 유사한 결과,  p0이 가장 최적으로 보임
pdq = (15, 1, 1)
arima_aic_check(np.log1p(train_raw), pdq)

# COMMAND ----------

#  줄여서 해보자.// 안됨 확실
guide = round(np.log1p(len(train_raw)))
pdq = (guide, 1, guide)
result = arima_aic_check(np.log1p(train_raw), pdq)
result

# COMMAND ----------

# MAGIC %md
# MAGIC ##### pdq 실험 종합
# MAGIC - 결론 : rmse가 낮은 log+차분 데이터의 010을 선택한다???
# MAGIC 
# MAGIC - raw+차분
# MAGIC   - d1,q0 일때 p1이 최적 : aic 7950.88, bic 7967.11, rmse 1923.14
# MAGIC   - d1,q1 일때 p1이 최적 : aic 7948.86, bic 7970.50, rmse 1910.89
# MAGIC   - d1 일때 p6,q3이 최적 : aic 7860.90, bic 7914.99, rmse 2004.75   
# MAGIC 
# MAGIC - log+차분
# MAGIC   - d1,q0일때 p0가 최적 : aic -2741.07, bic -2730.24, rmse 7.31
# MAGIC   - d1,q1일때 p0가 최적 : aic -2863.63, bic -2847.40, rmse 8.81
# MAGIC   
# MAGIC   -> p0은 말이안되지... 예측을 못하잖아...ㅜㅜ
# MAGIC   - > 516이 좋은건가?
# MAGIC   
# MAGIC   - d1 일때 : 정상성 오류로 모델 fit 실패

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) 모형 구축
# MAGIC - log데이터 및 선정된 데이터 및 pdq로 모형 구축

# COMMAND ----------

# MAGIC %md
# MAGIC #### raw+차분

# COMMAND ----------

# 110, 추세포함, raw데이터는 constant값이 유효하다
from statsmodels.tsa.arima_model import ARIMA

order = (1, 1, 0)
model = ARIMA(train_raw, order)
model_110 = model.fit()
# 모델저장
model_110.save('/dbfs/FileStore/nft/nft_market_model/model_110.pkl')
model_110.summary()

# COMMAND ----------

# 111, 추세포함, raw데이터는 constant값이 유효하다
from statsmodels.tsa.arima_model import ARIMA

order = (1, 1, 1)
model = ARIMA(train_raw, order)
model_111 = model.fit()
# 모델저장
model_111.save('/dbfs/FileStore/nft/nft_market_model/model_111.pkl')
model_111.summary()

# COMMAND ----------

# # 613, 추세포함
# from statsmodels.tsa.arima_model import ARIMA

# order = (6, 1, 3)
# model = ARIMA(train_raw, order)
# model_613 = model.fit()
# # 모델저장
# model_613.save('/dbfs/FileStore/nft/nft_market_model/model_613.pkl')
# model_613.summary()

# COMMAND ----------

# 411, 추세포함
from statsmodels.tsa.arima_model import ARIMA

order = (4, 1, 1)
model = ARIMA(train_raw, order)
model_411 = model.fit()
# 모델저장
model_411.save('/dbfs/FileStore/nft/nft_market_model/model_411.pkl')
model_411.summary()

# COMMAND ----------

# 411, 추세미포함
from statsmodels.tsa.arima_model import ARIMA

order = (4, 1, 1)
model = ARIMA(train_raw, order)
model_411_nc = model.fit(trend='nc')
# 모델저장
model_411_nc.save('/dbfs/FileStore/nft/nft_market_model/model_411_nc.pkl')
model_411_nc.summary()

# COMMAND ----------

# 511, 추세포함, raw데이터는 constant값이 유효하다
from statsmodels.tsa.arima_model import ARIMA

order = (5, 1, 1)
model = ARIMA(train_raw, order)
model_511 = model.fit()
# 모델저장
model_511.save('/dbfs/FileStore/nft/nft_market_model/model_511.pkl')
model_511.summary()

# COMMAND ----------

# 711, 추세포함, raw데이터는 constant값이 유효하다
from statsmodels.tsa.arima_model import ARIMA

order = (7, 1, 1)
model = ARIMA(train_raw, order)
model_711 = model.fit()
# 모델저장
model_711.save('/dbfs/FileStore/nft/nft_market_model/model_711.pkl')
model_711.summary()

# COMMAND ----------

# 1011, 추세포함, raw데이터는 constant값이 유효하다
from statsmodels.tsa.arima_model import ARIMA

order = (10, 1, 1)
model = ARIMA(train_raw, order)
model_1011 = model.fit()
# 모델저장
model_1011.save('/dbfs/FileStore/nft/nft_market_model/model_1011.pkl')
model_1011.summary()

# COMMAND ----------

# 1510, 추세포함, constant 유의함
from statsmodels.tsa.arima_model import ARIMA

order = (15, 1, 0)
model = ARIMA(train_raw, order)
model_1510 = model.fit()
# 모델저장
model_1510.save('/dbfs/FileStore/nft/nft_market_model/model_1510.pkl')
model_1510.summary()

# COMMAND ----------

# 1511, 추세포함, raw데이터는 constant값이 유효하다
from statsmodels.tsa.arima_model import ARIMA

order = (15, 1, 1)
model = ARIMA(train_raw, order)
model_1511 = model.fit()
# 모델저장
model_1511.save('/dbfs/FileStore/nft/nft_market_model/model_1511.pkl')
model_1511.summary()

# COMMAND ----------

# 1511_nc, 추세 미포함
from statsmodels.tsa.arima_model import ARIMA

order = (15, 1, 1)
model = ARIMA(train_raw, order)
model_1511_nc = model.fit(trend='nc')
# 모델저장
model_1511_nc.save('/dbfs/FileStore/nft/nft_market_model/model_1511_nc.pkl')
model_1511_nc.summary()

# COMMAND ----------

# 1710, 추세포함, 
from statsmodels.tsa.arima_model import ARIMA

order = (17, 1, 0)
model = ARIMA(train_raw, order)
model_1710 = model.fit()
# 모델저장
model_1710.save('/dbfs/FileStore/nft/nft_market_model/model_1710.pkl')
model_1710.summary()

# COMMAND ----------

# 1710, 추세미포함, 
from statsmodels.tsa.arima_model import ARIMA

order = (17, 1, 0)
model = ARIMA(train_raw, order)
model_1710_nc = model.fit(trend='nc')
# 모델저장
model_1710_nc.save('/dbfs/FileStore/nft/nft_market_model/model_1710_nc.pkl')
model_1710_nc.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC #### log+차분

# COMMAND ----------

# 010, contant 값이 유효하지 않다.
from statsmodels.tsa.arima_model import ARIMA

order = (0, 1, 0)
model = ARIMA(np.log1p(train_raw), order)
model_log_010 = model.fit()
# 모델저장
model_log_010.save('/dbfs/FileStore/nft/nft_market_model/model_log_010.pkl')
model_log_010.summary()

# COMMAND ----------

# # 010, contant 값이 유효하다.
# from statsmodels.tsa.arima_model import ARIMA

# order = (0, 1, 0)
# model = ARIMA(np.log1p(train_raw), order)
# model_log_010_nc = model.fit(trend='nc')
# # 모델저장
# model_log_010_nc.save('/dbfs/FileStore/nft/nft_market_model/model_log_010_nc.pkl')
# model_log_010_nc.summary()

# COMMAND ----------

# 011, contant 값이 0.05보다 높다.
from statsmodels.tsa.arima_model import ARIMA

order = (0, 1, 1)
model = ARIMA(np.log1p(train_raw), order)
model_log_011 = model.fit()
# 모델저장
model_log_011.save('/dbfs/FileStore/nft/nft_market_model/model_log_011.pkl')
model_log_011.summary()

# COMMAND ----------

# 011_nc, contant 값이 유효하다.
from statsmodels.tsa.arima_model import ARIMA

order = (0, 1, 1)
model = ARIMA(np.log1p(train_raw), order)
model_log_011_nc = model.fit(trend='nc')
# 모델저장
model_log_011_nc.save('/dbfs/FileStore/nft/nft_market_model/model_log_011_nc.pkl')
model_log_011_nc.summary()

# COMMAND ----------

# 3,1,1  추세 포함, constant p값이 0.05보다 높으니 추세를 빼보자.
from statsmodels.tsa.arima_model import ARIMA

order = (3, 1, 1)
model = ARIMA(np.log1p(train_raw), order)
model_log_311 = model.fit()
# 모델저장
model_log_311.save('/dbfs/FileStore/nft/nft_market_model/model_log_311.pkl')
model_log_311.summary()

# COMMAND ----------

# 3,1,1  추세 미포함
from statsmodels.tsa.arima_model import ARIMA

order = (3, 1, 1)
model = ARIMA(np.log1p(train_raw), order)
model_log_311_nc = model.fit(trend='nc')
# 모델저장
model_log_311_nc.save('/dbfs/FileStore/nft/nft_market_model/model_log_311_nc.pkl')
model_log_311_nc.summary()

# COMMAND ----------

# 5,1,1  추세 포함
from statsmodels.tsa.arima_model import ARIMA

order = (5, 1, 1)
model = ARIMA(np.log1p(train_raw), order)
model_log_511 = model.fit()
# 모델저장
model_log_511.save('/dbfs/FileStore/nft/nft_market_model/model_log_511.pkl')
model_log_511.summary()

# COMMAND ----------

# 5,1,1  추세 미포함
from statsmodels.tsa.arima_model import ARIMA

order = (5, 1, 1)
model = ARIMA(np.log1p(train_raw), order)
model_log_511_nc = model.fit(trend='nc')
# 모델저장
model_log_511_nc.save('/dbfs/FileStore/nft/nft_market_model/model_log_511_nc.pkl')
model_log_511_nc.summary()

# COMMAND ----------

# 5,1,6  추세 포함
from statsmodels.tsa.arima_model import ARIMA

order = (5, 1, 6)
model = ARIMA(np.log1p(train_raw), order)
model_log_516 = model.fit()
# 모델저장
model_log_516.save('/dbfs/FileStore/nft/nft_market_model/model_log_516.pkl')
model_log_516.summary()

# COMMAND ----------

# 5,1,6  추세 미포함
from statsmodels.tsa.arima_model import ARIMA

order = (5, 1, 6)
model = ARIMA(np.log1p(train_raw), order)
model_log_516_nc = model.fit(trend='nc')
# 모델저장
model_log_516_nc.save('/dbfs/FileStore/nft/nft_market_model/model_log_516_nc.pkl')
model_log_516_nc.summary()

# COMMAND ----------

# 6,1,0  추세 포함
from statsmodels.tsa.arima_model import ARIMA

order = (6, 1, 0)
model = ARIMA(np.log1p(train_raw), order)
model_log_610 = model.fit()
# 모델저장
model_log_610.save('/dbfs/FileStore/nft/nft_market_model/model_log_610.pkl')
model_log_610.summary()

# COMMAND ----------

# 6,1,1  추세 포함
from statsmodels.tsa.arima_model import ARIMA

order = (6, 1, 1)
model = ARIMA(np.log1p(train_raw), order)
model_log_611 = model.fit()
# 모델저장
model_log_611.save('/dbfs/FileStore/nft/nft_market_model/model_log_611.pkl')
model_log_611.summary()

# COMMAND ----------

# 7,1,1  추세 포함
from statsmodels.tsa.arima_model import ARIMA

order = (7, 1, 1)
model = ARIMA(np.log1p(train_raw), order)
model_log_711 = model.fit()
# 모델저장
model_log_711.save('/dbfs/FileStore/nft/nft_market_model/model_log_711.pkl')
model_log_711.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3) 예측 및 성능 평가
# MAGIC - 날짜로 예측 어케함?
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### 4) 성능 평가
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

# MAGIC %md
# MAGIC ###### [함수] 예측결과 비교 시각화

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
    fig.update_layout(title = '<b>[collectible_average_usd] Raw data ARIMA<b>', title_x=0.5, legend=dict(orientation="h", xanchor="right", x=1, y=1.2))
    fig.update_yaxes(ticklabelposition="inside top", title=None)
    fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### [함수] 예측 실행기

# COMMAND ----------

def forecast (train, test, modelName, datatype):
    y_preds = []
    pred_upper = []
    pred_lower = []
    
    if datatype == 'raw':
        pass
    elif datatype == 'log':
        train = np.log1p(train)
        test = np.log1p(test)
    else:
        print('입력값이 유효하지 않습니다.')
        
    # 예측
    y_pred, stderr, interval  = modelName.forecast(steps=len(test.index)) 
    y_preds = y_pred.tolist()      
    pred_upper.append(interval[1])
    pred_lower.append(interval[0])
    
    if datatype == 'raw':
        pass
    elif datatype == 'log':
        test =  np.expm1(test) 
        y_preds = np.expm1(y_preds) 
    else:
        print('입력값이 유효하지 않습니다.')
    
    
    # 성능 평가지표 출력
    r2, mae, mse, rmse, rmsle, mape = evaluation(test, y_preds)
    print(f'r2: {r2}, mae: {mae}, mse: {mse}, rmse: {rmse}, rmsle: {rmsle}, mape: {mape}')

    # 예측결과 시각화 
    if datatype == 'raw':
        forecast_plot(train, test, y_preds, pred_upper, pred_lower)
    elif datatype == 'log': # 로그변환된 데이터를 다시 역변환
#         forecast_plot(train, test, y_preds, pred_upper, pred_lower)
        forecast_plot(np.expm1(train), test, y_preds, np.expm1(pred_upper), np.expm1(pred_lower))
    else :
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC ###### raw+차분

# COMMAND ----------

#110
forecast(train_raw, test_raw, model_110, 'raw')

# COMMAND ----------

#111
forecast(train_raw, test_raw, model_111, 'raw')

# COMMAND ----------

#411
forecast(train_raw, test_raw, model_411, 'raw')

# COMMAND ----------

#411_nc
forecast(train_raw, test_raw, model_411_nc, 'raw')

# COMMAND ----------

#511
forecast(train_raw, test_raw, model_511, 'raw')

# COMMAND ----------

#711
forecast(train_raw, test_raw, model_711, 'raw')

# COMMAND ----------

#1011
forecast(train_raw, test_raw, model_1011, 'raw')

# COMMAND ----------

#1510
forecast(train_raw, test_raw, model_1510, 'raw')

# COMMAND ----------

#1511
forecast(train_raw, test_raw, model_1511, 'raw')

# COMMAND ----------

#1511_nc
forecast(train_raw, test_raw, model_1511_nc, 'raw')

# COMMAND ----------

#1710
forecast(train_raw, test_raw, model_1710, 'raw')

# COMMAND ----------

#1710_nc
forecast(train_raw, test_raw, model_1710_nc, 'raw')

# COMMAND ----------

# MAGIC %md
# MAGIC ###### log+차분

# COMMAND ----------

forecast(train_raw, test_raw, model_log_010, 'log')

# COMMAND ----------

forecast(train_raw, test_raw, model_log_011, 'log')

# COMMAND ----------

forecast(train_raw, test_raw, model_log_011_nc, 'log')

# COMMAND ----------

forecast(train_raw, test_raw, model_log_311, 'log')

# COMMAND ----------

forecast(train_raw, test_raw, model_log_311_nc, 'log')

# COMMAND ----------

forecast(train_raw, test_raw, model_log_511, 'log')

# COMMAND ----------

forecast(train_raw, test_raw, model_log_511_nc, 'log')

# COMMAND ----------

forecast(train_raw, test_raw, model_log_516, 'log')

# COMMAND ----------

forecast(train_raw, test_raw, model_log_516_nc, 'log')

# COMMAND ----------

# MAGIC %md
# MAGIC ###### 실험 종합
# MAGIC - log차분모델은 constant값이 유효하지 않아 nc설정이 필요하다.
# MAGIC 
# MAGIC |Data|PDQ|R2|MAE|MSE|RMSE|RMSLE|MAPE|
# MAGIC |----|----|----|----|----|----|----|----|
# MAGIC |Raw차분|110|-2.4906|63.5971| 4636.1154| 68.0890| 0.0503| 4.6023| 
# MAGIC |Raw차분|111	|-2.7731|	66.2462|	5011.3076| 	70.7906| 	0.0524| 	4.7945| 
# MAGIC |Raw차분|711	|-0.6974|	44.0199| 	2254.4264| 	47.4808| 	0.0348| 	3.1861| 
# MAGIC |Log차분|010	|-0.3871|	0.0264| 	0.0010| 	0.0313| 	0.0038| 	0.3646| 
# MAGIC |Log차분|011	|-5.9524|	0.0577| 	0.0049| 	0.0702| 	0.0085| 	0.7975| 
# MAGIC |Log차분|011_nc	|-7.6138|	0.0734|	0.0061| 	0.0781| 	0.0095| 	1.0155| 
# MAGIC |Log차분|311	|-13.3833|	0.0838| 	0.0102| 	0.1009| 	0.0121| 	1.1570| 
# MAGIC |Log차분|311_nc	|-8.1046|	0.0756| 	0.0064| 	0.0803| 	0.0098| 	1.0458| 
# MAGIC |Log차분|516	|-138.0773|	0.2769| 	0.0985| 	0.3139| 	0.0372| 	3.8273| 
# MAGIC |Log차분|516_nc|-6.6581|	0.0676|	 0.0054| 	0.0736| 	0.0090| 	0.9350|

# COMMAND ----------

# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
# print(len(test), len(test_preds[0]))
# y = test
# # y_preds = test_preds[0]
# print('r2_score : {:.3f}'.format(r2_score(y, y_preds)))
# print('MAE : {:.6f}'.format(mean_absolute_error(y, y_preds))) # 이상치 영향 받음
# print('MSE : {:.6f}'.format(mean_squared_error(y, y_preds))) # 특이 값이 커서  값이 너무 큼
# print('RMSE : {:.6f}'.format(np.sqrt(mean_squared_error(y, y_preds))))
# print('RMSLE : {:.6f}'.format(np.sqrt(mean_squared_log_error(y, y_preds))))

# COMMAND ----------

# # model_fit.plot_predict('2017-06-23', '2022-02-28')
# model_fit.plot_predict(1, 1334)

# COMMAND ----------

# 과거 데이터로 테스트해보자
# model_fit.predict(1, 10, typ='levels') # typ= default값이 linear, 예측할때 levels

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4) 모수 추정(pass)
# MAGIC - 시간부족으로 생략
# MAGIC - 1.LSE방법, 2. MLE 방법

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5) 모형 진단(pass)
# MAGIC - 잔차 분석, 시간부족으로 생략
# MAGIC - 예측값 정상성 검증
# MAGIC - 예측값의 잔차 ACF를 그려 정상성을 체크한다.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 계절성 조정 데이터

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1) 모수 설정

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

train_adj = df_adjusted[:'2021']
test_adj = df_adjusted['2022-01-10':]
print(len(train_adj), train_adj.tail())
print(len(test_adj), test_adj.head())# 차분먼저하고 로그변환하면 오류남..

# COMMAND ----------

# 조정데이터는 음수가 있어서 로그변환이 안됨, 
# 음수 최소값이 -9.9 이므로 +10해서 입력해보자

pdq = (round(np.log(len(train_adj+10))), 1, 1)
arima_aic_check(train_adj+10, pdq) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2) 모형 구축

# COMMAND ----------

# p0 모형 구축, ar t검정 적합
# constant 0.05 이하, t검정 0.05 이하, 
order = (0, 1, 1)
model = ARIMA(train_adj, order)
model_adj_011 = model.fit()
# 모델저장
model_adj_011.save('/dbfs/FileStore/nft/nft_market_model/model_adj_011.pkl')
model_adj_011.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3) 미래 예측

# COMMAND ----------

forecast(train_adj, test_adj, model_adj_011)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. SARIMA(Pass)
# MAGIC - PDQs, 시계열분해 결과 1일단위 1년 주기이므로 s는 365로 설정한다
# MAGIC - s 인자, 데이터 순환주기 아이디어 필요
# MAGIC   - 데이터가 월단위로 분리되거 계절주기가 1년이면 s를 12로 설정
# MAGIC   - 데이터가 일단위로 분리되고 계절주기가 주 이면 s를 7로 설정

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Auto ARIMA
# MAGIC - auto arima를 쓰면 모두 자동으로 값을 찾아주고,  신규 관측값 refresh(update)도 가능하다.
# MAGIC - https://assaeunji.github.io/data%20analysis/2021-09-25-arimastock/

# COMMAND ----------

# autoarima패키지 설치
# !pip install pmdarima

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1) 모수설정 및 구축

# COMMAND ----------

# 최적 차분을 찾아주는 함수 
from pmdarima.arima import ndiffs
kpss_diffs = ndiffs(train, alpha=0.05, test='kpss', max_d=2)
adf_diffs = ndiffs(train, alpha=0.05, test='adf', max_d=2)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"추정된 차수 d = {n_diffs}")

# COMMAND ----------

# 최적 pdq를 찾아주는 함수
# raw data 역시 0, 1, 1이 최선이네
import pmdarima as pm
model_pm = pm.auto_arima(y = train, d = 1, start_p = 0
                      , max_p = round(np.log(len(train)))   
                      , q = 1  
                      , m = 1 # 디폴트값 1은 계절적 특징이 없을 때 .  
                      , seasonal = False # 계절성 ARIMA가 아니라면 필수!
                      , stepwise = True # 최적의 모수를 찾기 위해 힌드만-칸다카르 알고리즘을 사용할지 여부
                      , trace=True) # stepwise 모델을 fit할 때마다 결과 출력여부
                       

# COMMAND ----------

# https://assaeunji.github.io/data%20analysis/2021-09-25-arimastock/
# adj 데이터로 0, 1, 1
import pmdarima as pm
model_adj_pm = pm.auto_arima(y = train_adj, d = 1, start_p = 0
                      , max_p = round(np.log(len(train_adj)))   
                      , q = 1  
                      , m = 1 # 디폴트값 1은 계절적 특징이 없을 때 .  
                      , seasonal = False # 계절성 ARIMA가 아니라면 필수!
                      , stepwise = True # 최적의 모수를 찾기 위해 힌드만-칸다카르 알고리즘을 사용할지 여부
                      , trace=True) # stepwise 모델을 fit할 때마다 결과 출력여부
                       

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) 모형 refresh 예측 및 평가

# COMMAND ----------

# 한스텝씩 예측
def forecast_pm_one_step(test, modelName):
      # 한 스텝씩!, 신뢰구간 출력
    y_pred, conf_int = modelName.predict(n_prediods=1, return_conf_int=True)
    return (
        y_pred.tolist()[0],
        np.asarray(conf_int).tolist()[0]
    )

# COMMAND ----------

def forecast_pm (train, test, modelName):
    y_preds = []
    pred_upper = []
    pred_lower = []
    temp = train.values
    

    for new_ob in test:
        y_pred, conf = forecast_pm_one_step(test, modelName) 
        y_preds.append(y_pred)
        pred_upper.append(conf[1])
        pred_lower.append(conf[0])
        ## 모형 업데이트 !!
        modelName.update(new_ob)
    # 성능 평가지표 출력
    r2, mae, mse, rmse, rmsle = evaluation(test, y_preds)
    print(f'r2: {r2}, mae: {mae}, mse: {mse}, rmse: {rmse}, rmsle: {rmsle}')
    # 예측결과 시각화
    forecast_plot(train, test, y_preds, pred_upper, pred_lower)

# COMMAND ----------

forecast_pm(train, test, model_pm)

# COMMAND ----------

forecast_pm(train_adj, test_adj, model_adj_pm)

# COMMAND ----------

# MAGIC %md
# MAGIC ## [실험4] 큰변곡점을 예측할 수 있을까?

# COMMAND ----------

# MAGIC %md
# MAGIC ### 데이터분리

# COMMAND ----------

# raw 데이터
train1 = data.loc[:'2021', 'collectible_average_usd']
test1 = data.loc['2022':, 'collectible_average_usd']
print(len(train1), train1.tail())
print(len(test1), test1.head())

# COMMAND ----------

# 계절성 조정 데이터
train1_adj = df_adjusted[:'2021']
test1_adj = df_adjusted['2022':]
print(len(train1_adj), train1_adj.tail())
print(len(test1_adj), test1_adj.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic+adj+011

# COMMAND ----------

# p0 모형 구축, ar t검정 적합
# constant 0.05 이하, t검정 0.05 이하, 
order = (0, 1, 1)
model = ARIMA(train1, order)
model1_011 = model.fit()
# 모델저장
model1_011.save('/dbfs/FileStore/nft/nft_market_model/model1_011.pkl')
model1_011.summary()

# COMMAND ----------

forecast(train1, test1, model1_011)

# COMMAND ----------

# p0 모형 구축, ar t검정 적합
# constant 0.05 이하, t검정 0.05 이하, 
order = (0, 1, 1)
model = ARIMA(train1_adj, order)
model1_adj_011 = model.fit()
# 모델저장
model1_adj_011.save('/dbfs/FileStore/nft/nft_market_model/model1_adj_011.pkl')
model1_adj_011.summary()

# COMMAND ----------

forecast(train1_adj, test1_adj, model1_adj_011)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Auto+adj+011

# COMMAND ----------

import pmdarima as pm
model1_pm = pm.auto_arima(y = train, d = 1, start_p = 0
                      , max_p = round(np.log(len(train1)))   
                      , q = 1  
                      , m = 1 # 디폴트값 1은 계절적 특징이 없을 때 .  
                      , seasonal = False # 계절성 ARIMA가 아니라면 필수!
                      , stepwise = True # 최적의 모수를 찾기 위해 힌드만-칸다카르 알고리즘을 사용할지 여부
                      , trace=True) # stepwise 모델을 fit할 때마다 결과 출력여부
                       

# COMMAND ----------

forecast_pm(train1, test1, model1_pm)

# COMMAND ----------

import pmdarima as pm
model1_adj_pm = pm.auto_arima(y = train, d = 1, start_p = 0
                      , max_p = round(np.log(len(train1_adj)))   
                      , q = 1  
                      , m = 1 # 디폴트값 1은 계절적 특징이 없을 때 .  
                      , seasonal = False # 계절성 ARIMA가 아니라면 필수!
                      , stepwise = True # 최적의 모수를 찾기 위해 힌드만-칸다카르 알고리즘을 사용할지 여부
                      , trace=True) # stepwise 모델을 fit할 때마다 결과 출력여부
                       

# COMMAND ----------

forecast_pm(train1_adj, test1_adj, model1_adj_pm)

# COMMAND ----------

# MAGIC %md
# MAGIC ### [실험3&4 요약]
# MAGIC - 실험3 : Basic보다 Auto의 예측 성능이 더 좋다.
# MAGIC - 실험4 : 큰변곡점이 있을 경우 basic raw보다 basic adj가 더 예측을 잘한다. auto는 둘다 비슷하다

# COMMAND ----------

# MAGIC %md
# MAGIC ## [모델링 실험 결론]
# MAGIC - 최적의 모델 : Seasonal Adjusted Data + AutoARIMA + pdq(0,1,1)
