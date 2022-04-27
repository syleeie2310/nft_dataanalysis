# Databricks notebook source
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import adfuller
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
# MAGIC # 그레인저 인과검정(Granger Causality)
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
# MAGIC ## 정상성 시차 찾기
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
# MAGIC ### collectible_avgusd & game_avgusd
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
# MAGIC ### all_avgusd & all_buyers
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
# MAGIC ## 그레인저 인과분석
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
# MAGIC ### collectible_avgusd & game_avgusd
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
# MAGIC ### all_avgusd & all_buyers
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
# MAGIC ### all_avgusd & collectible_avgusd
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
# MAGIC ### all_buyers & collectible_avgusd
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
# MAGIC ### all_avgusd & game_avgusd
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
# MAGIC ### all_buyers & game_avgusd
# MAGIC - buyers -> game  6~14 귀무가설 기각, 6기준 fstats 4.3648
# MAGIC - game -> buyers 4~15 귀무가설 기각, 6기준 fstats 39.6156

# COMMAND ----------

# MAGIC %md
# MAGIC ### 제3의 외부 변수 추가
# MAGIC - 가격 형성 요인으로 외부 이슈(언론, 홍보, 커뮤니티) 요인으로 추정됨
# MAGIC - 커뮤니티 데이터(ex: nft tweet)를 구하지 못해 포털 검색 데이터(rate, per week)를 대안으로 분석해보자

# COMMAND ----------

# MAGIC %md
# MAGIC #### 미니 EDA
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
# MAGIC #### 미니 시각화
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
# MAGIC #### 데이터 통합

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
# MAGIC #### 미니 상관분석
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
# MAGIC #### 시차상관분석

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
# MAGIC ##### nft&cau&gau 시차상관분석(2021년 이후 월중앙값 & 주간)
# MAGIC - nftgt & nftgt : +- 3개월 정도 상관성이 있음
# MAGIC - nftgt & cau : -2개월부터 상관성이 높고, +는 매우 높음, nftgt -> cau관계로 추정
# MAGIC - nftgt & gau : 1부터 상관성이 높음으나 cau에 상대적으로 낮음 nftgt -> cau관계로 추정
# MAGIC - nftgt & au : 0부터 높음, nftgt -> au관계로 추정
# MAGIC - nftgt & ub : -2 ~ 0높았은데, 1~2에 잠시 하락했다가 급등, ub->nftgt 관계인가? 뭐지??

# COMMAND ----------

# MAGIC %md
# MAGIC #### 공적분 검정
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
# MAGIC ##### (단변량) 앵글&그레인저 2단계 결과(주간)
# MAGIC - nftgt & cau : ctt기준 pval  0.3798로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음**
# MAGIC - nftgt & ub : ctt기준 pval 0.4232 로 0.05를 초과하여 귀무가설을 채택하여 **공적분관계 없음** 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 그레인저 인과검정
# MAGIC - nftgt 와 시차상관성이 높은 cau와 aub만 대표로 보자
# MAGIC - 월중앙값으로 보면 인과검정 실패하여, 주간으로 다시 봄

# COMMAND ----------

# MAGIC %md
# MAGIC ##### nft_gt & cau(주간)

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
# MAGIC ##### nft_gt & gau(주간)

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
# MAGIC ##### nft_gt & aau(주간)

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
# MAGIC ##### nft_gt & ub(주간)

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
# MAGIC ##### 외부변수 인과검정 결과
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
# MAGIC ### <인과검정 결과종합>
# MAGIC - [도표 문서](https://docs.google.com/presentation/d/1_XOsoLV95qqUwJI8kxFXS_7NUIQbp872UHT_cQ162Us/edit#slide=id.g122453ac673_0_0)
# MAGIC - 1) game → buyers/collectible
# MAGIC - 2) buyers → collectible
# MAGIC - 3) collectible →all
# MAGIC - 4) all → buyers 
# MAGIC - 결과적으로 다변량 시계열분석은.. 어떤 변수로 무엇을 예측해야할까?

# COMMAND ----------

# MAGIC %md
# MAGIC # 다변량 시계열 분석 (Pass)
# MAGIC - 시간 부족으로 보류...공부량이 상당한 부분이므로 다음에..
# MAGIC - 공적분 미존재시 VAR -> 요한슨검정 -> 공적분 존재시 VECM

# COMMAND ----------

# MAGIC %md
# MAGIC ## 공적분 미존재시 VAR(벡터자기회귀모형)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 공적분 검정(Johansen Test)
# MAGIC - VAR모형에 대한 가설검정을 통해 적분계열간 안정적인 장기균형관계가 존재하는지 점검하는 방법
# MAGIC - 3개 이상의 불안정 시계열 사이의 공적분 검정에 한계를 갖는 앵글&그렌저 검정 방법을 개선하여 다변량에도 공적분 검정을 할 수 있음
# MAGIC - statsmodels.tsa.vector_ar.vecm. coint_johansen 
# MAGIC   - VECM의 공적분 순위에 대한 요한센 공적분 검정
# MAGIC   - [signature](https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html)

# COMMAND ----------

from statsmodels.tsa.vector_ar.vecm import coint_johansen

# COMMAND ----------

X = data[avgusd_col_list]
X.head()

# COMMAND ----------

jresult = coint_johansen(X, det_order=0, k_ar_diff=1)
jresult.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 공적분 존재시 VECM(벡터오차수정모형)
# MAGIC - 불안정시계열X와 Y를 1차 차분한 변수를 이용하여 회귀분석을 수행함으로써 전통적 방법의 사용으로 인해 야기되는 문제점들을 어느정도 해결할 수 있으나, 두 변수 같의 장기적 관계에 대한 소중한 정보를 상실하게 된다.
# MAGIC - 이 경우 만일 두 변수 간에 공적분이 존재한다면 오차수정모형(error correction model)을 통해 변수들의 단기적 변동뿐만 아니라 장기균형관계에 대한 특성을 알 수 있게 된다.
# MAGIC - VECM은 오차수정모형(ECM)에 벡터자기회귀모형(VAR)과 같은 다인자 모형 개념을 추가 한 것
# MAGIC - [VECM 예제](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=gush14&logNo=120145414589)
# MAGIC - [파이썬 예제](http://incredible.ai/trading/2021/07/01/Pair-Trading/)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 충격반응분석

# COMMAND ----------


