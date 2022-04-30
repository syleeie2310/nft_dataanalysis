# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from warnings import filterwarnings
filterwarnings("ignore")
plt.style.use("ggplot")
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.precision', 2) # 소수점 글로벌 설정

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

dataW_median = data.resample('W').median()
dataW_median.tail()

# COMMAND ----------

# MAGIC %md
# MAGIC # 제3의 외부 변수 추가
# MAGIC - 6번CCA노트북에서 다수의 상호지연관계가 확인되어, 그레인저인과검정을 위해 **"제3의 외부변수"**를 추가한다. 
# MAGIC - 가격 형성 요인으로 외부 이슈(언론, 홍보, 커뮤니티) 요인으로 추정됨
# MAGIC - 커뮤니티 데이터(ex: nft tweet)를 구하지 못해 포털 검색 데이터(rate, per week)를 대안으로 분석해보자

# COMMAND ----------

# MAGIC %md
# MAGIC ## 미니 EDA
# MAGIC - 주단위 수치형 "비율" 데이터
# MAGIC - 1%미만은 1으로 사전에 변경

# COMMAND ----------

gt_data = pd.read_csv('/dbfs/FileStore/nft/google_trend/nft_googletrend_w_170423_220423.csv', index_col = "Date", parse_dates=True, thousands=',')

# COMMAND ----------

gt_data.tail()

# COMMAND ----------

gt_data.info()

# COMMAND ----------

gt_data.rename(columns={'nft':'nft_gt'}, inplace=True)
gt_data.describe()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 미니 시각화
# MAGIC - 분포 : 1이 77%
# MAGIC - 추세 : 2021년 1월부터 급등해서 6월까라 급락했다가 22년1월까지 다시 급등 이후 하락세
# MAGIC - 범위 : 21년도 이후 iqr범위는 10~40, 중위값은 약25, 최대 약 85, 

# COMMAND ----------

plt.figure(figsize=(30,5))

plt.subplot(1, 2, 1)   
plt.title('<Weekly(%) Distribution>', fontsize=22)
plt.hist(gt_data['2018':])

plt.subplot(1, 2, 2)   
plt.title('<Weekly(%) Trend>', fontsize=22)
plt.plot(gt_data['2018':])

plt.show()

# COMMAND ----------

gt2021 = gt_data['2021':].copy()
gt2021['index'] = gt2021.index
gt2021s = gt2021.squeeze()
gt2021s['monthly']= gt2021s['index'].dt.strftime('%Y-%m')
gt2021.set_index(keys=gt2021['monthly'])

# COMMAND ----------

ax = gt2021.boxplot(column = 'nft_gt', by='monthly', figsize=(30,5), patch_artist=True)
ax.get_figure().suptitle('')
ax.set_xlabel('')
plt.title('<monthly(%, median) IQR Distribution>', fontsize=22)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 데이터 통합

# COMMAND ----------

# 마켓데이터 주간 집계
marketW = data['2018':].resample('W').median()
marketW.tail()

# COMMAND ----------

# gt데이터 길이 인덱스 확인
gt_data['2018':'2022-02-20'].tail()

# COMMAND ----------

# 주간 데이터 통합
totalW = pd.merge(marketW, gt_data, left_index=True, right_index=True, how='left')
totalW.tail()

# COMMAND ----------

# 정규화
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
totalW_scaled = totalW.copy()
totalW_scaled.iloc[:,:] = minmax_scaler.fit_transform(totalW_scaled)
totalW_scaled.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # 상관분석
# MAGIC - 확인결과 raw데이터와 스케일링 정규데이터와 결과 동일, raw데이터로 보면됨, 월간과 주간 차이 없음

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

# 주간 2018년 이후 데이터 : nft_gt도 두루 상관성이 높음(인과분석가능) 
heatmapC(totalW, 'all')

# COMMAND ----------

# 주간 2021년 이후 데이터 : gt데이터가 급등한 21년도부터 상관성이 분명해짐,  avg_usd의 상관성이 약해졌으나 가격류는 유지됨 
heatmapC(totalW['2021':], 'all')

# COMMAND ----------

# [함수] 피처별 히트맵 생성기
import plotly.figure_factory as ff

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

# 주간 2018년 이후 데이터 : utility제외하고 gt와 대체로 상관관계가 높음(collectible가 가장 높음)  하지만 nft_gt데이터가 2018~2020까지 모두 1이라서 어뷰징이 있음
heatmapF(totalW, 'average_usd')

# COMMAND ----------

heatmapF(totalW['2019':], 'average_usd')

# COMMAND ----------

# 주간 2021년 이후 데이터 : nft검색량이 급등한 21년도부터 차이가 분명하다, utility의 상관성이 다시 높아진것에 반면 defi는 낮아짐.
# nft_gt기준 metaverse, collectible, art 순으로 상관성이 가장 높다.
heatmapF(totalW['2021':], 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 상관분석 결과
# MAGIC - 21년도 이후부터 분석하면 될듯
# MAGIC ---
# MAGIC ### all카테고리, 피처별 상관관계
# MAGIC - 주간 2021년 이후 데이터 : gt데이터가 급등한 21년도부터 상관성이 분명해짐,  avg_usd의 상관성이 약해졌으나 가격류는 유지됨
# MAGIC - **분석 피처 셀렉션 : 총매출, 총판매수, 총사용자수, 총평균가**
# MAGIC   - 상관성이 높고 시장흐름을 이해할 수 있는 주요 피처를 선정
# MAGIC ---
# MAGIC ### avgusd피처, 카테고리별 상관관계
# MAGIC - 주간 2021년 이후 데이터 : nft검색량이 급등한 21년도부터 차이가 분명하다, utility의 상관성이 다시 높아진것에 반면 defi는 낮아짐. nft_gt기준 metaverse, collectible, art 순으로 상관성이 가장 높다.
# MAGIC - **분석 피처 셀렉션 : metaverse, collectible, art, game**
# MAGIC   - 위에서 선정한 주요피처중에 가장 상관성이 낮아 해석이 용이할 것으로 추정되는 avgusd를 기준으로 다시 상관성과 매출 비중이 높은 주요 카테고리로 선정한다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 결과종합&1차셀렉션
# MAGIC - nft_18년도 이후부터 마켓변수들과 모두 상관성이 있는데, 그중 본격적으로 검색량이 발생하는 21년도부터 각 마켓변수들과의 상관성이 분명함,
# MAGIC   - all카테고리, 세일즈피처별 상관관계 : 분석용 변수 1차 셀렉션 (총매출, 총판매수, 총사용자수, 총평균가)
# MAGIC   - avgusd피처, 카테고리별 상관관계 : 분석용 변수 1차 셀렉션 (metaverse, collectible, art, game)

# COMMAND ----------

# MAGIC %md
# MAGIC # (study)Cross Correlation
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
# MAGIC ### 예제1 : statsmodel CCF
# MAGIC - adjusted (=unbiased): 참이면 교차 상관의 분모는 nk이고 그렇지 않으면 n입니다.
# MAGIC   - 편향되지 않은 것이 참이면 자기공분산의 분모가 조정되지만 자기상관은 편향되지 않은 추정량이 아닙니다.
# MAGIC - fft : True이면 FFT 컨볼루션을 사용합니다. 이 방법은 긴 시계열에 대해 선호되어야 합니다.

# COMMAND ----------

#define data 
marketing = np.array([3, 4, 5, 5, 7, 9, 13, 15, 12, 10, 8, 8])
revenue = np.array([21, 19, 22, 24, 25, 29, 30, 34, 37, 40, 35, 30]) 
import statsmodels.api as sm
#calculate cross correlation
sm.tsa.stattools.ccf(marketing, revenue, adjusted=False)
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
# MAGIC ### CCF-CC 교차상관계수(Cross Correlation)
# MAGIC - avgusd 카테고리별 비교, 시가총액과 비교
# MAGIC - 변수간 동행성(comovement) 측정
# MAGIC - 경기순응적(pro-cyclical) / 경기중립적(a-cyclical) / 경기역행적(counter-cyclical)

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
# MAGIC #### 자기교차상관
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
# MAGIC #### 상호교차상관
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
# MAGIC ### CCF-LC 시차 상관계수(leads and lags correlation)

# COMMAND ----------

# MAGIC %md
# MAGIC #### avg_usd피처, 카테고리별 시차상관분석

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
# MAGIC ##### [실험 결과] avg_usd 카테고리별 시차상관분석
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
# MAGIC ##### 대표 케이스 시차상관계수 비교 테이블

# COMMAND ----------

avgusd_col_list

# COMMAND ----------

# 월 중앙값 집계 데이터
dataM_median = data.resample('M').median()
dataM_median.head()

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
    pd.set_option('display.precision', 2) # 소수점 글로벌 설정
    pd.set_option('display.float_format',  '{:.2f}'.format)
    dataframe = dataframe.style.bar(subset=['X2표준편차','X2상대표준편차'])\
    .background_gradient(subset=[*dataframe.columns[4:]], cmap='Blues', vmin = 0.5, vmax = 0.9)\
    .set_caption(f"<b><<< X1변수({dataframe['X1변수'][0]})기준 X2의 시차상관계수'>>><b>")\
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
# gmae이 생각보다 상관이 낮게 나왔다. game데이터는 2017년 데이터 없으므로, 2018년 이후 데이터로 다시 해보자
visualDF(result_df) 

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
# MAGIC ##### [결론] 월 중앙값 기준 시차상관분석(collectible_avgusd 기준)
# MAGIC - 2018년이후 데이터로 분석하니, 모든 카테고리 상관성이 높아졌다.(특히 과거 시차관련)
# MAGIC - collectible의 자기상관도는 매우 높으나 RSD 정밀도가 낮다.
# MAGIC - RSD(상대표준편차)는 metaverse가 상대적으로 정밀도가 높고, art와 all의 정밀도가 낮다.
# MAGIC - utility는 상관성이 없다.
# MAGIC - metaverse는 y변수가 음수 일 때 상관성이 매우 높으므로 X가 후행한다. metaverse -> collectible  "매우 명확"
# MAGIC - all, art, game은 y변수가 양수일 때 상관성이 음수일 보다 상대적으로 더 높다.
# MAGIC   - 그런데 -2음수일때도 높은 것으로 보다 상호지연관계가 있으면서, 동시에 X의 선행 영향력이 더 크다. collectible <->> all/art/game(단 게임은 비교적 짧다)

# COMMAND ----------

# MAGIC %md
# MAGIC #### all카테고리, 피처별 시차상관분석

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
# MAGIC ##### [실험 결과] all카테고리 피처별 시차상관분석 
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
# MAGIC # 시차상관분석
# MAGIC #### CCF-LC 시차 상관계수(leads and lags correlation)
# MAGIC - 시차 상호 상관(TLCC) https://dive-into-ds.tistory.com/96
# MAGIC - 선행적(leading) / 동행적(coincident) / 후행적(lagging)

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
    .background_gradient(subset=[*dataframe.columns[4:]], cmap='Blues', vmin = 0.5, vmax = 0.9)\
    .set_caption(f"<b><<< X1변수({dataframe['X1변수'][0]})기준 X2의 시차상관계수'>>><b>")\
    .format(thousands=',')\
    .set_properties(
        **{'border': '1px black solid !important'})
    return dataframe

# COMMAND ----------

# nft_gt와 시차상관분석을 위한 피처리스트 -> 동일한 레벨끼리 교차분석하자.
all_flist = ['nft_gt',  'all_sales_usd',  'all_number_of_sales',  'all_active_market_wallets', 'all_average_usd']# 총매출, 총판매수, 총사용자수, 총평균가
avgusd_clist = ['nft_gt', 'game_average_usd', 'collectible_average_usd',  'art_average_usd', 'metaverse_average_usd'] # metaverse, collectible, art, game

# COMMAND ----------

# 주간 18년도 이후 데이터 기준
print(f"<<<X1기준 X2의 변동폭 및 시차상관계수 테이블>>>")
all_flist_result = TLCC_comparison_table(totalW, 'nft_gt', all_flist, -12, 12)
all_flist_result 

# COMMAND ----------

visualDF(all_flist_result)

# COMMAND ----------

# 주간 21년도 이후 데이터 기준
print(f"<<<X1기준 X2의 변동폭 및 시차상관계수 테이블>>>")
all_flist_result = TLCC_comparison_table(totalW['2021':], 'nft_gt', all_flist, -12, 12)
all_flist_result 

# COMMAND ----------

visualDF(all_flist_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## all_flist 결과(2021년 이후 주간기준)
# MAGIC - 정규화 전후결과 유사함, 앞으로 안봐도 될듯, 18년도는 티가 안나서 보기 어렵다. 21년도만 봐도 될듯.
# MAGIC - RSD(상대표준편차, 변동성CV)는 판매수와 평균가가 상대적으로 낮은 편
# MAGIC - nft_gt의 자기상관성은 12주 전후 모두 높은편.
# MAGIC - 매출은 12주 전후 모두 시차상관성이 높은데 그중 양수가 상대적으로 더 높다.
# MAGIC   - 상호지연관계가 있으면서 동시에 X1의 선행역향력이 더 크다 gt <->> 매출
# MAGIC - 판매수 역시 12주 전후 모두 시차상관선이 높지만, 상대적으로 음수가 더 높다.
# MAGIC   - 상호지연관계가 있으면서 동시에 X2의 선행영향력이 더 크다 gt <<-> 판매수
# MAGIC - 사용자수도 위와 상동, gt <<-> 사용자수
# MAGIC - 평균가는 분명하게 양수가 높은 것으로 보다 편지연관계로서 X1의 선행영향력만 존재한다. gt -> 평균가
# MAGIC - 특이사항 : 판매수와 사용자수는 5~8주 지점에서 소폭 감소하는 경향이 있다. 또다른 제3의 존재가 있는 듯(일단 pass)

# COMMAND ----------

# 주간 18년도 이후 데이터 기준
print(f"<<<X1기준 X2의 변동폭 및 시차상관계수 테이블>>>")
avgusd_clist_result = TLCC_comparison_table(totalW, 'nft_gt', avgusd_clist, -12, 12)
avgusd_clist_result 

# COMMAND ----------

visualDF(avgusd_clist_result)

# COMMAND ----------

# 주간 21년도 이후 데이터 기준
print(f"<<<X1기준 X2의 변동폭 및 시차상관계수 테이블>>>")
avgusd_clist_result = TLCC_comparison_table(totalW['2021':], 'nft_gt', avgusd_clist, -12, 12)
avgusd_clist_result 

# COMMAND ----------

visualDF(avgusd_clist_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## avgusd_clist 결과(2021년 이후 주간기준)
# MAGIC - 정규화 전후결과 유사함, 앞으로 안봐도 될듯, 18년도는 티가 안나서 보기 어렵다. 21년도만 봐도 될듯.
# MAGIC - RSD(상대표준편차, 변동성CV)는 game과 art가 상대적으로 낮은 편
# MAGIC - nft_gt의 자기상관성은 12주 전후 모두 높은편.
# MAGIC - game은 분명하게 양수가 높은 것으로 보아 편지연관계로서 X1의 선행영향력만 존재한다. gt -> game
# MAGIC - collectible은 12주 전후 모두 시차상관성이 높은데 그중 양수가 상대적으로 더 높다.
# MAGIC   - 상호지연관계이면서, 동시에 X1의 선행역향력이 더 크다 gt <->> collectible
# MAGIC - art 역시 전후 시차상관성이 존재하지만, 음수는 06부터 높으며 상대적으로 양수가 매우 높다.
# MAGIC   - 상호지연관계이면서, 동시에 X1의 선행영향력이 더 크고 긴데 반해 **X2의 선행영향력은비교적 짧다.** gt <->> art
# MAGIC - metaverse 역시 12주 전후 모두 시차상관성이 높은데, 그중 음수가 상대적으로 더 높다.
# MAGIC   - 상호지연관계이면서, 동시에 X2의 선행영향력이 더 크다 길다.  **X1의 선행영향력은비교적 짧다.** gt <<-> metaverse

# COMMAND ----------

# MAGIC %md
# MAGIC ## 결과종합
# MAGIC - 위와 동일하게 21년도 기준 상관성이 분명하게 드러남
# MAGIC - nft_gt와 기존 마켓변수들과 시차상관성을 확인, 상호지연관계로 또다른 제3의 변수가 있지만 시간관계상 pass
# MAGIC - 공적분 검정을 통해 셀렉션한 피처들을 좀더 줄여보자

# COMMAND ----------

# MAGIC %md
# MAGIC # 공적분 검정
# MAGIC - 앵글&그레인저, 주간데이터
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

print(len(all_flist), len(avgusd_clist))

# COMMAND ----------

for i in range(len(all_flist)):
    print(i, i/2, i%2+1, i//2)

# COMMAND ----------

# 장기적 관계 시각화를 위한 X2/X1 비율 그래프

def x1x2plot(data, x1, x2list):

    plt.figure(figsize=(30,10))

    ncols = len(x2list)//2
    nrows = ncols+1
    plt.suptitle('X2 / X1 Rate Line Plot', fontsize=40 )
    for i in range(len(x2list)):
        x2 = x2list[i]
        xrate = data[x2]/data[x1]
        
        plt.subplot(nrows, ncols, i+1)
        plt.plot(xrate)
        plt.axhline(xrate.mean(), color='red', linestyle='--')
        plt.title(f' [{i}] {x2} / {x1} Ratio', fontsize=22)
        plt.legend(f'[{i}] {x2} / {x1} Ratio', 'Mean')
        
    plt.tight_layout(h_pad=5, w_pad=5)
    plt.show()

# COMMAND ----------

# 기준이 될 nft_gt 추세 참고
totalW['nft_gt'].plot(figsize=(30,7))

# COMMAND ----------

# 21년도 이후를 보면 조금씩 연관성이 있어보인다. 직접 검정해보자.
x1x2plot(totalW, 'nft_gt', all_flist[1:])

# COMMAND ----------

# 공적분 검정 테이블 함수
import statsmodels.tsa.stattools as ts
def coint_test(data, x1, x2list, Trend):
   
    Coint_ADF_score = []
    Coint_ADF_Pval = []
    result = []
    x1list = []
    
    for x2 in x2list:
        x2data = data[x2]
        score, pvalue, _ = ts.coint(data[x1], x2data, trend = Trend)
        Coint_ADF_score.append(score)
        Coint_ADF_Pval.append(pvalue)
    
        if pvalue <= 0.05 :
            result.append('pass, go to VECM')
        else :
            result.append('fail, go to VAR')
            
        x1list.append(x1)
        
    result = pd.DataFrame(list(zip(x1list, x2list, Coint_ADF_score, Coint_ADF_Pval, result)), columns=[ 'x1', 'x2', 'Coint_ADF_score', 'Coint_ADF_Pval', 'Coint_result'])
    
    return result          

# COMMAND ----------

# 위 그래프를 볼때 2차 기울기 추세임을 알 수 있다. CTT
# 2018년도 이후 주간데이터 기준, avgusd외에 모두 nft_gt와 장기적 연관성이 있다.
coint_test(totalW, 'nft_gt', all_flist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# 2019년도 이후 주간데이터 기준, 장기적연관성 있음
coint_test(totalW['2019':], 'nft_gt', all_flist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# 2020년도 이후 주간데이터 기준, 모두 없음
coint_test(totalW['2020':], 'nft_gt', all_flist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# 2021년도 이후 주간데이터 기준, nftgt와 모두 장기적 연관성이 없다. -> 기간이 너무 짧은 듯, nft_gt검색량이 급등 전후의 데이터가 충분히 반영되지 않은 듯
coint_test(totalW['2021'], 'nft_gt', all_flist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# MAGIC %md
# MAGIC ## all_flist 결과(CTT기준)
# MAGIC - 2018년도, 2019년도 이후 : 매출, 판매수, 사용자수 모두 장기적 연관성이 있다.
# MAGIC - 2020년도, 2021년도 이후 : 모두 없음

# COMMAND ----------

# 21년도 이후를 보면 1번 2번이 있어보인다. 직접 검정해보자.
x1x2plot(totalW, 'nft_gt', avgusd_clist[1:])

# COMMAND ----------

# 위 그래프를 볼때 2차 기울기 추세임을 알 수 있다. CTT
# 2018년도 이후 주간데이터 기준, art외에 모두 nft_gt와 장기적 연관성이 있다.
coint_test(totalW, 'nft_gt', avgusd_clist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# 2019년도 이후 주간데이터 기준, 일부 있음
coint_test(totalW['2019':], 'nft_gt', avgusd_clist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# 2020년도 이후 주간데이터 기준, 모두 없음
coint_test(totalW['2020':], 'nft_gt', avgusd_clist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# 2021년도 이후 주간데이터 기준, 모두 없음
coint_test(totalW['2021':], 'nft_gt', avgusd_clist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# MAGIC %md
# MAGIC ## avgusd_clist 결과(CTT기준)
# MAGIC - 2018년도 이후 : game, collectible, metaverse 모두 장기적 연관성이 있다. (art도 있긴하다. pval값이 조금 아쉬움)
# MAGIC - 2019년도 이후 : collectible, metavers 있음
# MAGIC - 2020년도, 21년도 이후 : 모두 없음

# COMMAND ----------

# MAGIC %md
# MAGIC ## 결과종합&2차셀렉션
# MAGIC - 이 경우는 오히려 2018년도 및 2019년도까지 과거데이터를 포함해야 장기적 관계를 확인할 수 있다.(검색량 유, 무 정보 영향 추정)
# MAGIC   - all카테고리, 세일즈피처별 공적분 검정 : 평균가는 장기적 관계 없음
# MAGIC   - avgusd피처, 카테고리별 공적분 검정 : art의 pval값이 근소함을 고려시 모두 장기적관계가 있다.
# MAGIC ---
# MAGIC ### 정상성 검정을 위한 2차 피처 셀렉션
# MAGIC - 외부 변수 : nft 구글 검색량
# MAGIC - all카테고리, 세일즈 변수 :   총매출, 총판매수, 총사용자수, 깍두기(평균가-참고용)
# MAGIC - avgusd피처, 카테고리별 변수 : metaverse(참고용), collectible, art, game
# MAGIC - 데이터 기간 : 특징은 21년도부터 드러나지만 장기적연관성을 위해 과거데이터도 일부 필요하다. 기준잡아야함

# COMMAND ----------

# MAGIC %md
# MAGIC # 정상성 검정
# MAGIC ## 2번 노트북(TSA)에서 전체 변수들 모두 정상성 시차 1을 확인
# MAGIC - 데이터 : 주간 (일/주/월 결과 모두 다름, 외부변수와 함께 비교하려면 주간통일이 편함), 정규화 안함(데이터 특징 우려, )
# MAGIC - 대표칼럼 : avgusd 피처, 카테고리별 검정
# MAGIC - 대표칼럼 : all카테고리, 피처별 검정
# MAGIC ---
# MAGIC ## 1. Augmented Dickey-Fuller("ADF") Test
# MAGIC - ADF 테스트는 시계열이 안정적(Stationary)인지 여부를 확인하는데 이용되는 방법입니다.
# MAGIC - 시계열에 단위근이 존재하는지 검정,단위근이 존재하면 정상성 시계열이 아님.
# MAGIC - 귀무가설이 단위근이 존재한다.
# MAGIC - 검증 조건 ( p-value : 5%이내면 reject으로 대체가설 선택됨 )
# MAGIC - 귀무가설(H0): non-stationary. 대체가설 (H1): stationary.
# MAGIC - adf 작을 수록 귀무가설을 기각시킬 확률이 높다.
# MAGIC 
# MAGIC ## 2. Kwiatkowski-Phillips-Schmidt-Shin (“KPSS”) Test
# MAGIC - [KPSS 시그니처](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.kpss.html)
# MAGIC - KPSS 검정은 시계열이 평균 또는 선형 추세 주변에 고정되어 있는지 또는 단위 루트(unit root)로 인해 고정되지 않은지 확인합니다.
# MAGIC - KPSS 검정은 1종 오류의 발생가능성을 제거한 단위근 검정 방법이다.
# MAGIC - DF 검정, ADF 검정과 PP 검정의 귀무가설은 단위근이 존재한다는 것이나, KPSS 검정의 귀무가설은 정상 과정 (stationary process)으로 검정 결과의 해석 시 유의할 필요가 있다.
# MAGIC   - 귀무가설이 단위근이 존재하지 않는다.
# MAGIC - 단위근 검정과 정상성 검정을 모두 수행함으로서 정상 시계열, 단위근 시계열, 또 확실히 식별하기 어려운 시계열을 구분하였다.
# MAGIC - KPSS 검정은 단위근의 부재가 정상성 여부에 대한 근거가 되지 못하며 대립가설이 채택되면 그 시계열은 trend-stationarity(추세를 제거하면 정상성이 되는 시계열)을 가진다고 할 수 있습니다.
# MAGIC - 때문에 KPSS 검정은 단위근을 가지지 않고 Trend- stationary인 시계열은 비정상 시계열이라고 판단할 수 있습니다.

# COMMAND ----------

# 피처 분류기
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

# 카테고리 분류기
def category_classifier(data, category):
    col_list = []
    for i in range(len(data.columns)):
        if data.columns[i].split('_')[0] == category:
            col_list.append(data.columns[i])
        else :
            pass
    return col_list

# COMMAND ----------

# adf 검정
from statsmodels.tsa.stattools import adfuller

def adf_test(data):
#     print("Results of ADF Test")
    result = adfuller(data)
#     print('ADF Statistics: %f' % result[0])
#     print('p-value: %f' % result[1])
    return result
#     print('Critical values:')
#     for key, value in result[4].items():
#         print('\t%s: %.3f' % (key, value))

# COMMAND ----------

# KPSS 검정
from statsmodels.tsa.stattools import kpss

def kpss_test(data):
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

# 단위근 검정 실행기
pd.options.display.float_format = '{: .4f}'.format

def UnitRootTest(data, col_list) :
        
    adf_stats = []
    adf_Pval = []
    kpss_stats = []
    kpss_Pval = []
    result_list = []
    
    for col in col_list:
        col_data = data[col]
        
        # ADF검정기 호출
        adf_result = adf_test(col_data) 
        adf_stats.append(adf_result[0])
        adf_Pval.append(adf_result[1])
        
        # KPSS검정기 호출
        kpss_result = kpss_test(col_data)
        kpss_stats.append(kpss_result[0])
        kpss_Pval.append(kpss_result[1])
        
        # 종합
        if adf_result[1] <= 0.05 and kpss_result[1] >= 0.05:
            result_list.append('ALL Pass')
        elif adf_result[1] <= 0.05 or kpss_result[1] >= 0.05:
            if adf_result[1] <= 0.05:
                result_list.append('ADF Pass')
            else:
                result_list.append('KPSS Pass')
        else :
            result_list.append('fail')
        
    result_df = pd.DataFrame(list(zip(adf_stats, adf_Pval, kpss_stats, kpss_Pval, result_list)), index = col_list, columns=['adf_stats', 'adf_Pval', 'KPSS_stats', 'KPSS_Pval', 'result'])

    return result_df             

# COMMAND ----------

# 정상성 검정을 위해 데이터 차분(1)
totalW_diff = totalW.diff(periods=1).dropna()

# COMMAND ----------

# nft_gt 차분 그래프
totalW_diff['nft_gt'].plot(figsize=(30, 5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## all_flist 결과
# MAGIC - 직전 공적분 검정으로 18년도 19년도에 장기연관성이 있었다.

# COMMAND ----------

# 18년도 이후 차분데이터
UnitRootTest(totalW_diff, all_flist)

# COMMAND ----------

# 19년도 이후 차분데이터
UnitRootTest(totalW_diff['2019':], all_flist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## avgusd_clist 결과
# MAGIC - 직전 공적분 검정으로 18년도 19년도(일부)에 장기연관성이 있었다.

# COMMAND ----------

# 18년도 이후 차분데이터
UnitRootTest(totalW_diff, avgusd_clist)

# COMMAND ----------

# 19년도 이후 차분데이터
UnitRootTest(totalW_diff['2019':], avgusd_clist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 결과종합&3차셀렉션
# MAGIC #### 외부변수 검정 결과
# MAGIC - nft_gt 통과
# MAGIC 
# MAGIC #### all_flist 검정 결과
# MAGIC - 테스트 변수 : 총매출, 총판매수, 총사용자수, 총평균가
# MAGIC - 19년도기준, 매출 제외하고 모두 통과
# MAGIC 
# MAGIC #### avgusd_clist 검정 결과
# MAGIC - 테스트 변수 : metaverse, collectible, art, game
# MAGIC - 19년도기준, metaverse 제외하고 모두 통과
# MAGIC 
# MAGIC ---
# MAGIC #### 그레인저 인과검정을 위한 최종 피처 셀렉션
# MAGIC - 외부 변수 : nft 구글 검색량
# MAGIC - all카테고리, 세일즈 변수 : 총판매수, 총사용자수, 평균가(참고용)
# MAGIC - avgusd피처, 카테고리별 변수 : collectible, art, game
# MAGIC - 데이터 기간 : 2019년도 이후, 주간

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
# MAGIC ---
# MAGIC - 딕셔너리 언패킹을 못해서 시각화못함
# MAGIC - from statsmodels.tsa.stattools import grangercausalitytests [signature](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html)
# MAGIC   - 2개 시계열의 그랜저 비인과성에 대한 4가지 테스트.
# MAGIC   - 현재 일간데이터 길이 기준 maxlag = 15가 최대
# MAGIC   - 2번째 시계열이 1번째 시계열을 유발하는지 테스트(2->1) -> 즉 2번째열이 시차 보행하는 것
# MAGIC     - 그런데 lag를 양수만 입력가능하므로, 이는 X2의 과거lag값임.
# MAGIC     - 결국 X2의 t가 -n일 때의 X1회귀값의 pvalue, 즉, X2의 과거가 x1의 현재값을 통계적으로 유의미하게를 예측할 수 있는지를 본다

# COMMAND ----------

from statsmodels.tsa.stattools import grangercausalitytests

# COMMAND ----------

# MAGIC %md
# MAGIC ## all_flist 인과검정
# MAGIC - 외부 변수 : nft 구글 검색량
# MAGIC - all카테고리, 세일즈 변수 : 총판매수, 총사용자수, 평균가(참고용)
# MAGIC - 데이터 기간 : 2019년도 이후, 주간

# COMMAND ----------

# MAGIC %md
# MAGIC ### 총평균가 기준
# MAGIC - 경우의 수 : 3 X 2 = 6

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> nft_gt

# COMMAND ----------

# 총평균가->nft_gt    3,4,5,7~12 귀무가설 기각  f통계량(3기준= 3.69 , 7기준 = 4.79 , 12기준= 14.18 )
grangercausalitytests(totalW_diff[['nft_gt', 'all_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# nft_gt -> 총평균가 3,6~12 귀무가설 기각  f통계량(3기준= 3.35, 7기준= 2.79, 12기준= 3.25 )
grangercausalitytests(totalW_diff[['all_average_usd', 'nft_gt']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> 총사용자수

# COMMAND ----------

# 총평균가 -> 총사용자수    3~12 귀무가설 기각  f통계량(3기준= 2.78 , 12기준=78.16  )
grangercausalitytests(totalW_diff[['all_active_market_wallets', 'all_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# 총사용자수 -> 총평균가   3~12 귀무가설 기각  f통계량(3기준= 2.80 , 12기준= 21.17)
grangercausalitytests(totalW_diff[['all_average_usd', 'all_active_market_wallets']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> 총판매수

# COMMAND ----------

# 총평균가 -> 총판매수 : 약3~12 귀무가설 기각  f통계량(4기준= 6.20 , 12기준= 95.48 )
grangercausalitytests(totalW_diff[['all_number_of_sales', 'all_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# 총판매수 -> 총평균가 : 3~12 귀무가설 기각  f통계량(4기준= 6.50 , 12기준= 21.28 )
grangercausalitytests(totalW_diff[['all_average_usd', 'all_number_of_sales']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 총판매수 기준
# MAGIC - 경우의 수 : 2*2 = 4

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> nft_gt

# COMMAND ----------

# 총판매수 -> nft_gt : 3~12 귀무가설 기각  f통계량(3기준=  8.96, 12기준= 14.76 )
grangercausalitytests(totalW_diff[['nft_gt', 'all_number_of_sales']]['2019':], maxlag=12)

# COMMAND ----------

# nft_gt -> 총판매수 : 3~12 귀무가설 기각  f통계량(3기준= 6.36 , 12기준= 4.55 )
grangercausalitytests(totalW_diff[['all_number_of_sales', 'nft_gt']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> 총사용자수

# COMMAND ----------

# 총판매수 -> 총사용자수 : 7~12 귀무가설 기각  f통계량(7기준= 3.13 , 12기준=  39.7119)
grangercausalitytests(totalW_diff[['all_active_market_wallets', 'all_number_of_sales']]['2019':], maxlag=12)

# COMMAND ----------

# 총사용자수 ->총판매수 : 6~12 귀무가설 기각  f통계량(7기준= 3.24 , 12기준=  32.0442)
grangercausalitytests(totalW_diff[['all_number_of_sales', 'all_active_market_wallets']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 총사용자수 기준
# MAGIC - 경우의 수 : 1*2 = 2

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> nft_gt

# COMMAND ----------

# 총사용자수 -> nft_gt : 3~12 귀무가설 기각, f통계량(3기준=9.45 12기준=14.7262)
grangercausalitytests(totalW_diff[['nft_gt', 'all_active_market_wallets']]['2019':], maxlag=12)

# COMMAND ----------

# nft_gt -> 총사용자수 : 3~12 귀무가설 기각, f통계량(3기준=7.07 12기준=5.09) 
grangercausalitytests(totalW_diff[['all_active_market_wallets', 'nft_gt']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC ### <검정결과> all_flist
# MAGIC - 총판매수 <<-> nft_gt : 3~12 상호지연관계, 3기준 f통계량이 더 작음 관계(nft_gt -> 총판매수)
# MAGIC   - 총판매수 -> nft_gt : 3~12 귀무가설 기각  f통계량(3기준=  8.96, 12기준= 14.76 )
# MAGIC   - nft_gt -> 총판매수 : 3~12 귀무가설 기각  f통계량(3기준= 6.36 , 12기준= 4.55 )
# MAGIC   
# MAGIC - 총평균가 <<-> nft_gt : 약 3,7~12 상호지연관계, 3기준 f통계량이 더 작은 관계(nft_gt -> 총평균가)
# MAGIC   - 총평균가->nft_gt    3,4,5,7~12 귀무가설 기각  f통계량(3기준= 3.69 , 7기준 = 4.79 , 12기준= 14.18 )
# MAGIC   - nft_gt -> 총평균가 3,6~12 귀무가설 기각  f통계량(3기준= 3.35, 7기준= 2.79, 12기준= 3.25 )
# MAGIC 
# MAGIC - 총유저수 <<-> nft_gt : 3~12 상호지연관계, 3기준 f통계량이 더 작은 관계(nft_gt -> 총유저수)
# MAGIC   - 총사용자수 -> nft_gt : 3~12 귀무가설 기각, f통계량(3기준=9.45 12기준=14.7262)
# MAGIC   - nft_gt -> 총사용자수 : 3~12 귀무가설 기각, f통계량(3기준=7.07 12기준=5.09) 
# MAGIC 
# MAGIC - 총평균가 <->> 총유저수 : 3~12 상호지연관계, 3기준 f통계량이 더 작은 관계(총평균가 -> 총유저수)
# MAGIC   - 총평균가 -> 총사용자수    3~12 귀무가설 기각  f통계량(3기준= 2.78 , 12기준=78.16  )
# MAGIC   - 총사용자수 -> 총평균가   3~12 귀무가설 기각  f통계량(3기준= 2.80 , 12기준= 21.17)
# MAGIC 
# MAGIC - 총평균가 <-> 총판매수 : 판매수가 3으로 먼저 시작하고, 4~12 상호지연관계일때 4기준 f통계량이 더 작음 관계(총평균가 -> 총판매수)
# MAGIC   - 총평균가 -> 총판매수 : 4~12 귀무가설 기각  f통계량(4기준= 6.20 , 12기준= 95.48 )
# MAGIC   - 총판매수 -> 총평균가 : 3~12 귀무가설 기각  f통계량(4기준= 6.50 , 12기준= 21.28 )
# MAGIC 
# MAGIC - 총판매수 <-> 총유저수 : 유저수가 6으로 먼저 시작하고 약 7~12 상호지연관계일때 7기준 f통계량이 더 작은 관계(총판매수 -> 총유저수)
# MAGIC   - 총판매수 -> 총사용자수 : 7~12 귀무가설 기각  f통계량(7기준= 3.13 , 12기준=  39.7119)
# MAGIC   - 총사용자수 ->총판매수 : 6~12 귀무가설 기각  f통계량(7기준= 3.24 , 12기준=  32.0442)
# MAGIC 
# MAGIC ---
# MAGIC #### 요약 
# MAGIC - 흐름 : 대중의 관심(언론 등) -> 판매 활성화 -> 평균가 영향 -> 사용자 관심 -> 사용자 유입 -> 판매 증대(반복)
# MAGIC   - 3주차 : nft_gt <->> (총판매수 -> 총평균가 -> 유저수)
# MAGIC   - 4주차 이후 : 총평균가 -> 총판매수   
# MAGIC   - 6주차 이후 : 총유저수 -> 총판매수
# MAGIC   - 7주차 이후 : 총판매수 -> 총유저수
# MAGIC - 해석 : 유저가 바로 유입되지 않는다. 3주차부터 시장전체 영향을 받기 시작하며 6주차 부터 본격적으로 유저수 유입으로 인해 판매수에 영향을 주게 된다. 이후 7주부터 재귀적인 관계 돌입

# COMMAND ----------

# MAGIC %md
# MAGIC ## avgusd_clist 인과검정
# MAGIC - 외부 변수 : nft 구글 검색량
# MAGIC - avgusd피처, 카테고리별 변수 : collectible, art, game
# MAGIC - 데이터 기간 : 2019년도 이후, 주간

# COMMAND ----------

# MAGIC %md
# MAGIC ### Game 기준

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> nft_gt

# COMMAND ----------

# game -> nft_gt : 3~12 귀무가설 기각, f통계량(3기준=  3.94   12기준=  2.74 )
grangercausalitytests(totalW_diff[['nft_gt', 'game_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# nft_gt -> game : 3~12 귀무가설 기각, f통계량(3기준=  4.65   12기준= 2.77  )
grangercausalitytests(totalW_diff[['game_average_usd', 'nft_gt']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> Collectible

# COMMAND ----------

# game -> Collectible : 3~12 귀무가설 기각, f통계량(3기준= 5.03    7기준= 16.62  )
grangercausalitytests(totalW_diff[['collectible_average_usd', 'game_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# Collectible -> game : 2~7,9 귀무가설 기각, f통계량(3기준= 6.68    7기준= 2.19  )
grangercausalitytests(totalW_diff[['game_average_usd', 'collectible_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> Art

# COMMAND ----------

# game -> Art :  3~12  귀무가설 기각, f통계량(3기준= 6.26    12기준=  12.35 )
grangercausalitytests(totalW_diff[['art_average_usd', 'game_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# Art -> game : 2~6, 12   귀무가설 기각, f통계량(3기준= 3.90     12기준= 1.83  )
grangercausalitytests(totalW_diff[['game_average_usd', 'art_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Collectible 기준

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> nft_gt

# COMMAND ----------

# Collectible -> nft_gt : 8,10, 12   귀무가설 기각, f통계량(8기준= 3.01    12기준= 3.24  )
grangercausalitytests(totalW_diff[['nft_gt', 'collectible_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# nft_gt -> Collectible : 1~12 귀무가설 기각, f통계량(8기준=  2.72   12기준=  2.27 )
grangercausalitytests(totalW_diff[['collectible_average_usd', 'nft_gt']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> Art

# COMMAND ----------

# Collectible -> art : 1~12   귀무가설 기각, f통계량(1기준= 7.59    12기준=  8.64 )
grangercausalitytests(totalW_diff[['art_average_usd', 'collectible_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# art -> Collectible : 1~12   귀무가설 기각, f통계량(1기준= 48.19    12기준= 14.79  )
grangercausalitytests(totalW_diff[['collectible_average_usd', 'art_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Art 기준

# COMMAND ----------

# MAGIC %md
# MAGIC #### <-> nft_gt

# COMMAND ----------

# art -> nft_gt : 12   귀무가설 기각, f통계량(12기준= 2.99  )
grangercausalitytests(totalW_diff[['nft_gt', 'art_average_usd']]['2019':], maxlag=12)

# COMMAND ----------

# nft_gt -> art :  1~12  귀무가설 기각, f통계량(12기준=  1.97 )
grangercausalitytests(totalW_diff[['art_average_usd', 'nft_gt']]['2019':], maxlag=12)

# COMMAND ----------

# MAGIC %md
# MAGIC ### <검정결과> avgusd_clist
# MAGIC - game <->> nft_gt : 3~12 상호지연관계, 3기준 f통계량이 더 작은 관계(game -> nft_gt)
# MAGIC   - game -> nft_gt : 3~12 귀무가설 기각, f통계량(3기준=  3.94   12기준=  2.74 )
# MAGIC   - nft_gt -> game : 3~12 귀무가설 기각, f통계량(3기준=  4.65   12기준= 2.77  )
# MAGIC - game <->> collectible : collectible의 2부터 먼저 시작하지만, 3부터 상호지연관계일 경우(game->collectible)
# MAGIC   - game -> Collectible : 3~12 귀무가설 기각, f통계량(3기준= 5.03    7기준= 16.62  )
# MAGIC   - Collectible -> game : 2~7,9 귀무가설 기각, f통계량(3기준= 6.68    7기준= 2.19  )
# MAGIC - game <<-> Art : art가 2부터 먼저 시작하고, 3부터 상호지연관계일 경우(art->game)
# MAGIC   - game -> Art :  3~12  귀무가설 기각, f통계량(3기준= 6.26    12기준=  12.35 )
# MAGIC   - Art -> game : 2~6, 12   귀무가설 기각, f통계량(3기준= 3.90     12기준= 1.83  )
# MAGIC - collectible <<- nft_gt : nft_gt가 1부터 먼저 시작하고 상호지연관계일 때에도 nft_gt -> collectible
# MAGIC   - Collectible -> nft_gt : 8,10, 12   귀무가설 기각, f통계량(8기준= 3.01    12기준= 3.24  )
# MAGIC   - nft_gt -> Collectible : 1~12 귀무가설 기각, f통계량(8기준=  2.72   12기준=  2.27 )
# MAGIC - collectible <->> art : 1,~12 상호지연관계이고 1lag f통계량 기준 (collectible -> art)
# MAGIC   - Collectible -> art : 1~12   귀무가설 기각, f통계량(3기준= 7.59    12기준=  8.64 )
# MAGIC   - art -> Collectible : 1~12   귀무가설 기각, f통계량(3기준= 48.19    12기준= 14.79  )
# MAGIC - art <<- ngt_gt : nft_gt가 1부터 시작하고 f통계량 기준으로도 (nft_gt -> art)
# MAGIC   - art -> nft_gt : 12   귀무가설 기각, f통계량(12기준= 2.99  )
# MAGIC   - nft_gt -> art :  1~12  귀무가설 기각, f통계량(12기준=  1.97 )
# MAGIC   
# MAGIC ---
# MAGIC #### 요약
# MAGIC - 흐름 : 대중 관심 -> 주요 카테고리(art/collectible) 유입으로 평균가 영향 -> 주요카테고리 활성화 정도에 따른 게임 평균가 영향 -> 게임 활성화에 따른 대중의 관심 -> 반복 
# MAGIC   - 1주차 : nft_gt -> (art <<-> collectible)
# MAGIC   - 2주차 : art&collectible <->> game
# MAGIC   - 3주차 이후 : game <->> nft_gt
# MAGIC - 해석 : 대중의 관심이 1주만에 평균가에 빠른 영향을 주고, 3주부터 재귀적인 관계에 돌입한다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## <검정 결과 종합>
# MAGIC - [도표 문서](https://docs.google.com/presentation/d/1_XOsoLV95qqUwJI8kxFXS_7NUIQbp872UHT_cQ162Us/edit#slide=id.g122453ac673_0_0)
# MAGIC 
# MAGIC - avgusd_clist
# MAGIC   - 흐름 : 대중 관심 -> 주요 카테고리(art/collectible) 유입으로 평균가 영향 -> 주요카테고리 활성화 정도에 따른 게임 평균가 영향 -> 게임 활성화에 따른 대중의 관심 -> 반복 
# MAGIC     - 1주차 : nft_gt -> (art <<-> collectible)
# MAGIC     - 2주차 : art&collectible <->> game
# MAGIC     - 3주차 이후 : game <->> nft_gt
# MAGIC   - 해석 : 대중의 관심이 1주만에 평균가에 빠른 영향을 주고, 3주부터 재귀적인 관계에 돌입한다.
# MAGIC   
# MAGIC - all_flist
# MAGIC   - 흐름 : 대중의 관심(언론 등) -> 판매 활성화 -> 평균가 영향 -> 사용자 관심 -> 사용자 유입 -> 판매 증대(반복)
# MAGIC     - 3주차 : nft_gt <->> (총판매수 -> 총평균가 -> 유저수)
# MAGIC     - 4주차 이후 : 총평균가 -> 총판매수   
# MAGIC     - 6주차 이후 : 총유저수 -> 총판매수
# MAGIC     - 7주차 이후 : 총판매수 -> 총유저수
# MAGIC   - 해석 : 유저가 바로 유입되지 않는다. 3주차부터 시장전체 영향을 받기 시작하며 6주차 부터 본격적으로 유저수 유입으로 인해 판매수에 영향을 주게 된다. 이후 7주부터 재귀적인 관계 돌입
# MAGIC   
# MAGIC 
# MAGIC   
# MAGIC - 결과적으로 다변량 시계열분석은.. 어떤 변수로 무엇을 예측해야할까?

# COMMAND ----------

# MAGIC %md
# MAGIC # (보류)다변량 시계열 분석 (Pass)
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

# from statsmodels.tsa.vector_ar.vecm import coint_johansen

# COMMAND ----------

# X = data[avgusd_col_list]
# X.head()

# COMMAND ----------

# jresult = coint_johansen(X, det_order=0, k_ar_diff=1)
# jresult.

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
# MAGIC # (보류)충격반응분석

# COMMAND ----------


