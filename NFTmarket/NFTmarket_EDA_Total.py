# Databricks notebook source
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. 데이터 로드 및 통합
# MAGIC - https://nonfungible.com/market/history
# MAGIC - 차트별 ALL-Time Daily CSV파일 다운로드
# MAGIC - 2017.06.22 ~ 현재

# COMMAND ----------

# 70개 항목 파일명 리스트로 가져오기
import os
file_list = os.listdir('/dbfs/FileStore/nft/nft_market_220221')
len(file_list), print(file_list)

# COMMAND ----------

# 파일명 추출(확장자 제거)
file_name = []
for file in file_list:
    if file.count(".") == 1: 
        name = file.split('.')[0]
        file_name.append(name)

file_name

# COMMAND ----------

# csv파일 모두 판다스로 읽기
# 자동 변수 선언 + 변수 목록 생성
val_list = []
for file in file_name :
    globals()['data_{}'.format(file)] = 0
    val_list.append(f'data_{file}')
val_list

# COMMAND ----------

# 데이터셋들을 개별 데이터프레임변수로 생성하기& 칼럼명변경(카테고리 붙이기)
data_list = val_list.copy()
for i in range(len(file_name)):
    data_list[i] = pd.read_csv(f'/dbfs/FileStore/nft/nft_market_220221/{file_name[i]}.csv', index_col = "Date", parse_dates=True, thousands=',')
    data_list[i].columns = [file_name[i]] # 칼럼명 변경
    print(data_list[i])

# COMMAND ----------

total = data_list[0]
for i in range(1, len(data_list)):
    total = pd.merge(total, data_list[i], left_index=True, right_index=True, how='left')
total

# COMMAND ----------

total.info()

# COMMAND ----------

total.describe()

# COMMAND ----------

# 지수(e)출력 문제 해결
pd.set_option('float_format', '{:.1f}'.format) # 숫자 많아서 보기 어렵다. 소수점은 1자리만 보자
total.info()
total.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. 데이터 클렌징

# COMMAND ----------

# 결측치 체크
import missingno as msno
msno.matrix(total)

# COMMAND ----------

pd.isna(total).sum()

# COMMAND ----------

# 결측치 채우기, 앞쪽 결측치는 0으로 채우자
total.fillna(0, inplace=True)
msno.matrix(total)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 정제된 통합 데이터 파일 저장

# COMMAND ----------

# total.to_csv("/dbfs/FileStore/nft/nft_market_220221/total_cleaned.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. 데이터 프로파일링
# MAGIC ## 요약
# MAGIC - 일단위는 파악하기 어렵다. 월단위도 큰 차이 없음, 분기별로 리샘플링하자
# MAGIC - 기간이 다른(결측치) 카테고리 유의해야함
# MAGIC - 전반적으로 값 범위가 매우 다름, 스케일링 필요
# MAGIC   - 스케일링 전에 각 변수별 시각화를 해보자
# MAGIC   - 상관관계도 보자
# MAGIC - 누적 형식이다 보니 우상향 왜도가 많음, log변환해보자
# MAGIC 
# MAGIC ## 형식
# MAGIC - 데이터 사이즈 : 1704 * 70
# MAGIC - 데이터 종류 : 피처 전부 [수치형-연속형]
# MAGIC 
# MAGIC ## 카테고리 (종합 포함 총 7개)
# MAGIC - ALL, Art, Collectible, DeFi, Game, Utility, Metaverse 
# MAGIC 
# MAGIC ## 피처 (10개)
# MAGIC - active_market_wallets : 자산을 매매한 고유 지갑 수 (unique buyer + seller에서 중복 제거 추정)
# MAGIC - number_of_sales : 총 판매 수 (1차시장 + 2차시장)
# MAGIC - sales_usd : 판매완료 건에 대한 총 USD
# MAGIC - average_usd : 판매 평균 USD
# MAGIC - primary_sales : 1차 또는 주요시장의 판매 총 수
# MAGIC   - _**1차 시장 : 창작자의 웹사이트에서 이루어지는 거래**_
# MAGIC - primary_sales_usd : 완료된 주요시장 판매에 지불된 총 USD
# MAGIC - secondary_sales : 2차 시장(사용자간) 판매 총 수
# MAGIC   - _**2차 시장 : opensea와 같은 마켓플레이스에서 이루어지는 거래**_
# MAGIC - secondary_sales_usd : 완료된 2차 시장 판매에 지불된 총 USD
# MAGIC - unique_buyers : 자산을 구매한 고유 지갑 수
# MAGIC - unique_sellers : 자산을 판매한 고유 지갑 수

# COMMAND ----------

total.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 기초 통계 체크
# MAGIC - 피처가 너무 많으니 대표로 "ALL카테고리"를 먼저 보자
# MAGIC - 피처 구분은 크게 3가지로 분류할 수 있음, 1. 지갑(유저별) 수 , 2 시장별 판매 수, 3. 시장별 규모
# MAGIC - "일단위" 우상향 시계열데이터 기초통계에서 얻을 수 있는 정보는 "2/22까지의 max값"(현황, 현재가 가장 높다는 가정)
# MAGIC 
# MAGIC ### 포인트 요약
# MAGIC - 1. 시장 가치 관점
# MAGIC   - 전체 시장 가치와 평균 가치(객단가)와의 갭이 크다, 소수의 판매수가 average를 끌어올린 것으로 추정됨
# MAGIC     --> 분포를 통해 확인 해보자
# MAGIC   - 2차 시장이 1차시장보다 더 크지만, 1차 시장의 시장가치가 2차의 2배로 더 높다. 시장의 특성이 있을 것으로 추정됨
# MAGIC     --> 분포 변동 추이와 함께, 시장의 특성이 어떻게 다른지 확인해보자
# MAGIC - 2. 사용자 분포 관점 
# MAGIC   - 창작자(구매 only), 수집가(판매 only), 투자자(구매+판매) 별 특징을 파악하여, 시장의 활성화여부를 파악해보자
# MAGIC 
# MAGIC 
# MAGIC ### all카테고리 피처별 기초통계 분석(일단위 2/22까지의 Max값)
# MAGIC ---
# MAGIC ### 전체 시장 관점
# MAGIC   - all_average_usd : 약 3,000 달러
# MAGIC   - all_number_of_sales	: 약 3,586만 건
# MAGIC   - all_sales_usd : 약 466억 7.7천만 달러
# MAGIC     - average_usd와 number of sales를 곱하면 1,075.8억인데 실제 전체는 절반도 안된다.
# MAGIC ---
# MAGIC ### 1차/2차 시장 관점
# MAGIC   - all_primary_sales : 약 1,775만 건
# MAGIC     - 전체 판매의 절반 수준
# MAGIC   - all_secondary_sales : 약 1,810만 건
# MAGIC     - 전체 판매의 절반 수준이나 1차 시장 보다 높다.
# MAGIC   - all_primary_sales_usd : 약 373억 8.3천만 달러
# MAGIC     - 판매비중은 절반에 못미치지만, 가치규모는 전체의 70%, 2차의 2배 수준이다. avg usd $ 2,105 정도
# MAGIC   - all_secondary_sales_usd	: 약 182.7억 달러
# MAGIC     - 판매비중은 과반수 이지만, 가치규모는 전체의 30%, 1차의 절반 수준, avg usd $1,009 
# MAGIC ---
# MAGIC ### 사용자 관점
# MAGIC   - all_active_market_wallets : 약 309만개(DAU와 유사)
# MAGIC     - buyer+seller 428.5만 중에서 중복(119만) 제외로 추정
# MAGIC       - 즉, 중복된 119만이 판매와 구매를 모두 하는 "투자자"로 추정됨, 전체 대비 38.7% 수준
# MAGIC   - all_unique_buyers : 약 282.6 만 개
# MAGIC     - 전체 대비 91% 수준, 구매자가 판매자보다 2배 정도 많다. 좋은 생태계 징후, 시장이 죽진 않겠네 
# MAGIC   - all_unique_sellers : 약 145.8 만 개
# MAGIC     - 전체 대비 47% 수준, 

# COMMAND ----------

total.describe(include='all')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw데이터 시각화
# MAGIC - 히스토그램(분포 체크) : 값의 갭차이가 크고 데이터 구간이 너무 길어서 분포차이가 눈에 안띔, >>>>   "로그변환 + 기간 리샘플링" 필요
# MAGIC - 라인 차트(추세 체크) : 갭차이가 커서 추세 체크 어렵고, 변동이벤트가 많아 해석이 어렵다. >>>> "로그변환 + 기간 리샘플링 + 그루핑" 필요 
# MAGIC - 박스 플롯(분위수 체크) : 전체 기간 분위수는 무의미함, >>>> "로그변환 + 기간 리샘플링" 필요

# COMMAND ----------

# MAGIC %md
# MAGIC ### 히스토그램 : 분포 체크
# MAGIC - 비교 어렵다. 정규화 필요

# COMMAND ----------

# 값의 갭차이가 크므로 로그변환해서 봐야함, 그런데 여전히 데이터 구간이 너무 길어서 차이가 눈에 안띈다, 월별/분기별로 리샘플링해보자
display(total)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 라인 플롯 : 추세 체크
# MAGIC - 너무 많아 판단할 수 없음, 정규화 필요

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import patches
%matplotlib inline
plt.style.use("ggplot")

# COMMAND ----------

import matplotlib.dates as mdates

def line_plot(data):
    plt.figure(figsize=(25,80))
    plt.suptitle("Trend check", fontsize=40)

    cols = data.columns
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
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# COMMAND ----------

# 갭이 크고 피처가 너무 많다., 정규화(로그변환) 후 카테고리별로 시각화 하자
# line_plot(total)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 박스 플롯 : 분위수 체크
# MAGIC - pass
# MAGIC - 전체 기간 분위수는 무의미함, 구간별 체크 필요

# COMMAND ----------

import matplotlib.dates as mdates

def box_plot(data):
    plt.figure(figsize=(25,80))
    plt.suptitle("IQR check", fontsize=40)

    cols = data.columns
    for i in range(len(cols)):
        plt.subplot(14,5,i+1)
        plt.title(cols[i], fontsize=20)
        plt.boxplot(data[cols[i]])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# COMMAND ----------

# 전체범위 분위수는 무의미하다.
# box_plot(total)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. 데이터 트랜스포밍

# COMMAND ----------

# MAGIC %md
# MAGIC ## 데이터 리샘플링
# MAGIC - 판다스 시계열구간 집계(통계용어 복원추출 아님)
# MAGIC - 일 -> 월.중위값

# COMMAND ----------

# 반기단위 리샘플링, 딱 조으다
totalM_median = total.resample(rule='M').median()
totalM_median.head()

# COMMAND ----------

# 반기 값 맞는지 체크
total['all_active_market_wallets']['2017-7':'2017-7'].median()

# COMMAND ----------

totalM_median.describe(include='all')

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. 데이터 정규화

# COMMAND ----------

# MAGIC %md
# MAGIC ## 로그 변환

# COMMAND ----------

totalM_median_log = np.log1p(totalM_median)
totalM_median_log.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 스케일링
# MAGIC - 모형을 유지할 수 있고, 정규분포가 아니므로 min-max scaler가 적합
# MAGIC https://ysyblog.tistory.com/217

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
totalM_median_log_scaled = totalM_median_log .copy()
totalM_median_log_scaled.iloc[:,:] = minmax_scaler.fit_transform(totalM_median_log_scaled)
totalM_median_log_scaled.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. 변환 데이터 시각화
# MAGIC - Histogram : 분포 체크, 2Q_avg
# MAGIC - Line : 추세 체크, 2
# MAGIC - Candle Stick : 변동폭 체크
# MAGIC - Stack Area : 비중 체크 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 히스토그램 : 분포 체크
# MAGIC 
# MAGIC ### 요약
# MAGIC - 각 시계열구간별 값의 갭이 너무 커서 한번에 보기 어려움, 로그변환해서 봐야함.
# MAGIC - all카테고리 기준 눈에 띄는 분포는 많지 않음, 좌왜도가 매우 큼, 
# MAGIC - 전체시장관점에서 크게 4개의 분포 경향이 있음 -> 최소 3개 이상의 큰 이벤트가 있었을 것 

# COMMAND ----------

display(totalM_median)
# display(totalM_median['2017':'2020'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 라인 플롯 : 추세 비교
# MAGIC - 17년 4Q, 18년 1Q, 20년 4Q, 21년 4Q의 큰 변곡점이 있다.  -> (18~20년, 21년, 22년) 크게 3개의 구간을 나눠 분석이 필요해보임
# MAGIC - 2018년도부터 비교하는 것이 유의미함(아트,메타버스 등 대부분의 데이터들이 18년도부터 포함됨)
# MAGIC ---
# MAGIC 1. 17년 4Q : 17년 10월 크립토키티(**ERC721**) 출시로 이더리움 NFT가 세상에 알려지기 시작하며, 17년 12월 오픈씨가 출시 되었다. 실제로 크립토키티 인기로 인해 이더네트워크 정체 발생, TX사상 최고치 기록, 속도 크게 저하, 이더리움 트래픽 25%를 차지했다.
# MAGIC 
# MAGIC 
# MAGIC 2. 18년 1Q : NFT 캄브리아기, 100개 이상 프로젝트가 생겨남, metamask와 같은 지갑의 개선으로 온보딩이 쉬워지고 axi infinity(18년 3월 출시)등 유명 Dapp 프로젝트, 플랫폼 등이 생겨남
# MAGIC 3. 20년 4Q : 주요 마켓플레이스인 오픈시에서 무료로 NFT발행 서비스를 발표(폴리곤으로 추정)
# MAGIC 4. 21년 4Q : 세간의 이목을 끄는 판매와 미술품 경매, 그리고 페이스북의 사명변경(메타)등의 메타버스 시대로 인해 대중의 NFT관심 증대, 
# MAGIC ---
# MAGIC ### 참고 : 구글트랜드 NFT 검색 추이
# MAGIC - 21년 2월 증가, , 3월 감소, 7월 증가, 9월감소, 10월 증가
# MAGIC - 21년 8월 급등(plant vs undead 출시)
# MAGIC - 22년 2월부터 감소(왜지??, 메타 주가 급락과 연관이 있어보임)
# MAGIC https://trends.google.co.kr/trends/explore?date=today%205-y&q=nft

# COMMAND ----------

# 데이터브릭스 display chart에서 로그변환 가능
temp = totalM_median.copy()
temp['index'] = totalM_median.index
display(temp)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 캔들 스틱 : 변동성+추이 비교
# MAGIC - 주식에서 많이 사용하는 캔들스틱 차트를 써보자
# MAGIC - 17~20년와 21년도와의 차이가 큼, 21년도 기준으로분석하는게 유의미 해보임
# MAGIC   - (콜렉션 데이터도 21년 4월~9월)

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] 캔들스틱용 데이터셋 생성기

# COMMAND ----------

# 캔들스틱용 데이터셋 생성 함수(OHLC+리샘플링 변환 및 이동평균 데이터 생성)
def candleTransform(data, rule, start, end):
    total_ohlc = data[start:end].resample(rule).ohlc()
    total_median = data[start:end].resample(rule).median()
    total_median_MA2 = total_median.rolling(2).mean() 
    total_median_MA4 = total_median.rolling(4).mean()
    return total_ohlc, total_median, total_median_MA2, total_median_MA4

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] 캔들스틱 차트 생성기

# COMMAND ----------

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 그래프 생성 함수
def candlePlot(data, mediandata, ma2data, ma4data, col, rule):
    num = 0
    col_data = data[col]
    fig = go.Figure(data=[go.Candlestick(x= col_data.index,
                                                    open=col_data['open'],
                                                    high=col_data['high'],
                                                    low=col_data['low'],
                                                    close=col_data['close']),
                                      go.Scatter(x=col_data.index, y=ma2data[col], line=dict(color='orange', width=1), name ='Median(MA2)'),
                                      go.Scatter(x=col_data.index, y=ma4data[col], line=dict(color='green', width=1), name ='Median(MA4)'),
                                      go.Scatter(x=col_data.index, y=mediandata[col], line=dict(color='blue', width=1), name ='Median(rule)')
                                     ])

    num += 1
    fig.layout = dict(title= f'[{num}], {col}', xaxis_title = rule, yaxis_title= 'Value')

    #         주석 = [ dict ( 
    #         x ='2016-12-09' ,  y = 0.05 ,  xref = 'x' ,  yref = 'paper' , 
    #         showarrow = False ,  xanchor = 'left' ,  text = '증가 기간 시작' )] 
    fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] 캔들스틱용 카테고리 및 피처 분류기

# COMMAND ----------

# 카테고리 및 피처파트  분류기 함수
def category_part_classifier(category, part):
    # 피처파트 입력값 유효성 체크
    if part in ['wholeMarket', 'primaryMarket', 'secondaryMarket', 'userAddress']:

        # 피처파트별,  카테고리 + 파트칼럼을 합쳐 조회할 칼럼명 생성
        col_list = []
        if part == 'wholeMarket':
            for col in ['number_of_sales', 'sales_usd', 'average_usd']:
                col_list.append(category + '_' + col)
                result = col_list
        elif part == 'primaryMarket':
            for col in ['primary_sales', 'primary_sales_usd']:
                col_list.append(category + '_' + col)
                result = col_list
        elif part == 'secondaryMarket':
            for col in ['secondary_sales', 'secondary_sales_usd']:
                col_list.append(category + '_' + col)
                result = col_list
        elif part == 'userAddress':
            for col in ['active_market_wallets', 'unique_buyers', 'unique_sellers']:
                col_list.append(category + '_' + col)
                result = col_list
        else :
            print('입력값 또는 분기조건을 확인해주세요.')
            
        return result    
        
    else : 
         print("피처파트 입력값이 유효하지 않습니다. ['wholeMarket', 'primaryMarket', 'secondaryMarket', 'userAddress'] 에서 하나를 입력하세요")   

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] 캔들스틱 실행기

# COMMAND ----------

#  캔들스틱 실행 함수
def candle(data, category, part, rule, start, end):
    # 입력값 유효성 체크
    if category in ['all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility']:

        # 조회할 칼럼리스트 생성 : 카테고리 및 피처파트  분류기 호출
        col_list = category_part_classifier(category, part)

        # 캔들스틱용 데이터셋 생성 함수 호출
        total_ohlc, total_median, total_median_MA2, total_median_MA4 = candleTransform(data, rule, start, end)

        # 캔들스틱 생성 함수 호출
        for col in col_list:
            candlePlot(total_ohlc, total_median, total_median_MA2, total_median_MA4, col, rule)

    else : 
         print("카테고리 입력값이 유효하지 않습니다. 'all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility' 에서 하나를 입력하세요")      
        

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] 캔들스틱 차트 생성기

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2017년도 분석
# MAGIC - 다른 카테고리의 데이터가 포함되지 않아 갭이 크다. 막 시장이 시작되는 단계
# MAGIC -  Monthly 기준 17년 12월 급등, 판매수와 판매가치는 급등했으나 반면에 평가격은 하락함, 저가비중이 많은 듯
# MAGIC     - 판매 수 : 3.2만에서 76.3만으로 25배 이상 급상승, 중위값 57만으로 높은 수준, MA2, Ma4는 낮은데 갑자기 변동이 커서 그런것
# MAGIC     - 판매 가치 : Monthly 기준, 23.2만달러 에서 3천4백만달러로 148배 급상승, 중위값 2천만달러로 높은 수준, 
# MAGIC     - 평균 가격 : 7월 128달러로 고점을 찍고 11월까지 유지하다가 12월에 8로 급감, 감소추세

# COMMAND ----------

candle(total, 'all', 'wholeMarket', 'M', '2017', '2017') # data(raw), category, part, rule, start, end

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2018~20년도 분석
# MAGIC - art, metaver등 다양한 데이터가 취합되고, 수많은 프로젝트가 생기며 시장이 커지는 단계
# MAGIC -  2Q 기준 꾸준히 상승 
# MAGIC     - 판매 수 : 연초 77만에서 연말 5.2백만으로 약 6배 상승, 연말에 open/close 값이 작아지는 것으로 보아 성장폭이 다소 둔해진듯
# MAGIC     - 판매 가치 : 3천4백만에서 1억6천2백만으로 약 5배 상승
# MAGIC     - 평균 가격 : 연초 44불에서 계속 하락하다가 20년 3분기경에 반등하여 연말 31달러마감, 메타버스 매매가 활발했던 시기(평단가 높음)

# COMMAND ----------

candle(total, 'all', 'wholeMarket', 'M', '2018', '2020') # data(raw), category, part, rule, start, end

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2021~22년초 분석
# MAGIC - NFT 거래 이슈(디지털미술품 등), 21년말 메타버스 관심증대(페이스북 사명변경) 등으로 대중 시장에 관심이 시작되는 단계
# MAGIC -  Q기준, 21년 4Q 급등, 22년초 급등
# MAGIC     - 판매 수 : 21년 연초 5.2백만에서  연말 2천6백만으로 약 5배 급등, 22년초는 아직 2/22까지로 분기값이 모두 없음에도 3천5백만으로 상승 추세
# MAGIC         - 특이사항 : 21년 3분기경 급등하는 현상 있었음, 게임 판매수 급등 원인
# MAGIC     - 판매 가치 : 1억6천2백만에서 15.5B(155억달러) 으로 95배 상승,22년초 22.2B로 계속 상승중
# MAGIC     - 평균 가격 : 연초 31달러에서 연말 580달러로 급등(art 매매 활발 원인), 22년초 624달러로 상승중

# COMMAND ----------

candle(total, 'all', 'wholeMarket', 'M', '2021', '2022') # data(raw), category, part, rule, start, end

# COMMAND ----------

# MAGIC %md
# MAGIC ### 박스 플롯 : 분위수, 이상치 체크

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] 박스플롯용 데이터셋 생성기

# COMMAND ----------

# 박스플롯 용 데이터셋 생성 함수(분류라벨, 리샘플링 변환 및 이동평균 데이터 생성)
def boxTransform(data, rule, start, end):
    boxData =  data[start:end].copy()
    
    if rule =='year_month':
        boxData[rule] = (boxData.index).to_period('M').to_timestamp() # 'to_period는..그래프 x기준잡을 때 json시퀀스 에러남, 판다스 버그임 , 타임스탬프로 극복해야함
        boxData_median = boxData.resample('M').median() # 바로 위 작업값이 적용이 안되네.. 아래에 새로 만들어야할듯  
        boxData_median[rule] = (boxData_median.index).to_period('M').to_timestamp()
        boxData_median_MA2 = boxData_median.rolling(2).mean() 
        boxData_median_MA2[rule] = (boxData_median_MA2.index).to_period('M').to_timestamp()
        boxData_median_MA4 = boxData_median.rolling(4).mean()
        boxData_median_MA4[rule] = (boxData_median_MA4.index).to_period('M').to_timestamp()

    elif rule =='year_quater':
        boxData[rule] = (boxData.index).to_period('Q').to_timestamp()
        boxData_median = boxData.resample('Q').median()
        boxData_median[rule] = (boxData_median.index).to_period('Q').to_timestamp()
        boxData_median_MA2 = boxData_median.rolling(2).mean() 
        boxData_median_MA2[rule] = (boxData_median_MA2.index).to_period('M').to_timestamp()
        boxData_median_MA4 = boxData_median.rolling(4).mean()
        boxData_median_MA4[rule] = (boxData_median_MA4.index).to_period('M').to_timestamp()

    else:
        pass

    return boxData, boxData_median, boxData_median_MA2, boxData_median_MA4

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] 박스플롯용 IQR 값 생성기

# COMMAND ----------

# iqr 상한/하한값 생성 함수
def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    # 넘파이 값을 퍼센트로 표시해주는 함수
    iqr = q3 - q1
    lower_fence = q1 -(iqr*1.5) # 기본값
    upper_fence = q3 +(iqr*1.5) # 기본값
    min_list = []
    max_list = []
    # min whisker : 최소값, 또는 '중앙값 - 1.5 × IQR'보다 큰 데이터 중 가장 작은 값 
    # max whisker :  최대값 또는 '중앙값 + 1.5 × IQR'보다 작은 데이터 중 가장 큰 값
    for d in data:
        if d <= upper_fence: max_list.append(d)
        else : pass  
        if d >= lower_fence: min_list.append(d)
        else : pass     
    lower_fence = min(min_list) # 체크 값이 유효하면 덮어쓰기
    upper_fence = max(max_list)
    
    return lower_fence, upper_fence

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] 박스플롯 그래프 생성기

# COMMAND ----------

# import plotly.graph_objects as go
import plotly.express as px
# 그래프 생성 함수
def boxplot(data, data_median, median_MA2, median_MA4, rule, col, num, col_yax):
    fig = px.box(data, x=rule, y=data[col], points='all', title=f'[{num}], {col}')
    fig.add_scatter(x=data_median[rule], y = data_median[col], mode="lines", name ='Median(rule)')
    fig.add_scatter(x=median_MA2[rule], y = median_MA2[col], mode="lines", name ='Median_MA2(rule)')
    fig.add_scatter(x=median_MA4[rule], y = median_MA4[col], mode="lines", name ='Median_MA4(rule)')
#     fig.add_trace(go.Scatter(x=data_median[rule], y=data_median[col], mode="lines", name="Median(rule)"))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
#     fig.update_yaxes( ticklabelposition="inside top", title=None)
    fig.update_yaxes(range=[0,col_yax],  ticklabelposition="inside top", title=None)
    fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [함수] 박스플롯 실행기

# COMMAND ----------

#  박스플롯 실행 함수
def box(data, category, part, rule, start, end):
    
    # 입력값 유효성 체크
    if category in ['all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility']:

        # 조회할 칼럼리스트 생성 : 카테고리 및 피처파트  분류기 호출
        col_list = category_part_classifier(category, part)
        
        # 박스플롯용 데이터셋 생성  호출
        boxData, boxData_median, boxData_median_MA2, boxData_median_MA4 = boxTransform(data, rule, start, end)

        # 박스 플롯 생성
        num = 0 
        for col in col_list:
            # yax값 가이드라인 생성, 각 칼럼별 rule별 상한값중에서 가장 큰값
            uf_list = []
            for r in boxData[rule].unique():
                rdata = boxData[boxData[rule] == r]
                lower_fence, upper_fence =outliers_iqr(rdata[col])
#                 print(col, r, lower_fence, upper_fence)
                uf_list.append(upper_fence)
            col_yax = max(uf_list) 
            print(col, col_yax)
            num += 1
            boxplot(boxData, boxData_median, boxData_median_MA2, boxData_median_MA4, rule, col, num, col_yax)
    else : 
         print("카테고리 입력값이 유효하지 않습니다. 'all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility' 에서 하나를 입력하세요")      
        

# COMMAND ----------

# MAGIC %md
# MAGIC #### Y축 미조정(극단값 확인 가능)

# COMMAND ----------

#  Y축 조정 안함 , 이상치 다 보임
box(total, 'all', 'wholeMarket', 'year_month', '2017', '2017') # year_month or year_quater

# COMMAND ----------

#  Y축 조정 안함 , 이상치 다 보임
box(total, 'all', 'wholeMarket', 'year_month', '2018', '2020')

# COMMAND ----------

#  Y축 조정 안함 , 이상치 다 보임
box(total, 'all', 'wholeMarket', 'year_month', '2021', '2022')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Y축 조정(일부 극단값 확인 불가)

# COMMAND ----------

#  Y축 조정 안함
box(total, 'all', 'wholeMarket', 'year_month', '2017', '2017')

# COMMAND ----------

#  Y축 조정 안함
box(total, 'all', 'wholeMarket', 'year_month', '2021', '2022')

# COMMAND ----------

#  Y축 조정 안함
box(total, 'all', 'wholeMarket', 'year_month', '2021', '2022')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 카테고리별 비교(추세 및 비중)
# MAGIC ### 요약 종합
# MAGIC - 카테고리 전반, sales는 1차시장이 높지만(6-70%) sales_usd는 2차 시장이 '매우' 높음(7-90%)
# MAGIC     - 창작자의 새로운 작품 거래가 활발하지만, 2차시장에서 재거래가 많이 이루어지며 가치가 급등하기 때문
# MAGIC 
# MAGIC - 카테고리 전반, 대부분 21년 중후반에 큰변동기가 있었음
# MAGIC     - 21년 8월 런던포크(가스피 시스템 개선, ether burn), & plant vs undead 출시(뜨는 게임인듯, 추천)
# MAGIC 
# MAGIC - 카테고리 전반, 구매자가 6~70%수준으로 높음
# MAGIC     - utiliy만 90%수준, 1차시장도 90% 수준이며 sales_usd도 아직 1차시장이 더 높음, 아직 생성되어가는 시장으로 보임, Defi도 아직 걸음마 단계
# MAGIC     - 구매자가 판매자보도 소폭높으므로 앞으로도 거래는 활발하고 가격이 상승될 것으로 전망

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 카테고리 (칼럼) 분류기

# COMMAND ----------

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

# MAGIC %md
# MAGIC ### [함수] 카테고리별 라인차트 생성기

# COMMAND ----------

import plotly.express as px

def lineC(data, category):
    # 입력 유효성 체크
    if category in ['all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility']:
        # 카테고리 분류기 호출
        col_list = category_classifier(temp, category)
        # 라인차트 정의
        for j in range(len(col_list)):
            fig = px.line(data[col_list])     
        fig.layout = dict(title= f'{category}카테고리별 피처 *추세* 비교')
        fig.show()    
    else : 
        print("카테고리를 입력하세요, 'all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 카테고리별 피처파트 분류기

# COMMAND ----------

# 카테고리 및 피처파트  분류기 함수
def category_part_classifier2(category, part):
    
    # 피처파트 입력값 유효성 체크
    if part in ['sales', 'sales_usd', 'user']:

        # 피처파트별,  카테고리 + 파트칼럼을 합쳐 조회할 칼럼명 생성
        col_list = []
        if part == 'sales':
            for col in ['primary_sales', 'secondary_sales']:
                col_list.append(category + '_' + col)
                result = col_list
        elif part == 'sales_usd':
            for col in ['primary_sales_usd', 'secondary_sales_usd']:
                col_list.append(category + '_' + col)
                result = col_list
        elif part == 'user':
            for col in ['unique_buyers', 'unique_sellers']:
                col_list.append(category + '_' + col)
                result = col_list
        else :
            print('입력값 또는 분기조건을 확인해주세요.')
            
        return result    
        
    else : 
         print("피처파트 입력값이 유효하지 않습니다. ['sales', 'sales_usd', 'user'] 에서 하나를 입력하세요")   

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 카테고리별 주요피처 누적영역차트 생성기

# COMMAND ----------

# 누적 영역 차트 함수 생성
import plotly.express as px
import plotly.graph_objects as go

def stackareaC(data, category, part):
    # 입력 유효성 체크
    if category in ['all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility']: # 누적차트에서는 all 카테고리 제외
        
        # 피처파트 분류기 호출
        col_list= category_part_classifier2(category, part)
        
        # 누적영역차트 정의
        fig = go.Figure()
        for j in range(len(col_list)):
            fig.add_trace(go.Scatter(
                x = data[col_list[j]].index,
                y = data[col_list[j]].values,
                hoverinfo='x+y',
                mode='lines',
                line=dict(width = 0.5),
                stackgroup='one',
                groupnorm='percent',
                name = col_list[j]
            ))
        fig.layout = dict(title= f'{category}카테고리별 {part}피처파트별 *비중* 비교')
        fig.update_layout(showlegend=True, yaxis=dict(range=[1, 100], ticksuffix='%'))
        fig.show()
    else : 
        print("카테고리를 입력하세요,'all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### all 카테고리
# MAGIC - sales : 1차 시장 50% 수준, 감소세
# MAGIC - sales_usd : 2차 시장 80% 수준(21년 하반기 급증)
# MAGIC - user : 구매자수 70% 수준

# COMMAND ----------

lineC(totalM_median_log_scaled, 'all')

# COMMAND ----------

stackareaC(totalM_median, 'all', 'sales')

# COMMAND ----------

stackareaC(totalM_median, 'all', 'sales_usd')

# COMMAND ----------

stackareaC(totalM_median, 'all', 'user')

# COMMAND ----------

# MAGIC %md
# MAGIC ### collectible 카테고리
# MAGIC - sales : 1차시장  6~70% 수준 유지중
# MAGIC - sales_usd : 2차 시장 85% (21년 후반 60%수준에서 급등)
# MAGIC - user : 구매자 70% 수준 유지중

# COMMAND ----------

lineC(totalM_median_log_scaled, 'collectible')

# COMMAND ----------

stackareaC(totalM_median, 'collectible', 'sales')

# COMMAND ----------

stackareaC(totalM_median, 'collectible', 'sales_usd')

# COMMAND ----------

stackareaC(totalM_median, 'collectible', 'user')

# COMMAND ----------

# MAGIC %md
# MAGIC ### art 카테고리
# MAGIC - sales : 1차시장 60% 수준
# MAGIC - saled_usd : 2차 시장 70% 수준(21년 후반 40%수준에서 급등)
# MAGIC - user : 구매자 60% 수준 유지중

# COMMAND ----------

lineC(totalM_median_log_scaled, 'art')

# COMMAND ----------

stackareaC(totalM_median, 'art', 'sales')

# COMMAND ----------

stackareaC(totalM_median, 'art', 'sales_usd')

# COMMAND ----------

stackareaC(totalM_median, 'art', 'user')

# COMMAND ----------

# MAGIC %md
# MAGIC ### metaverse 카테고리
# MAGIC - sales : 1차시장 60% 수준(감소세)
# MAGIC - saled_usd : 2차 시장80%수준(19년 급등 이후 계속 증가세)
# MAGIC - user : 구매자 70% 수준으로 소폭 하향세

# COMMAND ----------

lineC(totalM_median_log_scaled, 'metaverse')

# COMMAND ----------

stackareaC(totalM_median, 'metaverse', 'sales')

# COMMAND ----------

stackareaC(totalM_median, 'metaverse', 'sales_usd')

# COMMAND ----------

stackareaC(totalM_median, 'metaverse', 'user')

# COMMAND ----------

# MAGIC %md
# MAGIC ### game 카테고리
# MAGIC - sales : 1차 시장 50% 수준(20년 급락이후 보합세)
# MAGIC - saled_usd : 2차 시장 90% 수준(19년 , 21년 급등이후 증가세)
# MAGIC - user : 구매자 60%수준으로 보합세

# COMMAND ----------

lineC(totalM_median_log_scaled, 'game')

# COMMAND ----------

stackareaC(totalM_median, 'game', 'sales')

# COMMAND ----------

stackareaC(totalM_median, 'game', 'sales_usd')

# COMMAND ----------

stackareaC(totalM_median, 'game', 'user')

# COMMAND ----------

# MAGIC %md
# MAGIC ### utility 카테고리
# MAGIC - sales : 1차시장 90%로 유지중
# MAGIC - saled_usd : 2차 시장이 40%수준으로 증가세
# MAGIC - user : 구매자 90% 수준

# COMMAND ----------

lineC(totalM_median_log_scaled, 'utility')

# COMMAND ----------

stackareaC(totalM_median, 'utility', 'sales')

# COMMAND ----------

stackareaC(totalM_median, 'utility', 'sales_usd')

# COMMAND ----------

stackareaC(totalM_median, 'utility', 'user')

# COMMAND ----------

# MAGIC %md
# MAGIC ### DeFi 카테고리
# MAGIC - sales : 1차시장 45%수준(21년중순부터 급감)
# MAGIC - saled_usd : 2차 시장 85%수준(21년 중순부터 급등)
# MAGIC - user : 구매자 65%수준 유지중(21년 중순부터 감소)

# COMMAND ----------

lineC(totalM_median_log_scaled, 'defi')

# COMMAND ----------

stackareaC(totalM_median, 'defi', 'sales')

# COMMAND ----------

stackareaC(totalM_median, 'defi', 'sales_usd')

# COMMAND ----------

stackareaC(totalM_median, 'defi', 'user')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 피처별 비교(추세, 비중)
# MAGIC - active_market_wallets, average_usd, number_of_sales, primary_sales, primary_sales_usd, sales_usd, secondary_sales, secondary_sales_usd, unique_buyers, unique_sellers
# MAGIC ---
# MAGIC ### 요약 종합
# MAGIC - 사용자 관점
# MAGIC   - 구매자/판매자 모두 초기는 콜렉션, 현재는 게임이 과반수
# MAGIC - 시장 가치 관점
# MAGIC   - 전체 시장 가치는 콜렉션이 견인함, 콜렉션 매매수는 급감했지만 평단가가 높기 때문, 게임은 평단가는 낮지만 판매 수가 압도적으로 높고 앞으로도 증가될 전망, art는 매매수는 희소하지만 평단가가 매우 높아 시장 가치에 영향이 크다. 메타버스도 art와 유사한 편이지만, 현재 판매추이가 더욱 감소하여 전망이 불투명함
# MAGIC 
# MAGIC   
# MAGIC ### 요약
# MAGIC - number of sales
# MAGIC   - 게임 71%로 가장 높고, 콜렉션 23.5%. 나머진 안보임
# MAGIC 
# MAGIC - avg usd
# MAGIC   - art가 41%로 가장 높고, 메타버스28%, 콜렉션16.6%, 디파이 8.9%, 게임 2.2%
# MAGIC 
# MAGIC - sales usd
# MAGIC   - 콜렉션 57%로 가장 높음, 게임 24%, art 13%, 메타버스 5%
# MAGIC 
# MAGIC - 1&2 sales
# MAGIC   - 처음엔 콜렉터블이 압도적이어으나 현재는 game이 대부분(65%,77%)
# MAGIC 
# MAGIC - 1 sales usd
# MAGIC   - 콜렉터블이 여전히 51%로 높음, 게임은 17%, art 높음21%, 메타버스는 50%에서 7%로 급감
# MAGIC 
# MAGIC - 2 sales usd
# MAGIC   - 콜렉터블 여전히 58%로 높음, 게임은 25%수준, art11%, 메타버스는 34%에서 4.5%수준으로 급갑
# MAGIC 
# MAGIC - all market wallet
# MAGIC   - 처음엔 콜렉션이 대부분이었으나, 현재는 게임 63%, 콜렉터블 21% 수준
# MAGIC 
# MAGIC - unique buyer/seller
# MAGIC   - 처음엔 collectible이 대다수였으나 현재는 game이 과반수로 역전(60%,70%)

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
# MAGIC ### [함수] 피처별 라인차트 생성기

# COMMAND ----------

def lineF(data, feature):
    # 피처 입력값 유효성 체크
    if feature in ['active_market_wallets', 'average_usd', 'number_of_sales', 'primary_sales', 'primary_sales_usd', 'sales_usd', 'secondary_sales', 'secondary_sales_usd', 'unique_buyers', 'unique_sellers']:
        # 피처 분류기 호출
        col_list = feature_classifier(data, feature)
        # 라인차트 정의
        for j in range(len(col_list)):
            fig = px.line(data[col_list])     
        fig.layout = dict(title= f'{feature} 피처별 *추세* 비교')
        fig.show()    
        
    else : 
        print("피처칼럼을 입력하세요, ['active_market_wallets', 'average_usd', 'number_of_sales', 'primary_sales', 'primary_sales_usd', 'sales_usd', 'secondary_sales', 'secondary_sales_usd', 'unique_buyers', 'unique_sellers']")

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 피처별 누적영역차트 생성기

# COMMAND ----------

# 누적 영역 차트 함수 생성
import plotly.express as px
import plotly.graph_objects as go

def stackareaF(data, feature):
    # 입력 유효성 체크
    if feature in ['active_market_wallets', 'average_usd', 'number_of_sales', 'primary_sales', 'primary_sales_usd', 'sales_usd', 'secondary_sales', 'secondary_sales_usd', 'unique_buyers', 'unique_sellers']:
        # 피처 분류기 호출
        col_list = feature_classifier(data, feature)
        new_col_list = [] # 누적영역차트에서는 all 피처을 빼주자
        for col in col_list:
            if col.split('_')[0] != 'all':
                new_col_list.append(col)
            else :
                pass
    
        # 누적영역차트 정의
        fig = go.Figure()
        for j in range(len(new_col_list)):
            fig.add_trace(go.Scatter(
                x = data[new_col_list[j]].index,
                y = data[new_col_list[j]].values,
                hoverinfo='x+y',
                mode='lines',
                line=dict(width = 0.5),
                stackgroup='one',
                groupnorm='percent',
                name = new_col_list[j]
            ))
        fig.layout = dict(title= f'{feature}피처별 *비중* 비교')
        fig.update_layout(showlegend=True, yaxis=dict(range=[1, 100], ticksuffix='%'))
        fig.show()
    else : 
        print("피처칼럼을 입력하세요, ['active_market_wallets', 'average_usd', 'number_of_sales', 'primary_sales', 'primary_sales_usd', 'sales_usd', 'secondary_sales', 'secondary_sales_usd', 'unique_buyers', 'unique_sellers']")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of Sales 비교

# COMMAND ----------

lineF(totalM_median_log_scaled, 'number_of_sales')

# COMMAND ----------

stackareaF(totalM_median, 'number_of_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sales USD 비교

# COMMAND ----------

lineF(totalM_median_log_scaled, 'sales_usd')

# COMMAND ----------

stackareaF(totalM_median, 'sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Average USD 비교

# COMMAND ----------

lineF(totalM_median_log_scaled, 'average_usd')

# COMMAND ----------

stackareaF(totalM_median, 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Active Market Wallets 비교

# COMMAND ----------

lineF(totalM_median_log_scaled, 'active_market_wallets')

# COMMAND ----------

stackareaF(totalM_median, 'active_market_wallets')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Primary Sales 비교

# COMMAND ----------

lineF(totalM_median_log_scaled, 'primary_sales')

# COMMAND ----------

stackareaF(totalM_median, 'primary_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Secondary Sales 비교

# COMMAND ----------

lineF(totalM_median_log_scaled, 'secondary_sales')

# COMMAND ----------

stackareaF(totalM_median, 'secondary_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Primary Sales USD 비교

# COMMAND ----------

lineF(totalM_median_log_scaled, 'primary_sales_usd')

# COMMAND ----------

stackareaF(totalM_median, 'primary_sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Secondary Sales USD 비교

# COMMAND ----------

lineF(totalM_median_log_scaled, 'secondary_sales_usd')

# COMMAND ----------

stackareaF(totalM_median, 'secondary_sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unique Buyers 비교

# COMMAND ----------

lineF(totalM_median_log_scaled, 'unique_buyers')

# COMMAND ----------

stackareaF(totalM_median, 'unique_buyers')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unique Sellers 비교
# MAGIC - 처음엔 collectible이 대다수였으나 현재는 game이 과반수로 역전함.

# COMMAND ----------

lineF(totalM_median_log_scaled, 'unique_sellers')

# COMMAND ----------

stackareaF(totalM_median, 'unique_sellers')

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. EDA 종합 및 결론
# MAGIC ## [가설1] NFT마켓에서는 최소 3개이상의 큰변곡점이 있을 것이고, 이는 전체 피처 및 시장에 영향을 줬을 것이다.
# MAGIC - all 카테고리 피처별 추세 비교 결과, 4개의 큰 변곡점이 있음을 확인
# MAGIC - 해당 기간에 대한 NFT마켓 히스토리 리서치 결과
# MAGIC   - 17년말 특이사항
# MAGIC     - 17년 10월 크립토키티(erc721) 출시되며 이더NFT가 세상에 알려지기 시작, 실제로 크립토 키티 인기로 인해 이더네트워크 정체 발생, TX사상 최고치 기록, 속도 크게 저하, 이더리움 트래픽 25%를 차지
# MAGIC     - 17년 12월 오픈씨 출시
# MAGIC     - 18년도부터 art, metavers등 주요 데이터 추가
# MAGIC   - 18년-20년 특이사항
# MAGIC     - NFT 캄브리아기로 시장 성장세, 100개 이상 프로젝트가 생겨나고 암호화폐지갑으로 온보딩이 쉬워짐, axi infinity등 유명 Dapp 플랫폼 및 프로젝트 출시
# MAGIC   - 21년중 특이사항
# MAGIC     - 이더리움 런던포크(수수료 시스템 개편, ether burnt)
# MAGIC     - plant vs undead 게임 출시
# MAGIC     - 주요 마켓플레이스인 오픈시에서 무료 NFT발행 서비스 시작(폴리곤 추정)
# MAGIC   - 22년초 특이사항 **(현재 캐즘전 단계로 추정)**
# MAGIC     - 세간의 이목을 끄는 미술품 등 NFT 판매와, 페이스북 사명변경등으로 메타버스시대, NFT에 대한 대중의 관심 증대
# MAGIC 
# MAGIC 
# MAGIC ## [가설2]전체 시장가치와 평균가치(평단가)와의 갭이 큰 것은 소수 판매수가 평단가를 끌어올리기 때문이다
# MAGIC - all카테고리 2Q 히스토그램 분포 파악 결과, 범주1보다  최소 6배~10배가 높은 범주 3가 있음, 갭이 매우 크다.
# MAGIC - 피처별 카테고리 비중 비교를 위한 누적영역 그래프에서 더욱 자세히 알 수 있다.
# MAGIC 
# MAGIC ## [가설3] 2차 시장이 1차시장보다 더 판매수가 많지만, 1차 시장가치가 2차보다 배가 높은 것으로 보아, 재판매로 인한 가치 상승과 시장에 따른 특성 때문일 것이다.
# MAGIC - 실제대로 모든 카테고리가 유사한 현상을 보이고 있다. 이는 재판매로 인한 투자거래로서 가치가 있다는 뜻
# MAGIC - 생각보다 1차 시장(창작자) 직접 거래 규모가 크다
# MAGIC 
# MAGIC ## [가설4] 사용자 분포 관점창작자(구매 only), 수집가(판매 only), 투자자(구매+판매) 별 특징에 따른 카테고리별 시장 활성화 여부를 알 수 있을 것이다.
# MAGIC - 현데이터에서는 사용자를 구분할 수 없다. 다만 카테고리별로 구매/판매 비중을 통한 추정정도는 가능하다.
# MAGIC - all기준 비중이 유사하면서 buyer가 소폭 많으므로 매매가 활발하고 가격이 계속 오를 것으로 보인다. 실제로 2차 시장의 가치가 훨씬더 높다. 앞으로도 NFT거래 활성화 전망이 좋음.
# MAGIC - 일부 메타버스, utility 등 구매자와 사용자 비중이 한쪽으로 치우친 사례는 거래활동성이 매우 낮을 것으로 전망됨

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. 상관관계 분석

# COMMAND ----------

# MAGIC %md
# MAGIC ## [함수] 카테고리/피처 히트맵 생성기

# COMMAND ----------

# MAGIC %md
# MAGIC ## ALL카테고리 피처별 상관분석
# MAGIC - 정규화를 안한 데이터 분석이 더 정확함
# MAGIC - 전반적으로 서로 상관관계가 높게 나오는데, 파생변수들이 섞였기 때문, 이를 제외하고 다시 보자.
# MAGIC   - average_usd는 예상대로 다른 피처들과 상관관계가 없음 -> 사람들이 임의로 가치를 올리기 때문, 가격예측 불가
# MAGIC   - sales_usd는 상관관계가 높으나, sales에 avg_usd가 계산된 파생 변수임 -> 예측에 적절하지 않음 제외
# MAGIC   - market_wallt 또한 buyer와 seller의 파생변수이므로 상관성이 매우 높게 나옴

# COMMAND ----------

import plotly.graph_objects as go
import plotly.express as px

def heatmapC(data, category):
    col_list = category_classifier(data, category)
    z = data[col_list].corr()
    fig = go.Figure()
#     fig = ff.create_annotated_heatmap(z, annotation_text=True)
    fig = px.imshow(z, 
                    width = 800,
                    height = 800 ,
#                     text_auto=True, # 5.5 plotly 설치햇는데도 안됨...ㅜ
                    range_color=[-1,1],
                    color_continuous_scale=["red", "yellow",  "white", "green", "blue"])
    fig.show()

# COMMAND ----------

heatmapC(totalM_median, 'all')

# COMMAND ----------

heatmapC(totalM_median_log, 'all')

# COMMAND ----------

import plotly.graph_objects as go
import plotly.express as px

def heatmapC1(data, category):
    col_list = category_classifier(data, category)
    z = data[col_list].corr()
    fig = go.Figure()
#     fig = ff.create_annotated_heatmap(z, annotation_text=True)
    fig = px.imshow(z, 
#                     width = 800,
#                     height = 800,
                   color_continuous_scale = 'Blues', aspect='auto') # RdBu
#                     text_auto=True, # 5.5 plotly 설치햇는데도 안됨...ㅜ)
                   
    fig.show()

# COMMAND ----------

heatmapC1(totalM_median_log, 'all')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 상관분석 요약 종합 및 피처 셀렉션
# MAGIC - average_usd, sales_usd는 상관관계가 낮음(인위적인 가치 부여로 예측 불가)
# MAGIC - primary sales는 사용자수와 상관관계가 낮음(현재 사용자 수의 시장 분포를 알수 없어서 원인 파악 불가)
# MAGIC - 위 상관관계가 낮은 피처와 그 파생변수들을 제외하면, "사용자수(구매/판매/합계)"와 "판매수(1차/2차/합계)" 가 남음
# MAGIC - 이 경우, "합계"피처가 다른 피처들과 상관성이 높게 나오므로 active_market_wallets 와 number_of_sales 로 추릴 수 있는데, 1차 시장의 상관성이 낮았으므로
# MAGIC ## 최종적으로 전체 피처들과 두루 상관성이 높아 예측효용성이 높은 피처는 "전체 사용자 수"로 결정한다.

# COMMAND ----------

# MAGIC %md
# MAGIC # 피처 예측 가설 설정(단변량 모델링 기준)
# MAGIC - 외부 요인이 클 것 같긴 하지만.. 다변량은 다음에 하고..
# MAGIC - 시계열 패턴이 있을지 모르겠어서 예측이 유의미 할지 모르겠지만, 일단 해보자.

# COMMAND ----------

import plotly.graph_objects as go
import plotly.express as px

def heatmapF(data, feature):
    col_list = feature_classifier(data, feature)
    z = data[col_list].corr()
    fig = go.Figure()
#     fig = ff.create_annotated_heatmap(z, annotation_text=True)
    fig = px.imshow(z, 
#                     width = 800,
#                     height = 800 ,
#                     text_auto=True, # 5.5 plotly 설치해야함, 안됨...ㅜ
                    color_continuous_scale="Blues", aspect='auto')
    fig.show()

# COMMAND ----------

# all은 collectible, defi와 상관성이 높고, metaverse와 utility는 거의없다. defi는 데이터가 많지 않아서 판단하기 어려움.
# collectible 기준으로 다른피처와 상관성도 all과 유사하다. collectible -> all 로 보면 될 듯. -> 이 피처를 예측하면 되려나?
heatmapF(totalM_median_log, 'average_usd')

# COMMAND ----------



# COMMAND ----------

다른 피처들은 모두 상관성이 높은 것이 변동폭이 비슷한데, 평균 가격은 변동폭이 커서 예측이 유의미할 것 같다. 

# COMMAND ----------

all만 대표로 했는데
avg_usd, 기준으로 전체 카테고리  비교(상관분석)

# COMMAND ----------

# MAGIC %md
# MAGIC # 시계열 검증, 분해, 모델 예측
# MAGIC - acf-raw
# MAGIC - 아리마 모델링 떄 추가 차분 고려

# COMMAND ----------

# MAGIC %md
# MAGIC # 시계열 모델링
# MAGIC - arima, 지수평활법, prophet

# COMMAND ----------

# MAGIC %md
# MAGIC # 모델 평가 및비교
# MAGIC - arima, 지수평활법, prophet
