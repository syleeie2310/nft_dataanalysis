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
line_plot(total)

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
box_plot(total)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. 데이터 리샘플링
# MAGIC - 일 -> 반기(월단위는 분포 비교 어렵고, 1분기도 여전히 범주가 많다)

# COMMAND ----------

# 반기단위 리샘플링, 딱 조으다
total2Q_avg = total.resample(rule='2Q').mean()
total2Q_avg

# COMMAND ----------

# 반기 값 맞는지 체크
total['all_active_market_wallets']['2017-7':'2017-12'].mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 리샘플링 데이터 프로파일링

# COMMAND ----------

total2Q_avg.describe(include='all')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 리샘플링 데이터 시각화

# COMMAND ----------

# MAGIC %md
# MAGIC ### 히스토그램 : 분포 체크
# MAGIC 
# MAGIC ### 요약
# MAGIC - 전체시장관점에서 크게 2회의 큰변동구간이 있다. -> 이벤트 확인 필요
# MAGIC - 1차/2차시장관점 판매수에서 크게 3회의 큰변동구간이 있다. -> 이벤트 확인 필요
# MAGIC - 사용자 관점도 위와 상동 -> 위 주요 2~3개의 이벤트가 모두 동일하게 영향이 있을 것이다.(가정) 
# MAGIC 
# MAGIC ---
# MAGIC ### 전체 시장 관점
# MAGIC - all_number_of_sales : 반기평균기준, 약 1천만 정도 판매수 발생(최근 반기로 추정)
# MAGIC   - 범주1 : 80 ~ 327만, 5.052
# MAGIC   - 범주2 : 327만 ~ 654만, 3.871
# MAGIC   - 범주3 : 981만 ~1308만, 0.928
# MAGIC 
# MAGIC - all_average_usd : 반기평균기준, 적게는 약 100불 전후 크게는 약 600불 정도의 가격대 형성
# MAGIC   - 범주1 : 23.7 ~ 83.2, 6.996
# MAGIC   - 범주2 : 83.2 ~ 142.8, 1.918
# MAGIC   - 범주3:  559.6 ~  619.2,1.988
# MAGIC 
# MAGIC - all_sales_usd : 반기평균기준, 시장가치규모가 19억에서 최근 10배로 상승
# MAGIC   - 범주1 : 4.47백만 ~ 19.43억, 8.928
# MAGIC   - 범주2 : 77.73억 ~ 97.17억, 0.901
# MAGIC   - 범주3 : 174.91억 ~ 194.34억, 1.009
# MAGIC ---
# MAGIC ### 1차/2차시장 관점
# MAGIC - all_primary_sales
# MAGIC   - 범주1 : 0 ~ 165만, 4.032
# MAGIC   - 범주2 : 165만 ~330만, 3.913
# MAGIC   - 범주3 : 330만 ~ 495만, 1.001
# MAGIC   - 범주4 : 660만 ~ 825만, 0.917
# MAGIC   - 범주5 : 1485만 ~ 1650만, 1.011
# MAGIC 
# MAGIC - all_primary_sales_usd
# MAGIC   - 범주1 : 0 ~ 3.62억, 8.913
# MAGIC   - 범주2 : 18.1억 ~ 21.72억, 0.904
# MAGIC   - 범주3 : 32.58억 ~ 36.2억, 1.015
# MAGIC 
# MAGIC - all_secondary_sales
# MAGIC   - 범주1 : 80 ~ 1.62백만, 6.974
# MAGIC   - 범주2 : 1.62백만 ~ 3.24백만, 1.979
# MAGIC   - 범주3 : 4.86백만 ~ 6.48백만, 0.904
# MAGIC   - 범주4 : 1.46천만 ~ 1.61천만, 1.006 
# MAGIC 
# MAGIC - all_secondary_sales_usd
# MAGIC   - 범주1 : 4500~ 15.8억, 8.933
# MAGIC   - 범주2 : 47.44억 ~ 63.25억, 0.909
# MAGIC   - 범주3 : 142.32억 ~ 158.1억, 1.008
# MAGIC ---
# MAGIC ### 사용자 관점
# MAGIC - all_active_market_wallets
# MAGIC   - 범주1 : 약 34 ~ 27.7만,  8.91
# MAGIC   - 범주2 : 약 83만 ~ 110만, 0.907
# MAGIC   - 범주3 : 약 249만~ 277만, 1.006
# MAGIC 
# MAGIC - all_unique_buyers
# MAGIC   - 범주1 : 18 ~ 25.3만, 8.027
# MAGIC   - 범주2 : 25.3만 ~ 50.6만, 0.928
# MAGIC   - 범주3 : 75.9만 ~ 101만, 0.909
# MAGIC   - 범주4 : 228만 ~ 253만, 1.007 
# MAGIC 
# MAGIC - all_unique_sellers
# MAGIC   - 범주1 : 17.8 ~ 13만, 8.92 
# MAGIC   - 범주2 : 25.8만 ~ 38.8만, 0.926
# MAGIC   - 범주3 : 116.4만 ~ 129.3만, 1.005

# COMMAND ----------

# 데이터프로파일링에서 log 변환해서 볼 수 있음
display(total2Q_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 라인 플롯 : 추세 비교
# MAGIC - 17년 4Q, 18년 1Q, 20년 4Q, 21년 4Q의 큰 변곡점이 있다.  -> (17년, 18~20년, 21년, 22년) 크게 4개의 구간을 나눠 분석이 필요해보임
# MAGIC ---
# MAGIC 1. 17년 4Q : 17년 10월 크립토키티(**ERC721**) 출시로 이더리움 NFT가 세상에 알려지기 시작하며, 17년 12월 오픈씨가 출시 되었다. 실제로 크립토키티 인기로 인해 이더네트워크 정체 발생, TX사상 최고치 기록, 속도 크게 저하, 이더리움 트래픽 25%를 차지했다.
# MAGIC 2. 18년 1Q : NFT 캄브리아기, 100개 이상 프로젝트가 생겨남, metamask와 같은 지갑의 개선으로 온보딩이 쉬워지고 axi infinity(18년 3월 출시)등 유명 Dapp 프로젝트, 플랫폼 등이 생겨남
# MAGIC 3. 20년 4Q : 주요 마켓플레이스인 오픈시에서 무료로 NFT발행 서비스를 발표(폴리곤으로 추정)
# MAGIC 4. 21년 4Q : 세간의 이목을 끄는 판매와 미술품 경매, 그리고 페이스북의 사명변경(메타)등의 메타버스 시대로 인해 대중의 NFT관심 증대, 
# MAGIC ---
# MAGIC ### 참고 : 구글트랜드 NFT 검색 추이
# MAGIC - 21년 2월 증가, , 3월 감소, 7월 증가, 9월감소, 10월 증가
# MAGIC - 22년 2월부터 감소(왜지??, 메타 주가 급락과 연관이 있어보임)
# MAGIC https://trends.google.co.kr/trends/explore?date=today%205-y&q=nft

# COMMAND ----------

# chart에서 로그변환 할 수 있음
temp = total2Q_avg.copy()
temp['index'] = total2Q_avg.index
display(temp)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 캔들 스틱 : 분위수+추이 비교
# MAGIC - 주식에서 많이 사용하는 캔들스틱 차트를 써보자
# MAGIC - 17~20년와 21년도와의 차이가 큼, 21년도 기준으로분석하는게 유의미 해보임
# MAGIC   - (콜렉션 데이터도 21년 4월~9월)
# MAGIC 
# MAGIC ### all카테고리만 구간별로 볼수 있게 바꿔야하는데 귀찮아서 아직 안바꿈
# MAGIC - 2017년 하반기, 2018년~20년, 21년 22년 이렇게 4개로 구분해서 분위수 파악 필요 

# COMMAND ----------

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 그래프 생성 함수
def candlestick(data, mediandata, ma2data, ma4data, col_data, i, j):
    num = 0
    fig = go.Figure(data=[go.Candlestick(x= col_data.index,
                                                    open=col_data['open'],
                                                    high=col_data['high'],
                                                    low=col_data['low'],
                                                    close=col_data['close']),
                                      go.Scatter(x=col_data.index, y=ma2data.iloc[:,j], line=dict(color='orange', width=1), name ='Median(MA2)'),
                                      go.Scatter(x=col_data.index, y=ma4data.iloc[:,j], line=dict(color='green', width=1), name ='Median(MA4)'),
                                      go.Scatter(x=col_data.index, y=mediandata.iloc[:,j], line=dict(color='blue', width=1), name ='Median(rule)')
                                     ])
    num += 1
    fig.layout = dict(title= f'[{num}], {data.columns[i][0]}', xaxis_title = 'Quater', yaxis_title= 'Value')

    #         주석 = [ dict ( 
    #         x ='2016-12-09' ,  y = 0.05 ,  xref = 'x' ,  yref = 'paper' , 
    #         showarrow = False ,  xanchor = 'left' ,  text = '증가 기간 시작' )] 
    fig.show()

# COMMAND ----------

def candle(data, rule, option,  start, end):
        
    # 캔들스틱용 데이터셋 생성
    total_ohlc = data[start:end].resample(rule).ohlc()
    total_median = data[start:end].resample(rule).median()
    total_median_MA2 = total_median.rolling(2).mean() 
    total_median_MA4 = total_median.rolling(4).mean()

    # 조건별 그래프 생성 함수 호출
    if option == 'all':
        for i in range(0, len(total_ohlc.columns), 4):
            j = i//4 # location에서는 연산안됨
            col_data = total_ohlc[total_ohlc.columns[i][0]] # 멀티 칼럼이라 그래프 만들기 복잡해진다, 피처(상위칼럼)별로 데이터를 따로 빼주자
            candlestick(total_ohlc, total_median, total_median_MA2, total_median_MA4, col_data, i, j)
    elif option == 'test':
        i = 0
        j = i//4 # location에서는 연산안됨
        col_data = total_ohlc[total_ohlc.columns[i][0]]
        candlestick(total_ohlc, total_median, total_median_MA2, total_median_MA4, col_data, i, j)
    else :
        print("data와 rule, option(all or test), start(유효한 date index를 입력)를 입력하세요")
        

# COMMAND ----------

# 반기로 바꿔도 2020년도까지는 값을 보기 어렵다.  로그변환할 수는 없고.. 축기준에 맞도록 범위 조정필요
candle(total, '2Q', 'test', '2017-06', '2022-02')  # 전체일자를 보려면 start인자에 0 입력,

# COMMAND ----------

candle(total, 'M', 'test', '2017-6', '2017-12') 

# COMMAND ----------

candle(total, 'Q', 'test', '2018', '2020') 

# COMMAND ----------

candle(total, 'Q', 'test', '2021','2021') 

# COMMAND ----------

candle(total, 'M', 'test', '2022', '2022') 

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. 리샘플링 데이터 정규화

# COMMAND ----------

# MAGIC %md
# MAGIC ## 로그 변환

# COMMAND ----------

totalQ_avg = total.resample(rule='Q').mean()
totalQ_avg_log = totalQ_avg.copy()
totalQ_avg_log = np.log1p(totalQ_avg_log)
totalQ_avg_log.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 스케일링
# MAGIC - 모형을 유지할 수 있고, 정규분포가 아니므로 min-max scaler가 적합
# MAGIC https://ysyblog.tistory.com/217

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
totalQ_avg_log_scaled = totalQ_avg_log.copy()
totalQ_avg_log_scaled.iloc[:,:] = minmax_scaler.fit_transform(totalQ_avg_log_scaled)
totalQ_avg_log_scaled.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. 인터렉티브 시각화(비교)
# MAGIC - 분기 기준 고정
# MAGIC - 추세 비교 그래프
# MAGIC - 비중 비교 그래프

# COMMAND ----------

# MAGIC %md
# MAGIC ## Category base

# COMMAND ----------

# MAGIC %md
# MAGIC ### [함수] 카테고리 (칼럼) 분류기

# COMMAND ----------

# 카테고리 분류기

def category_classifier(data, category):
    col_list = []
    for i in range(len(data.columns)):
        if category == 'total':
            return category
        elif data.columns[i].split('_')[0] == category:
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
    if category in ['total', 'all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility']:
        # 카테고리 분류기 호출
        col_list = category_classifier(temp, category)
        # 라인차트 정의
        for j in range(len(col_list)):
            fig = px.line(data[col_list])     
        fig.layout = dict(title= f'{category}카테고리별 피처 *추세* 비교')
        fig.show()    
        
    else : 
        print("카테고리를 입력하세요, ['total', 'all', 'art', 'defi', 'metaverse', 'collectible', 'game', 'utility'")

# COMMAND ----------

# MAGIC %md
# MAGIC ### all 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'all')

# COMMAND ----------

# MAGIC %md
# MAGIC ### collectible 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'collectible')

# COMMAND ----------

# MAGIC %md
# MAGIC ### art 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'art')

# COMMAND ----------

# MAGIC %md
# MAGIC ### metaverse 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'metaverse')

# COMMAND ----------

# MAGIC %md
# MAGIC ### game 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'game')

# COMMAND ----------

# MAGIC %md
# MAGIC ### utility 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'utility')

# COMMAND ----------

# MAGIC %md
# MAGIC ### DeFi 카테고리

# COMMAND ----------

lineC(totalQ_avg_log_scaled, 'defi')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature base
# MAGIC - active_market_wallets, average_usd, number_of_sales, primary_sales, primary_sales_usd, sales_usd, secondary_sales, secondary_sales_usd, unique_buyers, unique_sellers
# MAGIC ---
# MAGIC ### 카테고리별 종합
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

lineF(totalQ_avg_log_scaled, 'number_of_sales')

# COMMAND ----------

stackareaF(totalQ_avg, 'number_of_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sales USD 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'sales_usd')

# COMMAND ----------

stackareaF(totalQ_avg, 'sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Average USD 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'average_usd')

# COMMAND ----------

stackareaF(totalQ_avg, 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Active Market Wallets 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'active_market_wallets')

# COMMAND ----------

stackareaF(totalQ_avg, 'active_market_wallets')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Primary Sales 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'primary_sales')

# COMMAND ----------

stackareaF(totalQ_avg, 'primary_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Secondary Sales 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'secondary_sales')

# COMMAND ----------

stackareaF(totalQ_avg, 'secondary_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Primary Sales USD 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'primary_sales_usd')

# COMMAND ----------

stackareaF(totalQ_avg, 'primary_sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Secondary Sales USD 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'secondary_sales_usd')

# COMMAND ----------

stackareaF(totalQ_avg, 'secondary_sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unique Buyers 비교

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'unique_buyers')

# COMMAND ----------

stackareaF(totalQ_avg, 'unique_buyers')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unique Sellers 비교
# MAGIC - 처음엔 collectible이 대다수였으나 현재는 game이 과반수로 역전함.

# COMMAND ----------

lineF(totalQ_avg_log_scaled, 'unique_sellers')

# COMMAND ----------

stackareaF(totalQ_avg, 'unique_sellers')

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. EDA 종합 및 결론
# MAGIC ## 1. (가정)NFT마켓에서는 최소 3개이상의 큰변곡점이 있을 것이고, 이는 전체 피처에 영향을 줬을 것이다.
# MAGIC --> all 카테고리 피처별 추세 비교 결과, 4개의 큰 변곡점이 있다.
# MAGIC --> 해당 기간에 대한 NFT마켓 히스토리 리서치 결과 000 히스토리가 확인되었다.
# MAGIC --> 2018년도부터 비교해야할 듯(아트,메타버스 등 대부분의 데이터들이 18년도부터 포함됨)
# MAGIC 
# MAGIC ## 2. (가정)전체 시장가치와 평균가치(평단가)와의 갭이 큰 것은 소수 판매수가 평단가를 끌어올리기 때문이다
# MAGIC --> all카테고리 2Q 히스토그램 분포 파악 결과, 범주1보다  최소 6배~10배가 높은 범주 3가 있었다.
# MAGIC --> 피처별 카테고리 비중 비교를 위한 누적영역 그래프에서 더욱 자세히 알 수 있다.
# MAGIC 
# MAGIC ## 3. (가정) 2차 시장이 1차시장보다 더 판매수가 많지만, 1차 시장가치가 2차보다 2배가 높은 것으로 보아, 시장에 따른 특성이 있을 것이다. 그리고 추이 변동도 있을 것이다.
# MAGIC --> all카테고리의 시장비중 비교그래프를 보면,
# MAGIC 판매수는 초기에 1차가 60%로 많았으나 점차 비등한 추세로 감소하고 있고
# MAGIC 시장가치는 초기에 1차가 과반수로 많았으나 현재는 2차가 80%로 대다수를 차지한다.
# MAGIC --> 이상한데 이건 확인 필요함
# MAGIC 
# MAGIC ## 4. 사용자 분포 관점창작자(구매 only), 수집가(판매 only), 투자자(구매+판매) 별 특징에 따른 카테고리별 시장 활성화 여부를 알 수 있을 것이다.
# MAGIC --> 현데이터에서는 사용자를 구분할 수 없다.
# MAGIC --> 다만 카테고리별로 구매/판매 비중을 통한 추정정도는 가능하다.
# MAGIC --> all기준 비중이 유사하면서 buyer가 더 많으므로 매매가 활발하고 가격이 계속 오를 것으로 보인다.
# MAGIC ㄴ 특이사항으로 메타버스, utility 등은 sell이 압도적으로 적은 케이스는 매매 자체가 활성화 되지 않을 수 있다. art는 구매/판매자수 비중이 비슷함.

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA 보강해야할일 
# MAGIC - 캔들스틱, 주요 피처들을 구간별로 비교하기 2018-2020, 2021, 2022 1Q
# MAGIC - 1차/2차 시장, 구매자/판매자 비교 그래프 만들기 (2018년 도 부터보자)
# MAGIC ㄴ  3번가정에 대한 기초통계와 분포그래프다 다른 원인 찾기
# MAGIC - 상관분석
# MAGIC - 모델링 피처 셀렉션
# MAGIC   - all카테고리 기준에서 2022년도 1분기 시장규모 예측하기(평단가+판매수)?
# MAGIC   - 음.. 이건 다변량으로 분석해야 함.. 단변량으로 할 수 있는 건...
# MAGIC   -  모르겠다. 시계열 패턴이 있는지 모르겠어서 시계열 예측이 유의미 할지 모르겠음
# MAGIC   - 그래서 전체 피처를 시계열검증/분해 과정을 함수로 돌려 가장 시계열특성이 높은 피처를 선정하여 예측 하는 것은 어떨까? 
# MAGIC # 모델링
# MAGIC - 시계열 검증, 분해, 모델 예측

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. 상관관계 분석

# COMMAND ----------

# MAGIC %md
# MAGIC ## [함수] 카테고리/피처 히트맵 생성기

# COMMAND ----------

# 음 이게 아닌듯... 피처별로 자기 상관 계수 검증하는 방식으로 가야할듯
 # 특정일자에 따른 히트맵은 그릴 수 있을지 않을까?
  correl() 먼저 해야함

# COMMAND ----------

# MAGIC %md
# MAGIC ## 카테고리별 피처 상관분석

# COMMAND ----------

import plotly.express as px

def heatmapC(data, category):
    col_list = category_classifier(data, category)
    fig = px.imshow(data[col_list].corr())
    fig.show()

# COMMAND ----------

totalQ_avg_log_scaled.tail()

# COMMAND ----------

test = totalQ_avg_log_scaled.copy()
test = test.reset_index().iloc[:,1:]
test

# COMMAND ----------

# avg usd는 낮다, 이것과 연관된 피처들은 모두 뺴야함, 시장규모 같은.
# 남은 피처는 wallet, buyer, seller, sales(whole, first, second)
# 파생피처 빼면, buyer, seller, first sales, second sales, . 이렇게 4개 가 나옴
# 위 4개를 분석해야함 아니면, all wallet vs all sales로 하던가
# 그래 사용자 수를 예측하자. 결국 판매수로 사용자수 증대 영향을 받을 테니.
# 솔직히 안될것 같긴한데.. 실제로 2월이후 감소한것도 같고...외부 요인이 크니까..
# 그래도 배운거지, 외부 데이터랑 상관관계를 한번봐볼까?, 그래서 이후에는 다변량으로 외부데이터를 포함해서 보고싶다? 아니 주식같은 데이터는 예측이 어렵다?
# 상관성, 시계열 패턴까지 해봐야 분석이 가능할지안할지 알수있지 않을까.
# 결론은 파악해보니 이데이터는 미래를 예측할 수 없는 데이터다. 외부 데이터 영향이 크다. 그리고 시계열 예측에 적합하지 않다. 이런식으로 결론을 내릴수 있으려나?
heatmapC(test, 'all')

# COMMAND ----------



# COMMAND ----------

heatmapC(totalQ_avg_log_scaled, 'all', 2021-12-31) # 전체일자를 보려면 index인자에 0 입력

# COMMAND ----------

# MAGIC %md
# MAGIC ## 히트맵 : 상관분석

# COMMAND ----------



# COMMAND ----------


