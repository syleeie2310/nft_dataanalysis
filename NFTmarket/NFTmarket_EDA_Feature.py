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

# MAGIC %md
# MAGIC # 2. 데이터 정제

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

# 소수점 통일
pd.set_option('float_format', '{:.4f}'.format)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 정제된 통합 데이터 파일 저장

# COMMAND ----------

# total.to_csv("/dbfs/FileStore/nft/nft_market_220221/total_cleaned.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. 데이터 살펴보기
# MAGIC - 데이터 사이즈 : 1704 * 70
# MAGIC - 데이터 종류 : 피처 전부 [수치형-연속형]
# MAGIC 
# MAGIC ## 카테고리 (종합 포함 총 7개)
# MAGIC - ALL, Art, Collectible, DeFi, Game, Utility, Metaverse 
# MAGIC 
# MAGIC ## 피처 (10개)
# MAGIC - active_market_wallets : 자산을 매매한 고유 지갑 수
# MAGIC - number_of_sales : 총 판매 수
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

# MAGIC %md
# MAGIC ## 데이터 프로파일링
# MAGIC - 기간이 다른(결측치) 카테고리 유의해야함
# MAGIC - 전반적으로 값 범위가 매우 다름, 스케일링 필요
# MAGIC   - 스케일링 전에 각 변수별 시각화를 해보자
# MAGIC   - 상관관계도 보자
# MAGIC - 누적 형식이다 보니 우상향 왜도가 많음, log변환해보자

# COMMAND ----------

# MAGIC %md
# MAGIC ## 기초 통계 분석

# COMMAND ----------

total.info()

# COMMAND ----------

total.describe(include='all')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 그래프 시각화
# MAGIC - 추세 및 분포 체크 해보자
# MAGIC - 수치형 연속 데이터이므로 이상치 체크 생략

# COMMAND ----------

#너무 무거움
#display(total)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import patches
%matplotlib inline
plt.style.use("ggplot")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 라인 : 추세 체크

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

line_plot(total) 

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. 데이터 정규화

# COMMAND ----------

# MAGIC %md
# MAGIC ## 로그 변환

# COMMAND ----------

total_log = total.copy()
total_log = np.log1p(total_log)
total_log.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 스케일링
# MAGIC - 모형을 유지할 수 있고, 정규분포가 아니므로 min-max scaler가 적합
# MAGIC https://ysyblog.tistory.com/217

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
total_log_scaled = total_log.copy()
total_log_scaled.iloc[:,:] = minmax_scaler.fit_transform(total_log_scaled)
total_log_scaled.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. 인터렉티브 시각화

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Base
# MAGIC - active_market_wallets, average_usd, number_of_sales, primary_sales, primary_sales_usd, sales_usd, secondary_sales, secondary_sales_usd, unique_buyers, unique_sellers

# COMMAND ----------

# index를 추가해서 특정 카테고리만 display메서드를 실행하는 함수
# display 메서드가 index를 인식 못함... 그렇다고 넣어두기에 뒤에 전처리에 계속 걸리적 거림, 변수 분리해서 관리하기로 번거로움
def displayf(data, feature):
    temp = data.copy()
    temp['index_date'] = data.index # _date를 붙여야 split[1]에서 인덱스 에러 안남
    col_list = []
    if feature in ['active_market_wallets', 'average_usd', 'number_of_sales', 'primary_sales', 'primary_sales_usd', 'sales_usd', 'secondary_sales', 'secondary_sales_usd', 'unique_buyers', 'unique_sellers']:
        for i in range(len(temp.columns)):
            split_col = temp.columns[i].split('_', maxsplit=1)[1]
            if split_col == feature:       
                col_list.append(temp.columns[i])
            elif split_col == 'all_sales_usd' and feature == 'sales_usd' : #콜렉터블만 sales_usd앞에 all이붙어서 따로 처리해줌
                col_list.append('collectible_all_sales_usd')
            else :
                pass
        col_list.append('index_date')
        display(temp[col_list])
    else : 
        print("피처칼럼을 입력하세요, ['active_market_wallets', 'average_usd', 'number_of_sales', 'primary_sales', 'primary_sales_usd', 'sales_usd', 'secondary_sales', 'secondary_sales_usd', 'unique_buyers', 'unique_sellers']")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Number of Sales 비교

# COMMAND ----------

# 추세 비교(raw)
displayf(total, 'number_of_sales')

# COMMAND ----------

# 추세 비교(정규화)
displayf(total_log_scaled, 'number_of_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sales USD 비교

# COMMAND ----------

# 추세 비교(raw)
displayf(total, 'sales_usd')

# COMMAND ----------

# 추세 비교(정규화)
displayf(total_log_scaled, 'sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Average USD 비교

# COMMAND ----------

# 추세 비교(raw)
displayf(total, 'average_usd')

# COMMAND ----------

# 추세 비교
displayf(total_log_scaled, 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Active Market Wallets 비교

# COMMAND ----------

# 추세 비교(raw)
displayf(total, 'active_market_wallets')

# COMMAND ----------

# 추세 비교(정규화)
displayf(total_log_scaled, 'active_market_wallets')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Primary Sales 비교

# COMMAND ----------

# 추세 비교(RAW)
displayf(total, 'primary_sales')

# COMMAND ----------

# 추세 비교(정규화)
displayf(total_log_scaled, 'primary_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Secondary Sales 비교

# COMMAND ----------

# 추세 파악(RAW)
displayf(total, 'secondary_sales')

# COMMAND ----------

# 추세 패턴 파악
displayf(total_log_scaled, 'secondary_sales')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Primary Sales USD 비교

# COMMAND ----------

# 추세 파악
displayf(total, 'primary_sales_usd')

# COMMAND ----------

# 추세 패턴 파악
displayf(total_log_scaled, 'primary_sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Secondary Sales USD 비교

# COMMAND ----------

# 추세 파악(RAW)
displayf(total, 'secondary_sales_usd')

# COMMAND ----------

# 추세 패턴 파악
displayf(total_log_scaled, 'secondary_sales_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unique Buyers 비교

# COMMAND ----------

# 추세 파악(raw)
displayf(total, 'unique_buyers')

# COMMAND ----------

# 추세 패턴 파악
displayf(total_log_scaled, 'unique_buyers')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unique Sellers 비교

# COMMAND ----------

# 추세 파악(raw)
displayf(total, 'unique_sellers')

# COMMAND ----------

# 추세 패턴 파악
displayf(total_log_scaled, 'unique_sellers')
