# Databricks notebook source
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. 데이터 로드 및 통합
# MAGIC - 이더스캔 통계차트 페이지(https://etherscan.io/charts)에서 도표를 제외한 모든 데이터 다운로드(220215)
# MAGIC - 2015년 7월 30일부터 현재까지 일단위 데이터
# MAGIC - Daily Active ERC20 Address는 페이지 오류로 데이터 다운로드 불가 https://etherscan.io/chart/tokenerc-20txns
# MAGIC - pendingQue 는 데이터 인덱스(당일 시간 단위)가 달라서 제외
# MAGIC - Full Node Sync(default) 와 (Archive)는 각 파일당 2개 데이터가 들어가있고, 의미해석이 어려워 제외
# MAGIC - AverageDailyTransactionFee_Average Txn Fee (Ether) 는 값이 전부 0이라서 제외

# COMMAND ----------

# 23개 항목 파일명 리스트로 가져오기
import os
file_list = os.listdir('/dbfs/FileStore/nft/etherscan_stats')
len(file_list), print(file_list)

# COMMAND ----------

# 파일명 추출(확장자&export 제거)
file_name = []
for file in file_list:
    if file.count(".") == 1: 
        name = file.split('.')[0].split('_')[1]
        file_name.append(name)
    else:
        for k in range(len(file)-1,0,-1):
            if file[k]=='.':
                file_name.append(file[:k])
                break

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

# 데이터셋들을 개별 데이터프레임변수로 생성하기 + 각 칼럼 체크하기
data_list = val_list.copy()
for i in range(len(file_name)):
    print(file_name[i])
    data_list[i] = pd.read_csv(f'/dbfs/FileStore/nft/etherscan_stats/export_{file_name[i]}.csv', index_col="Date(UTC)",  parse_dates=True, thousands=',')
    print(file_name[i],'\n', data_list[i].columns, '\n', data_list[i])
    print("="*100)

# COMMAND ----------

# 변수별 칼럼, 데이터 타입 체크
for i in range(len(data_list)):
    data_list[i].info()

# COMMAND ----------

#  데이터셋별 인덱스 타입 확인
for i in range(len(data_list)):
    print(val_list[i], len(data_list[i]), data_list[i].index.dtype)
    print('='*100)

# COMMAND ----------

# 칼럼 정리 - 미사용 칼럼 삭제, 칼럼명 변경, 
rm_col = ['UnixTimeStamp', 'DateTime']

for i in range(len(data_list)):
    for j in range(len(rm_col)) : # 미사용 칼럼 제거
        if rm_col[j] in data_list[i].columns :
            data_list[i].drop(rm_col[j], axis=1, inplace=True)
    for col in range(len(data_list[i].columns)): # 칼럼명 변경
        curName = data_list[i].columns[col]
        changeName = file_name[i] + '_' + curName
        data_list[i].rename(columns={curName : changeName}, inplace=True)       
    print(data_list[i].columns)
    print('=' * 50)

# COMMAND ----------

# 각 데이터변수별 데이터 체크
for i in range(len(data_list)):
    print(file_name[i],'\n', data_list[i].columns, '\n', data_list[i])
    print("="*100)

# COMMAND ----------

#  데이터 통합
total = data_list[0]
for i in range(1, len(data_list)):
    total = pd.merge(total, data_list[i], left_index=True, right_index=True, how='left')
#     print(i, '\n', val_list[i-1], val_list[i],'\n')
#     print(len(total))
#     print('='*100)
#     print(total)
#     print('='*100)

# COMMAND ----------

total

# COMMAND ----------

# 데이터셋별 길이 체크
for i in range(len(data_list)):
#     print(val_list[i], len(data_list[i]))
    if len(data_list[i]) < len(data_list[0]):
        print(val_list[i], len(data_list[i]))
        print('='*100)
    else :
        pass

# COMMAND ----------

total.info()

# COMMAND ----------

total.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. 데이터 정제

# COMMAND ----------

# daily eth burnt와 EnsRegistrations 결측치 채우기, 0으로 채우면 될 듯
total.fillna(0, inplace=True)

# COMMAND ----------

# 소수점 4자리로 모두 통일
pd.set_option('float_format', '{:.4f}'.format)

# COMMAND ----------

# tx value 문자형을 수치형으로 형변환
total['TransactionFee_Value'] = total['TransactionFee_Value'].astype('float64')
# block int를 float으로 형변환(추후에 이상치 변경하고 체크 할때 소수점이 반영안되어 혼돈있음)
total['BlockCountRewards_Value'] = total['BlockCountRewards_Value'].astype('float64')

# COMMAND ----------

# 필요없는 칼럼 제거
# 데이터가 전부 0인 칼럼 AverageDailyTransactionFee_Average Txn Fee (Ether)
# 다른 칼럼과 중복되는 칼럼 MarketCap_Suppl, MarketCap_Price
rmcol = ['AverageDailyTransactionFee_Average Txn Fee (Ether)', 'MarketCap_Supply', 'MarketCap_Price']
for i in range(len(rmcol)):
    total.drop(columns=rmcol[i], inplace=True)

# COMMAND ----------

# 칼럼명 수정 : display메서드 data profile 오류남, index를 자동으로 잡는 듯
total.rename(columns = {'verifiedContracts_No. of Verified Contracts':'verifiedContracts_No of Verified Contracts'}, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 정제된 통합 데이터 파일 저장

# COMMAND ----------

# total.to_csv("/dbfs/FileStore/nft/etherscan_stats/total_cleaned.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. 데이터 살펴보기
# MAGIC - 데이터 사이즈 : 2392 * 27
# MAGIC - 데이터 종류 : 피처 전부 [수치형-연속형]
# MAGIC 
# MAGIC ## 마켓 데이터
# MAGIC - **EtherPrice_Value : 일일 이더리움 가격(USD)**
# MAGIC - ~~MarketCap_Supply : 일일 이더리움 시가 총액에 대한 공급량~~
# MAGIC - MarketCap_MarketCap : 일일 이더리움 시가 총액(공급량*가격)
# MAGIC - ~~MarketCap_Price : 일일 이더리움 시가 총액에 대한 가격(USD)~~
# MAGIC - **Ethersupply2_Value : 일일 이더리움 공급량(블록보상+uncle포함보상+uncle보삼+eth2스테이킹-소각량)**
# MAGIC 
# MAGIC ## 블록체인 데이터
# MAGIC - **TxGrowth_Value : 일일 이더리움 블록체인의 총 트랜젝션 수(참조 : 평균 난이도, 예상 해시 비율, 평균 블록시간, 평균 블록 사이즈, 총 블록 수, 총 엉클 수, 새 주소)**
# MAGIC - DailyActiveERC20Address_Unique Address Total Count : 일일 활성 ERC20토큰 고유 주소 총 수
# MAGIC - DailyActiveERC20Address_Unique Address Receive Count : 일일 활성 ERC20토큰 고유 주소 "수신" 수
# MAGIC - DailyActiveERC20Address_Unique Address Sent Count : 일일 활성 ERC20토큰 고유 주소 "발신" 수
# MAGIC - **DailyActiveEthAddress_Unique Address Total Count : 일일 활성 이더리움 고유 주소 총 수**
# MAGIC - DailyActiveEthAddress_Unique Address Receive Count : 일일 활성 이더리움 고유 주소 "수신" 수
# MAGIC - DailyActiveEthAddress_Unique Address Sent Count : 일일 활성 이더리움 고유 주소 "발신" 수
# MAGIC - BlockSize_Value : 일일 이더리움 평균 블록 크기(bytes)
# MAGIC - **BlockTime_Value : 일일 이더리움 평균 블록이 블록체인에 포함되는 시간(second)**
# MAGIC - **AvgGasPrice_Value (Wei) : 일일 이더리움 평균 가스 가격(wei)**
# MAGIC - GasLimit_Value : 일일 이더리움 평균 가스 한도 가격
# MAGIC - **GasUsed_Value : 일일 이더리움 가스 총 사용량**
# MAGIC - BlockReward_Value : 일일 이더리움 블록 보상 수(참조 : 블록 보상, 엉클포함 보상, 엉클보상, 거래보상)
# MAGIC - BlockCountRewards_Value : 일일 이더리움 블록 생성 수(보상 데이터 미포함)
# MAGIC - Uncles_Value : 일일 이더리움 엉클 수(블록 보상수 미포함)
# MAGIC - **AddressCount_Value  : 일일 이더리움 고유 주소 수**
# MAGIC - AverageDailyTransactionFee_Average Txn Fee (USD) : 일일 이더리움 거래 평균 수수료(USD)
# MAGIC - ~~AverageDailyTransactionFee_Average Txn Fee (Ether) : 일일 이더리움 거래 평균 수수료(ether)~~
# MAGIC - **DailyEthBurnt_BurntFees : EIP-1559로 인한 일일 이더리움 소각 량(Ether)**
# MAGIC 
# MAGIC ## 네트워크 데이터
# MAGIC - **NetworkHash_Value : 일일 이더리움 네트워크 해시율(처리 능력 측정값 GH/s)**
# MAGIC - **BlockDifficulty_Value : 일일 이더리움 네트워크의 채굴 난이도(TH)**
# MAGIC - **TransactionFee_Value : 일일 이더리움 네트워크 거래 수수료 총 수**
# MAGIC - NetworkUtilization_Value : 일일 이더리움 네트워크 활용률(가스한도를 초과하여 사용된 평균 가스 백분율)
# MAGIC 
# MAGIC ## 이더리움 네임 서비스(ENS) 데이터
# MAGIC - EnsRegistrations_Value : 일일 이더리움 네임 서비스 등록 수
# MAGIC 
# MAGIC ## 스마트 계약 데이터
# MAGIC - **verifiedContracts_No. of Verified Contracts : 일일 이더리움 검증된 총 계약 수**

# COMMAND ----------

total.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 기초 통계 분석
# MAGIC - 전반적으로 수치 기준이 매우 다름, 스케일링 필요
# MAGIC   - 스케일링 전에 각 변수에 대한 분석, 그리고 교차분석이 필요함, 일단 그래프부터 전부 그려보고
# MAGIC   - 과거 해석하기 : 각 그래프 안의 주요 변동이슈들에 대해서 파악해보자(업데이트 등)
# MAGIC   - 그 안에서 NFT에 대해서 추정이 가능할지 봐보자, NFT 마켓 데이터를 추가해보던가,(nftgo, nonfungible, )
# MAGIC   - nft 코인도 비교해야할까>?
# MAGIC   - 다른 변수를 추가해서 비교?(btc, usd환율, 나스닥, 금, 부동산, 그래픽카드가격/판매수, 전기소비, 온도/탄소, 유명인 트위터, 인기검색어 등)
# MAGIC   - 스케일링 후엔 상관관계를 보자(구간을 나눠야 할듯)
# MAGIC   - 가설을 발굴해서 비교해보자
# MAGIC   - 유의미한 예측 목표를 세워보자. (타겟?, 알고리즘?)
# MAGIC - 대체로 누적 그래프로서 우상향 왜도가 큰데, 심지어 0으로 채워진 정보들이 많아 갭이 매우 큼  -> 그래서 log변환해서 많이 보는 듯(기능제공함)
# MAGIC   - 정규분포화는 안해도 되려나? 추후 정상성 처리 필요?

# COMMAND ----------

total.describe(include='all')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Profiling

# COMMAND ----------

# index를 추가해서 display메서드를 실행하는 함수
# display 메서드가 index를 인식 못함... 그렇다고 넣어두기에 뒤에 전처리에 계속 걸리적 거림, 변수 분리해서 관리하기로 번거로움
def dp(data):
    temp = data.copy()
    temp['index'] = data.index
    display(temp)

# COMMAND ----------

# log 처리 안된 raw data 
dp(total)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 그래프 시각화 분석

# COMMAND ----------

# 변수별로 그래프 시각화해서 보기
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import patches
%matplotlib inline
plt.style.use("ggplot")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 라인 : 추세 체크
# MAGIC - 시작일이 다른 피처가 많음, 차이가 많이 나는 피처는 따로 나누는게 좋을듯?(burntfee, ens) -> 더 확인해보자
# MAGIC   - ens는 추세성이 없으므로 제외해도 될 듯
# MAGIC   - burntfee는 구간을 나눠 분석해야 할 듯
# MAGIC   
# MAGIC - 전반적으로 우상향이 많으나, 형태가 다양하다. 형태/추세별로 묶어서 연관성을 봐도 좋겠다.
# MAGIC - 추세 기울기 (절대)값이 비슷한 피처들만 따로 교차 분석해보자

# COMMAND ----------

import matplotlib.dates as mdates

def line_plot(data):
    plt.figure(figsize=(25,50))
    plt.suptitle("Trend check", fontsize=40)

    cols = data.columns
    for i in range(len(cols)):
        plt.subplot(9,3,i+1)
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

line_plot(total) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 박스 : 이상치 체크

# COMMAND ----------

def box_plot(data):
    plt.figure(figsize=(25,50))
    plt.suptitle("Outlier Check", fontsize=40)

    cols = data.columns
    for i in range(len(cols)):
        plt.subplot(9,3,i+1) 
        plt.title(cols[i], fontsize=20)
        sb.boxplot(data=data[cols[i]])
    #     sb.swarmplot(data=train[cols[i]]) 너무 커서 안되나..
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
box_plot(total)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 히트맵 : 상관관계 체크
# MAGIC -> 피처별 값 기준이 매우 달라 알기 어려움, 현재로선 0에 가까운 상태

# COMMAND ----------

def heatmap_plot(data):
    plt.figure(figsize=(30,30))

    total_corr = data.corr()
    ax = sb.heatmap(total_corr, annot=False, cmap='coolwarm')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    plt.title('correlation between features', fontsize=40)
    plt.show()
    
heatmap_plot(total)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 산점도 : 상관관계 체크

# COMMAND ----------

# sb.pairplot(total)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. 데이터 정규화(ver1)
# MAGIC - 이상치를 함부로 대체하면 안됨, 로그 변환 필요, ver2에서 새로 정규화

# COMMAND ----------

# MAGIC %md
# MAGIC ## 이상치 대체
# MAGIC **[이상치 없음] 8개**
# MAGIC -----
# MAGIC   - AddressCount_Value
# MAGIC   - BlockReward_Value
# MAGIC   - DailyActiveEthAddress_Unique Address Total Count
# MAGIC   - DailyActiveEthAddress_Unique Address Receive Count 
# MAGIC   - TxGrowth_Value
# MAGIC   - Ethersupply2_Value
# MAGIC   - GasUsed_Value
# MAGIC   - NetworkUtilization_Value
# MAGIC 
# MAGIC **[이상치 있음] 17개, 0.0%대부터 16.8% 까지 다양함, ens와 burntfee는 제외, 보통 20%넘으면 피처 제외함**
# MAGIC -----
# MAGIC   - AverageDailyTransactionFee_Average Txn Fee (USD), 이상치 수=389, 이상치 비율=16.3% 상한값=3.305, 하한값=-1.935
# MAGIC   - AvgGasPrice_Value (Wei), 이상치 수=249, 이상치 비율=10.4% 상한값=113950425919.5, 하한값=-43351110176.5
# MAGIC   - BlockCountRewards_Value, 이상치 수=133, 이상치 비율=5.6% 상한값=7422.125, 하한값=4803.125
# MAGIC   - BlockDifficulty_Value, 이상치 수=186, 이상치 비율=7.8% 상한값=7910.382625, 하한값=-4431.984375
# MAGIC   - BlockSize_Value, 이상치 수=73, 이상치 비율=3.1% 상한값=80917.375, 하한값=-44675.625
# MAGIC   - BlockTime_Value, 이상치 수=285, 이상치 비율=11.9% 상한값=16.875, 하한값=10.994999999999997
# MAGIC   - DailyActiveERC20Address_Unique Address Total Count, 이상치 수=4, 이상치 비율=0.2% 상한값=651059.0, 하한값=-386899.0
# MAGIC   - DailyActiveERC20Address_Unique Address Receive Count, 이상치 수=3, 이상치 비율=0.1% 상한값=534699.5, 하한값=-317608.5
# MAGIC   - DailyActiveERC20Address_Unique Address Sent Count, 이상치 수=1, 이상치 비율=0.0% 상한값=324267.0, 하한값=-193201.0
# MAGIC   - DailyActiveEthAddress_Unique Address Sent Count, 이상치 수=3, 이상치 비율=0.1% 상한값=576691.0, 하한값=-302055.0
# MAGIC   - ~~DailyEthBurnt_BurntFees, 이상치 수=193, 이상치 비율=8.1% 상한값=0.0, 하한값=0.0~~
# MAGIC   - ~~EnsRegistrations_Value, 이상치 수=400, 이상치 비율=16.7% 상한값=65.0, 하한값=-39.0~~
# MAGIC   - EtherPrice_Value, 이상치 수=390, 이상치 비율=16.3% 상한값=1296.0787500000001, 하한값=-709.7112500000001
# MAGIC   - GasLimit_Value, 이상치 수=193, 이상치 비율=8.1% 상한값=22370056.25, 하한값=-5881101.75
# MAGIC   - MarketCap_MarketCap, 이상치 수=402, 이상치 비율=16.8% 상한값=132232.01983989897, 하한값=-73214.56524501201
# MAGIC   - NetworkHash_Value, 이상치 수=174, 이상치 비율=7.3% 상한값=641983.9867750001, 하한값=-361909.6242250001
# MAGIC   - TransactionFee_Value, 이상치 수=333, 이상치 비율=13.9% 상한값=3.4877727885311187e+21, 하한값=-1.9730613482283852e+21
# MAGIC   - Uncles_Value, 이상치 수=403, 이상치 비율=16.8% 상한값=738.5, 하한값=86.5
# MAGIC   - verifiedContracts_No. of Verified Contracts, 이상치 수=36, 이상치 비율=1.5% 상한값=333.5, 하한값=-190.5

# COMMAND ----------

# # 데이터 카피
# total_outlier = total.copy()
# total_outlier

# COMMAND ----------

# # 상한이나 하한 기준을 넘는 값의 인덱스를 리턴하는 함수
# upper_index = []
# def outliers_iqr(data):
#     q1, q3 = np.percentile(data, [25, 75])
#     # 넘파이 값을 퍼센트로 표시해주는 함수
#     iqr = q3 - q1
#     lower_bound = q1 -(iqr*1.5)
#     upper_bound = q3 +(iqr*1.5)
#     index = np.where((data>upper_bound) | (data<lower_bound))
#     return lower_bound, upper_bound, index[0]

# COMMAND ----------

# outliers_iqr(total_outlier['AvgGasPrice_Value (Wei)'])

# COMMAND ----------

# # 칼럼별 이상치 현황 체크
# def outlier_check(data):
#     outlier_index_all= []
#     outlier_col = []
#     for col in data.columns:
#         lb, ub, oi = outliers_iqr(data[col])
#         # 아웃라이어가 존재하는 칼럼 분류
#         if len(oi) != 0:  
#             outlier_col.append(col)
#             print(f'{col}, 이상치 수={len(oi)}, 이상치 비율={"%0.1f%%"%(len(oi)/len(data[col])*100)}')
#             print(f'상한값={ub}, 하한값={lb}')
#             print(f'이상치 위치 출력 = {data[col][oi]}')
#     #         print(f'상한값={ub}, 하한값={lb}, 이상치 인덱스={oi}')
#             print('='*100)
#         else : 
#             print(f'{col} = 이상치 없음')
#             print('='*100)
#         outlier_index_all.append(oi)
#     # print(f'모든 칼럼의 인덱스 리스트 = \n {outlier_index_all}')
#     print(f'아웃라이어 있는 칼럼 = {outlier_col}, {len(outlier_col)}')
#     return outlier_index_all, outlier_col

# outlier_index_all, outlier_col = outlier_check(total_outlier)

# COMMAND ----------

# # 전체 인덱스를 합쳐서, 각 칼럼별로 몇개가 중복되는지 비중을 체크해보자
# # 전체 인덱스 3850개 중  1368개가 중복됨, 음 전체 데이터에서 비중은 심각하진 않은 것 같기도 하고.. ㅜ
# concat_outlier_index = np.concatenate(outlier_index_all, axis=None)
# print(concat_outlier_index)
# print(f'전체 인덱스 수 = {len(concat_outlier_index)}, 고유 인덱스 수 = { len(set(concat_outlier_index))}')
# print(f'전체 데이터 레코드 수 대비 이상치 비율 = {((1-(total_outlier.size-len(concat_outlier_index)) / total_outlier.size)) * 100}')

# COMMAND ----------

# # 이상치를 상한|하한값으로 변경
# # np.where와 인덱스가 맞지 않아서.. 일일이 체크해서 바꿔줘야함..
# for col in outlier_col:
#     lb, ub, oi = outliers_iqr(total_outlier[col])
#     print(col)
#     for i in oi:
#         if total_outlier[col][i] > ub:
#             total_outlier[col][i] = ub
#         elif total_outlier[col][i] < lb:
#             total_outlier[col][i] = lb
#         else :
#             continue
            
# # np.where 실패
# #     np.where(total_outlier[col][oi] > ub, ub,
# #              np.where(total_outlier[col][oi] < lb, lb, total_outlier[col][oi]))    

# COMMAND ----------

# # 잘 바꼈는지 체크
# outlier_index_all, outlier_col = outlier_check(total_outlier)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 박스 : 이상치 체크

# COMMAND ----------

# box_plot(total_outlier)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 스케일링
# MAGIC - 모형을 유지할 수 있고, 정규분포가 아니므로 min-max scaler가 적합
# MAGIC https://ysyblog.tistory.com/217

# COMMAND ----------

# # ENS와 burnt fee 피처 제외, 새 변수 만들기
# total_scaled = total_outlier.copy()
# total_scaled.drop(columns=['EnsRegistrations_Value', 'DailyEthBurnt_BurntFees'], inplace=True)
# total_scaled.head()

# COMMAND ----------

# from sklearn.preprocessing import MinMaxScaler
# minmax_scaler = MinMaxScaler()
# total_scaled.iloc[:,:] = minmax_scaler.fit_transform(total_scaled.iloc[:,:])
# print(total_scaled.shape)
# print(total_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 라인 : 추세 체크

# COMMAND ----------

# line_plot(total_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 상관분석

# COMMAND ----------

# MAGIC %md
# MAGIC ### 히트맵 : 상관관계 체크

# COMMAND ----------

# heatmap_plot(total_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 산점도 : 상관관계 체크

# COMMAND ----------

# sb.pairplot(total_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC # 피처 추가 : 변동수가 일정하여 분석이 무의미함
# MAGIC - 파생 피처 : 블록 카운트 = BlockCountRewards_Value 일일 이더리움 블록 생성 수(보상 데이터 미포함)를 누적합

# COMMAND ----------

# total['BlockCount_AccumulateValue'] = total['BlockCountRewards_Value'].cumsum()
# total

# COMMAND ----------

# dp(total)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # 4. 데이터 정규화(ver2)
# MAGIC - 이상치 보존, log변환 추가

# COMMAND ----------

# MAGIC %md
# MAGIC ## 로그 변환

# COMMAND ----------

# log10 변환, 결과는 log, log1p와 동일
total_log = total.copy()
total_log = np.log1p(total_log)
dp(total_log)

# log 변환
# dp_total1_log = dp_total1.copy()
# dp_total1_log.iloc[:,:-1]= np.log1p(dp_total1_log.iloc[:,:-1])
# display(dp_total1_log)

# log1p 변환
# dp_total1_log1p = dp_total1.copy()
# dp_total1_log1p.iloc[:,:-1]= np.log1p(dp_total1_log1p.iloc[:,:-1])
# display(dp_total1_log1p)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 스케일링

# COMMAND ----------

total_log_scaled = total_log.copy()
total_log_scaled.head()

# COMMAND ----------



# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()
total_log_scaled.iloc[:,:] = minmax_scaler.fit_transform(total_log_scaled)
total_log_scaled

# COMMAND ----------

dp(total_log_scaled)

# COMMAND ----------

dp(total_log_scaled)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 인터렉티브 시각화

# COMMAND ----------

!pip install dash

# COMMAND ----------

!pip install Jinja2

# COMMAND ----------

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

df = px.data.gapminder()
all_continents = df.continent.unique()

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Checklist(
        id="checklist",
        options=[{"label": x, "value": x} 
                 for x in all_continents],
        value=all_continents[3:],
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Graph(id="line-chart"),
])

@app.callback(
    Output("line-chart", "figure"), 
    [Input("checklist", "value")])
def update_line_chart(continents):
    mask = df.continent.isin(continents)
    fig = px.line(df[mask], 
        x="year", y="lifeExp", color='country')
    return fig

app.run_server(debug=True)

# COMMAND ----------

from plotly.offline import plot
from plotly.graph_objs import *
import numpy as np
 
x = np.random.randn(2000)
y = np.random.randn(2000)
 
# Instead of simply calling plot(...), store your plot as a variable and pass it to displayHTML().
# Make sure to specify output_type='div' as a keyword argument.
# (Note that if you call displayHTML() multiple times in the same cell, only the last will take effect.)
 
p = plot(
  [
    Histogram2dContour(x=x, y=y, contours=Contours(coloring='heatmap')),
    Scatter(x=x, y=y, mode='markers', marker=Marker(color='white', size=3, opacity=0.3))
  ],
  output_type='div'
)
 
displayHTML(p)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. 데이터 해석

# COMMAND ----------

# MAGIC %md
# MAGIC ## 히스토리 파악
# MAGIC - 블록발행 추이를 통해 히스토리를 파악해보자
# MAGIC - [이더리움 블록체인 히스토리](https://docs.google.com/document/d/1vwo9mGp6ACt2Ips8tyqH2xu-xBD9tfKDnVPn39NTnKw/edit)
# MAGIC -> 교차해서 같이 보면 좋은 피처는?  블록 생성 수, 블록 사이즈, 이더 가격, 거래 수, 고유 주소 수, 활성 주소 수, 네트워크 해시율 

# COMMAND ----------

dp(total_log_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC # 가설 설정 및 검증

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 가설1) 블록발행 수에 따른 상관관계가 있는 피처는 아래와 같을 것이다
# MAGIC - 정비례 피처(14) : 블록 수, 이더 가격(마켓캡), 이더 공급량, 거래 수, 활성 주소수(erc20 등), 네트워크 해시율, 가스가격, 가스 총 사용량, 
# MAGIC - 반비례 피처(2) : 블록 보상 수, 블록 생성 수, 채굴 난이도, 

# COMMAND ----------

# 정비례 피처
dp(total_log_scaled)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 가설2) 블록발행 수과 무관한 피처(9)는 아래와 같고, 이는 불규칙적인 거래 폭증 때문일 것이다.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 가설3) 불규칙적인 거래 폭증은 외부 요인이 클 것 이다.(트위터 등 커뮤니티 언급 또는 언론매체 노출 등)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 가설4) 이더리움 업데이트는 000 영향을 줬을 것이다.

# COMMAND ----------

# MAGIC %MD
# MAGIC ## 가설 5) 블록 가스한도는 00 영향을 줄 것이다.
# MAGIC 블록 사이즈는 000과 연관이 있을 것이다.

# COMMAND ----------

# MAGIC %md
# MAGIC # 이더리움 이벤트 시퀀스
# MAGIC <img src="https://w.namu.la/s/e56ee462f258ea32f449988d1660d6e1c032b47a6a559cdb9eec0e3f0b8df7ed6bfe9a58ffe310287beae8278036aee2d6f8a02daa1ef0505425d7758f088f496fc66ee00f249d43a507865563543899712263d06c77d85d6ec2d8dd83de4ab89c953bacb2747f8c92ff6a150c125f65" ></img>
