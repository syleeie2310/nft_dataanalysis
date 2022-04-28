# Databricks notebook source
import numpy as np
import pandas as pd

# COMMAND ----------

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# 관련 라이브러리 임포트 
import matplotlib.font_manager as fm

# #  한글글꼴로 변경
# # plt.rcParams['font.family'] = '한글글꼴명'
# plt.rcParams['font.size'] = 11.0
# # plt.rcParams['font.family'] = 'batang'
# plt.rcParams['font.family'] = 'Malgun Gothic'

# # 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
# matplotlib.rcParams['axes.unicode_minus'] = False
# plt.style.use('ggplot')

# COMMAND ----------

# MAGIC %md # 1. Ethereum NFTs dataset?
# MAGIC 
# MAGIC - On-chain activity from the Ethereum NFT Market (이더리움 NFT market에서의 온체인 활동)
# MAGIC - 이러한 데이터는 문스트림의 공개 데이터 노력의 일환으로 Moonstream.to을 사용하여 수집됨.
# MAGIC - 2021.04.01~2021.09.25 까지 수집.

# COMMAND ----------

import os
file_list = os.listdir('/dbfs/FileStore/nft/kaggle')
len(file_list), print(file_list)

# COMMAND ----------

# MAGIC %md #2. 데이터 설명
# MAGIC [ 핵심 관계 ]
# MAGIC - Mints : 새로운 NFT 생성 이벤트
# MAGIC - Transfers : 이전에 발행된 NFT 소유권 이전 이벤트. 
# MAGIC - (mint 테이블과 transfers 테이블에는 동일한 스키마 존재)
# MAGIC 
# MAGIC [ 컬럼 설명 ]
# MAGIC - Event_id : 데이터 세트 생성 시 생성된 각 이벤트와 연결된 고유 ID
# MAGIC - Transaction_hash : 이벤트를 관찰한 이더리움 트랜젝션 해시
# MAGIC - Block_number : 이벤트 포함하는 트랜젝션 채굴된 이더리움 블록 번호 
# MAGIC - NFT_address : 이벤트 설명하는 NFT 포함하는 스마트 계약의 주소
# MAGIC - Token_id : 주소가 있는 스마트 계약 내에서 이벤트 설명하는 NFT 나타내는 식별자.(NFT_address)
# MAGIC - From_address : 관계에서 표시된 전송 이벤트 이전에 NFT를 소유한 주소( transaction 시작한 주소가 아님.) 
# MAGIC - To_address : 관계에서 표시된 전송 이벤트 이후 NFT 소유한 주소( transaction 수신자였던 주소 아님)
# MAGIC - Transaction_value : 이벤트 발생 거래와 함께 전송된 wei 양(가스비)
# MAGIC - Timestamp : 숫자 있는 트랜젝션 블록이 block_number 이더리움 블록체인으로 마이닝된 시간. (타임스탬프)
# MAGIC 
# MAGIC [ 파생 관계 ]
# MAGIC * 분석을 쉽게 하기 위해 데이터 세트에 포함된 몇 가지 파생관계
# MAGIC - Current_market_values : Wei에서 각 NFT의 현재 예상 시장가치. 
# MAGIC - Current_owners : 각 NFT 현재 소유자 
# MAGIC - Nfts : 데이터 세트에 표시된 NFT 계약에 대한 사용 가능한 메타 데이터. (문 스트림에서)
# MAGIC - Transfer_statistics_by_address : NFT 전송에 관련된 모든 주소 안팎으로 전송
# MAGIC - Transfer_values_quartile_10_distribution_per_address
# MAGIC - Transfer_values_quartile_25_distribution_per_address 
# MAGIC - 각 NFT 컬렉션의 10번째 또는 4번째 분위수에 대해 이러한 관계는 분위수가 나타내는 해당 컬렉션에서 가장 가치 있는 토큰 값의 비율을 제공

# COMMAND ----------

# MAGIC %md ## 1. Checkpoint
# MAGIC 컬럼 : event_type(erc721, mint, transfer), offset (이벤트 타입이랑 변위)

# COMMAND ----------

checkpoint = spark.read.csv("/FileStore/nft/kaggle/checkpoint.csv", header= "true", inferSchema = "true")

# COMMAND ----------

checkpoint.printSchema()

# COMMAND ----------

checkpoint.head(5)

# COMMAND ----------

# MAGIC %md ## 2. ★ current_market_value
# MAGIC 
# MAGIC ** 이 데이터 세트의 경우, 우리는 이 토큰을 포함하는 전송에 대한 마지막 0이 아닌 트랜잭션 값으로 시장 가치를 추정한다.
# MAGIC 이 견적은 일부 이전(예: 단일 거래에서 에스크로 계약에 의해 수행된 여러 토큰 전송)의 경우 부정확할 수 있습니다.
# MAGIC 대다수의 토큰에 대해서는 상당히 정확해야 한다.
# MAGIC 
# MAGIC 컬럼 : nft_address, token_id, market_value (nft 주소, 식별자, 해당 마켓 가치)
# MAGIC * nft_address: 우리가 시장 가치를 나타내는 토큰을 포함하는 NFT 컬렉션의 주소입니다.
# MAGIC 2. token_id: 시장 가치를 나타내는 토큰(수집 내)의 ID입니다.
# MAGIC 3. market_value: 이 데이터 집합 구성 시 토큰의 예상 시장 가치입니다.

# COMMAND ----------

cmv = spark.read.csv("/FileStore/nft/kaggle/current_market_values.csv", header= "true", inferSchema = "true")

# COMMAND ----------

cmv.printSchema()

# COMMAND ----------

cmv.head(5)

# COMMAND ----------

cmv.show()
# _c0 이런건 삭제하면 될듯. 

# COMMAND ----------

# MAGIC %md ##3. ★ current_owners
# MAGIC 컬럼 : nft_address, token_id, owner (nft주소, 식별자, 소유자 지갑 주소)
# MAGIC * nft_address: 소유권을 나타내는 토큰이 포함된 NFT 컬렉션 주소입니다.
# MAGIC 2. token_id: 소유권을 나타내는 토큰(수집 내)의 ID입니다.
# MAGIC 3. owner: 이 데이터 집합을 구성할 때 토큰을 소유했던 주소입니다.

# COMMAND ----------

co = spark.read.csv("/FileStore/nft/kaggle/current_owners.csv", header= "true", inferSchema = "true")

# COMMAND ----------

co.printSchema()

# COMMAND ----------

co.show(5)
# 여기도 c0컬럼 제거해야할듯

# COMMAND ----------

# MAGIC %md ## 4. market_values_distribution
# MAGIC 컬럼: address, token_id, relative_value(상대 가치)
# MAGIC * 여기서 address는 nft address임! 상대가치는 어떻게 구했는지 좀 더 알아봐야할듯.

# COMMAND ----------

mvd = spark.read.csv("/FileStore/nft/kaggle/market_values_distribution.csv", header= "true", inferSchema = "true")

# COMMAND ----------

mvd.printSchema()

# COMMAND ----------

mvd.head(5)

# COMMAND ----------

# MAGIC %md ## 5. mint_holding_times
# MAGIC 컬럼 : days, num_holds 날짜별 홀딩한 수

# COMMAND ----------

mht = spark.read.csv("/FileStore/nft/kaggle/mint_holding_times.csv", header= "true", inferSchema = "true")

# COMMAND ----------

mht.printSchema()

# COMMAND ----------

mht.head(5)

# COMMAND ----------

# MAGIC %md ## 6. ★ mints
# MAGIC 컬렴: event_id, transaction_hash, block_number, nft_address, token_id, from_address, to_address, transaction_value, time_stamp 
# MAGIC * event_id: 이벤트와 관련된 고유한 이벤트 ID입니다.
# MAGIC 2. transaction_messages: 이벤트를 트리거한 트랜잭션의 해시입니다.
# MAGIC 3. block_number: 트랜잭션이 마이닝된 트랜잭션 블록입니다.
# MAGIC 4. nft_address: 주화 토큰이 포함된 NFT 컬렉션의 주소입니다.
# MAGIC 5. token_id: 주조된 토큰의 ID입니다.
# MAGIC 6. from_address: 전송 이벤트의 "보낸 사람" 주소입니다. 조폐국 주소는 0x0000000000000000000000000000입니다.
# MAGIC 7. to_address: 전송 이벤트의 "받는 사람" 주소입니다. 이것은 갓 주조된 토큰의 소유자를 나타냅니다.
# MAGIC 8. transaction_value: 토큰이 발행된 트랜잭션과 함께 전송된 WEI 금액입니다.
# MAGIC 9. time_stamp: 조폐 작업이 블록체인에 마이닝된 시간(마이닝된 블록의 타임스탬프)입니다.

# COMMAND ----------

mints = spark.read.csv("/FileStore/nft/kaggle/mints.csv", header= "true", inferSchema = "true")

# COMMAND ----------

mints.printSchema()

# COMMAND ----------

mints.head()
# c0 는 없애주는게 좋을듯. 

# COMMAND ----------

# MAGIC %md ## 7. ★ nfts
# MAGIC 컬럼: address, name, symbol
# MAGIC * 주소 : NFT 계약의 이더리움 주소 
# MAGIC * 이름 : 계약이 나타내는 NFT 컬렉션 이름
# MAGIC * 기호 : 계약이 나타내는 NFT 컬렉션 기호 

# COMMAND ----------

nfts = spark.read.csv("/FileStore/nft/kaggle/nfts.csv", header= "true", inferSchema = "true")

# COMMAND ----------

nfts.printSchema()

# COMMAND ----------

nfts.head(5)
# _c0컬럼은 지워야할듯.

# COMMAND ----------

# MAGIC %md ## 8. ownership_transitions(소유권 이전)
# MAGIC 컬럼: from_address, to_address, num_transitions
# MAGIC * ~에서 , ~에게 , transitions 수

# COMMAND ----------

ot = spark.read.csv("/FileStore/nft/kaggle/ownership_transitions.csv", header= "true", inferSchema = "true")

# COMMAND ----------

ot.printSchema()

# COMMAND ----------

ot.head(5)

# COMMAND ----------

# MAGIC %md ## 9. Transfer_holding_times
# MAGIC 컬럼 : days, num_holds
# MAGIC * transfer에 걸린 시간? 홀딩 시간?

# COMMAND ----------

tht = spark.read.csv("/FileStore/nft/kaggle/transfer_holding_times.csv", header= "true", inferSchema = "true")

# COMMAND ----------

tht.printSchema()

# COMMAND ----------

tht.head(5)

# COMMAND ----------

# MAGIC %md ## 10. ★ transfer_statistics_by_address(주소별 전송 통계)
# MAGIC 
# MAGIC ** 이 표는 nfts, mints, transfer table에서 파생됩니다. 에 참여한 각 주소에 대해
# MAGIC 2021년 4월 1일부터 2021년 9월 25일 사이에 적어도 하나의 NFT 전송, 이 표는 정확히 얼마나 많은 NFT 주소가 전송되었는지를 보여준다.
# MAGIC 기타 주소 및 해당 주소가 수신된 NFT 전송 수.
# MAGIC 
# MAGIC 컬럼 : address, transfers_out, transfers_in
# MAGIC * address: 2021년 4월 1일부터 2021년 9월 25일 사이에 적어도 한 번의 NFT 전송에 참여한 이더리움 주소입니다.
# MAGIC 2. transfers_out: 2021년 4월 1일부터 2021년 9월 25일 사이에 지정된 주소가 다른 주소로 전송된 NFT의 수입니다.
# MAGIC 3. transfers_in: 2021년 4월 1일부터 2021년 9월 25일 사이에 다른 주소가 지정된 주소로 이전한 NFT의 수입니다.

# COMMAND ----------

tsba = spark.read.csv("/FileStore/nft/kaggle/transfer_statistics_by_address.csv", header= "true", inferSchema = "true")

# COMMAND ----------

tsba.printSchema()
# c0은 없는 처리를 하자. 

# COMMAND ----------

tsba.head(5)

# COMMAND ----------

# MAGIC %md ## 11. transfer_values_quantile_10_distribution_per_address (전송값 주소당 10분위수)
# MAGIC 컬럼 : address, quantiles(분위수), relative_value(상대가치)
# MAGIC * address 가 nft address인듯함. 

# COMMAND ----------

tvq10 = spark.read.csv("/FileStore/nft/kaggle/transfer_values_quantile_10_distribution_per_address.csv", header= "true", inferSchema = "true")

# COMMAND ----------

tvq10.printSchema()

# COMMAND ----------

tvq10.head(5)

# COMMAND ----------

# MAGIC %md ## 12. transfer_values_quantile_25_distribution_per_address(전송값 주소당 4분위수)
# MAGIC 컬럼 : address, quantiles(분위수), relative_value(상대가치)
# MAGIC * address 가 nft address인듯함.

# COMMAND ----------

tvq25 = spark.read.csv("/FileStore/nft/kaggle/transfer_values_quantile_25_distribution_per_address.csv", header= "true", inferSchema = "true")

# COMMAND ----------

tvq25.printSchema()

# COMMAND ----------

tvq25.head(5)

# COMMAND ----------

# MAGIC %md ## 13. ★ transfers
# MAGIC 컬럼 : event_id, transaction_hash, block_number, nft_address, token_id, from_address, to_address, transaction_value, time_stamp
# MAGIC * event_id: 이벤트와 관련된 고유한 이벤트 ID입니다.
# MAGIC 2. transaction_messages: 이벤트를 트리거한 트랜잭션의 해시입니다.
# MAGIC 3. block_number: 트랜잭션이 마이닝된 트랜잭션 블록입니다.
# MAGIC 4. nft_address: 전송된 토큰을 포함하는 NFT 컬렉션의 주소입니다.
# MAGIC 5. token_id: 전송된 토큰의 ID.
# MAGIC 6. from_address: 전송 이벤트의 "보낸 사람" 주소입니다. 전송 *시작* 시 토큰을 소유했던 주소입니다.
# MAGIC 7. to_address: 전송 이벤트의 "받는 사람" 주소입니다. 전송 *끝*에서 토큰을 소유했던 주소입니다.
# MAGIC 8. transaction_value: 토큰이 전송된 트랜잭션과 함께 전송된 WEI 금액입니다.
# MAGIC 9. time_stamp: 전송 작업이 블록체인으로 마이닝된 시간(마이닝된 블록의 타임스탬프)입니다.

# COMMAND ----------

transfers = spark.read.csv("/FileStore/nft/kaggle/transfers.csv", header= "true", inferSchema = "true")

# COMMAND ----------

transfers.printSchema()
# c0 버리기

# COMMAND ----------

transfers.head()

# COMMAND ----------

# MAGIC %md ## 14. transfers_mints
# MAGIC 컬럼: transfer_id, mint_id

# COMMAND ----------

tm = spark.read.csv("/FileStore/nft/kaggle/transfers_mints.csv", header= "true", inferSchema = "true")

# COMMAND ----------

tm.printSchema()

# COMMAND ----------

tm.head(5)

# COMMAND ----------

# MAGIC %md # 3. 노트북 예시 따라하기

# COMMAND ----------

# MAGIC %md ## 1. 누가 NFT를 소유하고있는가? 
# MAGIC * 먼저, 누가 NFT를 소유하고 있는지 알아보자. 대다수의 NFT는 작은 주소 집단이 소유하고 있는지, 아니면 시장이 그것보다 더 분산되어 있는지?

# COMMAND ----------

co.head()

# COMMAND ----------

co = co.drop(co._c0)

# COMMAND ----------

co.head()
# _c0 컬럼 드랍 했음.

# COMMAND ----------

top_owners = co.groupBy("owner")

# COMMAND ----------

# MAGIC %md rdd 객체로 하다가 그냥 판다스로 하는게 빠르다고 판단. 판다스로 해봄! 

# COMMAND ----------

co_df = pd.read_csv("/dbfs/FileStore/nft/kaggle/current_owners.csv", low_memory = False)

# COMMAND ----------

co_df.head()

# COMMAND ----------

# 인덱스 안쓰고 싶어서 False 설정
# owner로 groupby, owner의 size 확인, 그 도출된 열 이름 size를 num_tokens 라는 컬럼명으로 변환
top_owners_df = co_df.groupby(["owner"], as_index=False).size().rename(columns={"size": "num_tokens"})

# COMMAND ----------

top_owners_df.sort_values("num_tokens", inplace = True, ascending = False)

# COMMAND ----------

# 상위 20명의 NFT 소유자 (NFT 보유수로 계산)
top_owners_df.head(20)

# COMMAND ----------

# MAGIC %md ## 2. nft 소유권에 대한 히스토그램 
# MAGIC * nft를 하나 이상 소유하는 주소에 대한 밀도 파악

# COMMAND ----------

plt.xlabel("Num of tokens owned")#소유한 토큰 수 
plt.ylabel("Num of addresses owning n tokens(log scale)")#토큰 n개 소유하는 주소 수 (로그)
_, _, _ = plt.hist(top_owners_df['num_tokens'], bins = 100, log = True)
# 압도적으로 많은 NFT 소유자들이 적은 수의 토큰을 보유하고 있음. 수백개 수천개 가지고 있는 주소는 몇 안됨. 

# COMMAND ----------

plt.xlabel("Num of tokens owned")#소유한 토큰 수 
plt.ylabel("Num of addresses owning n tokens(log scale)")#토큰 n개 소유하는 주소 수 (로그)
_, _, _ = plt.hist(top_owners_df['num_tokens'], bins = 30, log = True)
# 압도적으로 많은 NFT 소유자들이 적은 수의 토큰을 보유하고 있음. 수백개 수천개 가지고 있는 주소는 몇 안됨. 

# COMMAND ----------

# 주소당 토큰 수의 분포를 확인할 수 있음.
plt.xlabel("Log of rank of token owner")#토큰 소유자 순위
plt.ylabel("Log of number of tokens owned")#소유한 토큰 수(로그)
_, = plt.plot([np.log(i+1) for i in range(len(top_owners_df["num_tokens"]))], np.log(top_owners_df["num_tokens"]))
# 리스트 컴프렌션 썼음. i가 range(len(top_owners_df["num_tokens"]))에 있을 때 np.log(i+1) 해줌. 그리고 num_tokens 로그취한거. 
# np.log => 해당 입력에 대한 자연 log 배열 반환해줌. 

# COMMAND ----------

# 토큰을 많이 소유하는 주소 = 자동으로 토큰 구매하거나 / 소유하는 컬렉션에 자금을 조달하거나. 
# 토큰 수가 많지 않은 주소의 소유권 추이를 분석해보자. 
# 이 분석을 통해 비알고리즘& 비스마트 계약 소유자 간의 NFT 소유권 추세 추정이 가능! 
# 이 분석을 위해 scale_ cutoff 설정, 이 기준을 초과하지 않는 다수의 토큰을 소유하는 주소만 고려. 

# COMMAND ----------

scale_cutoff = 1500

# COMMAND ----------

low_scale_owners = [num_tokens for num_tokens in top_owners_df["num_tokens"] if num_tokens <= scale_cutoff]
# top_owners_df에서 num_tokens가 scale cut off를 넘지 않는 것만 for문 돌려서 가져옴. 

# COMMAND ----------

plt.xlabel(f"number of tokens owned - n <= {scale_cutoff}")
plt.ylabel("Number of addresses owning n tokens")
_ = plt.hist(low_scale_owners, bins=int(scale_cutoff/5))
# 이것도 로그로 보는게 좋을 것 같음. 

# COMMAND ----------

plt.xlabel(f"Number of tokens owned -n <= {scale_cutoff}")
plt.ylabel("Number of address owning n tokens (log scale)")
_ = plt.hist(low_scale_owners, bins = int(scale_cutoff/50), log = True)
# NFT 시장이 산업적 규모로 발행된다면, 소유되는 NFT는 적다는 걸 알 수 있음.

# COMMAND ----------

# MAGIC %md ## 3. 이더리움 네임 서비스 (ENS)와 같은 NFT 소유권 분포와 현재 그러한 활용사례가 없는 NFT의 소유권 분포에 차이가 있나?
# MAGIC - NFT는 컬렉션으로 출시되면 단일 계약이 여러 토큰을 가지고 있음. 
# MAGIC - 이 질문에 답할 수 있는 한 가지 방법? 각 NFT 컬렉션이 해당 컬렉션의 토큰 소유자에 대해 얼마나 많은 정보를 제공하는지 확인하는 것
# MAGIC 
# MAGIC 1. 각 컬렉션을 해당 컬렉션의 토큰 소유자에 대한 확률분포로 간주. 
# MAGIC 
# MAGIC ```
# MAGIC ex) 집합 C가 n개의 토큰으로 구성, 주소 A가 이러한 토큰 중 m개를 소유하는 경우
# MAGIC 컬렉션 관련 확률 분포에서 pA = m/n의 확률을 주소화. 그 다음 엔트로피 계산. 
# MAGIC 
# MAGIC H(C)=−∑ApAlog(pA).
# MAGIC 
# MAGIC 여기서 합계는 C로부터 적어도 하나의 토큰을 소유하는 모든 주소 A에 걸쳐있음. 
# MAGIC 
# MAGIC H(C)는 이러한 정보를 포함함.
# MAGIC 1. C 컬렉션의 일부로 발행된 토큰 개수 
# MAGIC 2. C의 토큰이 해당 토큰을 소유하는 주소 A에 얼마나 고르게 분포되어 있는지.
# MAGIC 
# MAGIC ```

# COMMAND ----------

contract_o_df = co_df.groupby(["nft_address", "owner"], as_index = False).size().rename(columns = {"size":"num_tokens"})

# COMMAND ----------

contract_o_df.head()

# COMMAND ----------

contract_o_groups = contract_o_df.groupby("nft_address")

entropies = {}

for contract_address, owners_group in contract_o_groups:
    total_supply = owners_group["num_tokens"].sum()
    owners_group["p"] = owners_group["num_tokens"]/total_supply
    owners_group["log(p)"] = np.log2(owners_group["p"])
    owners_group["-plog(p)"] = (-1) * owners_group["p"] * owners_group["log(p)"]
    entropy = owners_group["-plog(p)"].sum()
    entropies[contract_address] = entropy

# COMMAND ----------

plt.xlabel(f"Ownership entropy of NFT collection")
plt.ylabel("Number of NFT collections")
_ = plt.hist(entropies.values(), bins=80)

# COMMAND ----------

# MAGIC %md ### 엔트로피 스펙트럼에서 최고/최저
# MAGIC * 이 엔트로피 스펙트럼 극한에서 컬렉션이 어떻게 보이는지 이해.
# MAGIC * NFT Contract address, Ownership entropy df 만듬

# COMMAND ----------

sorted_entropies = [it for it in entropies.items()]
sorted_entropies.sort(key = lambda it : it[1], reverse = True)
entropies_df = pd.DataFrame.from_records(sorted_entropies, columns = ["nft_address", "entropy"])

# COMMAND ----------

entropies_df.head()
# 가장 높은 entropy 순으로 정렬

# COMMAND ----------

# MAGIC %md ### high entropy 행 설명
# MAGIC - 0x57f1887a8BF19b14fC0dF6Fd9B2acc9Af147eA85 = Ethereum Name Service. (ENS)(https://ens.domains/ko/)
# MAGIC - 0x60F80121C31A0d46B5279700f9DF786054aa5eE5 = Rarible's governance token. Airdrops are the cause of this high entropy.(에어드랍때문에 엔트로피 높음)
# MAGIC - 0xC36442b4a4522E871399CD717aBDD847Ab11FE88 = Uniswap's position NFT, representing non-fungible liquidity positions on Uinswap v3. (https://uniswap.org/)
# MAGIC - 0xabc207502EA88D9BCa29B95Cd2EeE5F0d7936418 = are badges for Yield Guild Games, which seem to have been airdroped to many existing NFT holders.(https://yieldguild.io/)
# MAGIC - 0x5537d90A4A2DC9d9b37BAb49B490cF67D4C54E91 = OneDayPunk collection, which has gained popularity as a down-market Crypto Punks alternative.(https://punkscape.xyz/)

# COMMAND ----------

entropies_df.tail()
# zero entropy 확인
# 이 데이터는 공개는 되었지만 더이상 활동이 없는 NFT. 실패한 프로젝트거나 정식으로 출시하지 않은 프로젝트임.

# COMMAND ----------

# low entropy확인
# 2 이상인 entropy
entropies_df.loc[entropies_df["entropy"] > 2].tail()

# COMMAND ----------

# MAGIC %md ### low entropy 행 설명 
# MAGIC - 0x08CdCF9ba0a4b5667F5A59B78B60FbEFb145e64c = 월드컵 토큰. 4년 전에 마지막으로 활동했음. 최근 활동이 증가했는데 2022년 차기 축구 월드컵 기대때문일 수 있음. 
# MAGIC (https://coinclarity.com/dapp/worldcuptoken/)
# MAGIC - 0xA4fF6019f9DBbb4bCC61Fa8Bd5C39F36ee4eB164 = instigators 프로젝트 연관. (https://instigators.network/)
# MAGIC - 0xB66c7Ca15Af1f357C57294BAf730ABc77FF94940 = Gems of Awareness Benefit 연관 토큰 (https://nftcalendar.io/event/gems-of-awareness-benefit-for-entheon-art-by-alex-grey-x-allyson-grey/)
# MAGIC - 0x5f98B87fb68f7Bb6F3a60BD6f0917723365444C1 = 에미넴 관련 NFT(SHADYCON) Nifty Gateway에서 마케팅한것으로 보임(https://www.eminem.com/news/shadycon-x-nifty-gateway)
# MAGIC - 0x374DBF0dF7aBc89C2bA776F003E725177Cb35750 = WyldFrogz 크립토펑크 파생어, 지구를 구하는 주제를 가지고 있는 프로젝트(현재 페이지 닫아놨음)

# COMMAND ----------

# MAGIC %md ## 4. 이더리움 NFT의 Minting 전략

# COMMAND ----------

mints_df = pd.read_csv("/dbfs/FileStore/nft/kaggle/mints.csv")

# COMMAND ----------

mints_df.head()
# 보니까 transaction value를 정제할 필요는 있어보임(만약에 쓴다면)

# COMMAND ----------

# MAGIC %md ### 계약 당 Mint 수

# COMMAND ----------

# mints_df에서 nft address로 groupby, 인덱스 false, size 확인, 나온 열을 num_nfts라고 이름바꿈!
mint_stats_df = mints_df.groupby("nft_address", as_index = False).size().rename(columns = {"size" : "num_nfts"})

# COMMAND ----------

mint_stats_df.hist("num_nfts", bins=200, log = True);
# bins는 해당 막대의 영역(bins)얼마나 채울지 결정해주는 변수. 
# 정수가 들어가면 해당 정수 +1을 범위로 나눈 값으로 너비 입력함. 
# 참고로 수가 작을수록 막대 뚠뚠해짐(굵어짐)

# COMMAND ----------

# quantile = 분위수. 현재는 10분위로 나눴음.
mint_stats_df.quantile( q= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# COMMAND ----------

# 민팅 스탯에서 num_nfts가 100 이상인거 뽑아옴. sample로
analysis_sample_df = mint_stats_df[mint_stats_df["num_nfts"] >= 100]

# COMMAND ----------

analysis_sample_df.head()

# COMMAND ----------

analysis_sample_df.count()
# nft 2357개 있음. 

# COMMAND ----------

# MAGIC %md ### 그렇다면 한 주소로 얼마나 많은 NFT 발행되었는지.

# COMMAND ----------

# 민팅된거 mints_df에서 nft address (analysis sample 구해놓은거에서 nft address 있는거) 가져옴
analysis_mints_df = mints_df[mints_df["nft_address"].isin(analysis_sample_df["nft_address"])]

# COMMAND ----------

analysis_mints_df.head()
# 확인해봄

# COMMAND ----------

mint_receivers_df = analysis_mints_df.groupby("nft_address", as_index = False)["to_address"].nunique().rename(columns = {"to_address": "num_receivers"})
# nunique = 고유값 수 찾아주는 function 
# 

# COMMAND ----------

mint_receivers_df.head()
# 민팅된거 받은 사람 

# COMMAND ----------

# receiver 분석으로 df 만듬. mint receivers랑 mint_stats_df nft_address(키)로 병합
receiver_analysis_df = mint_receivers_df.m_receivers_df.merge(mint_stats_df, on = "nft_address")
erge(mint_stats_df, on = "nft_address")

# COMMAND ----------

receiver_analysis_df.head()
# address 기준으로 잘 merge 된 모습. 

# COMMAND ----------

receiver_analysis_df.count()
# 카운트 해보니까 동일함. 

# COMMAND ----------

# receiver_df에서 mint_receivers_df의 num_receivers ==1 인거로 hist 만듬. 
receiver_analysis_df[mint_receivers_df["num_receivers"] == 1].hist("num_nfts", bins = 100, log = True);

# COMMAND ----------

# weighted_num_receivers(받은사람의 수 가중치) 컬럼 만듬 
# num_receivers / num_nfts로 계산했음. 
receiver_analysis_df["weighted_num_receivers"] = receiver_analysis_df["num_receivers"] / receiver_analysis_df["num_nfts"]

# COMMAND ----------

receiver_analysis_df.hist("weighted_num_receivers", bins = 100);
# num_receivers 가중치로 히스토그램 찍어봄.

# COMMAND ----------

# MAGIC %md ### 각 NFT에서 Minting(발행) 기간이 얼마나 되었는지

# COMMAND ----------

# 민팅 기간 df 만듬. analysis mints 데이터프레임에서 address로 가져오고, agg (매핑연산)해줌. 
# timestamp min하고 max 연산해서 넣어달라는 표현. 
minting_period_df = analysis_mints_df.groupby("nft_address", as_index = False).agg(min_timestamp = pd.NamedAgg(column = "timestamp", aggfunc ="min"), max_timestamp = pd.NamedAgg(column = "timestamp", aggfunc = "max"))

# COMMAND ----------

minting_period_df.head()

# COMMAND ----------

# duration(걸리는/ 지속되는) 시간 구해주고 컬럼 추가
minting_period_df["duration"] = minting_period_df["max_timestamp"] - minting_period_df["min_timestamp"]

# COMMAND ----------

minting_period_df.hist("duration", bins = 500);
# 일단 1e7 = 천만

# COMMAND ----------

print(f"Number of days per x step: {0.2 * 1e7 / (24 * 3600)}")
# 0.2 * 천만 / (24 * 3600초) 이런 계산임. 
# 3600초는 1시간임. 

# COMMAND ----------

# 민팅 기간 분석 df에. 민팅 기간 df랑 받은사람 분석 df address로 병합
minting_period_analysis_df = minting_period_df.merge(receiver_analysis_df, on = "nft_address")

# COMMAND ----------

minting_period_analysis_df.head()
# 방금까지 만들었던 분석 결과를 한 df에 합쳐놓음 

# COMMAND ----------

minting_period_analysis_df.plot.scatter("duration", "num_receivers", c = 'b', logy = True);
# 참고로 logy 는 y축에 log 적용하는거임. 

# COMMAND ----------

# MAGIC %md ### NFT 민팅 시 요금을 받는지 여부 확인하기

# COMMAND ----------

analysis_mints_df.head()

# COMMAND ----------

minting_costs_df = analysis_mints_df.groupby("nft_address", as_index = False).agg(mean_cost = pd.NamedAgg(column = "transaction_value", aggfunc = "mean"), min_cost = pd.NamedAgg(column = "transaction_value", aggfunc = "min"), max_cost = pd.NamedAgg(column = "transaction_value", aggfunc = "max"))
# 민팅 cost 데이터프레임 만들기 
# analysis mints_에서 nft address로 groupby, index false. 
# agg(집계연산) 사용해서 transaction value mean, min, max로 해서 데이터프레임 만듦. 

# COMMAND ----------

# 데이터프레임 확인
minting_costs_df.head()
# 보면 다 엄청 작은 값으로 되어있는걸 확인할 수 있음. e+17 막 이래서 보기쉽게 만들어주자. 

# COMMAND ----------

# 부동소수점 처리 
minting_costs_df["mean_cost_ether"] = minting_costs_df["mean_cost"]/(10**18)
minting_costs_df["min_cost_ether"] = minting_costs_df["min_cost"]/(10**18)
minting_costs_df["max_cost_ether"] = minting_costs_df["max_cost"]/(10**18)

# COMMAND ----------

minting_costs_df.head()

# COMMAND ----------

cost_analysis_df = minting_costs_df.merge(minting_period_analysis_df, on = "nft_address")
# merge로 합쳐줌. 

# COMMAND ----------

cost_analysis_df.head()

# COMMAND ----------

cost_analysis_df.plot.scatter("num_receivers", "mean_cost_ether");
# 수신자 수와 mean_cost_ether랑 스캐터 찍어봄.

# COMMAND ----------

cost_analysis_df[cost_analysis_df["mean_cost_ether"] > 100]
# 거래수 평균이 100 이상인것. 

# COMMAND ----------

cost_analysis_df[cost_analysis_df["max_cost"] != 0].plot.scatter("num_receivers", "mean_cost", logy = True);
# 거래량 max cost가 0이 아닌것으로 수신자 수, 평균값(로그) 로 찍어본 scatter

# COMMAND ----------

cost_analysis_df[cost_analysis_df["max_cost"] != 0].plot.scatter("num_receivers", "max_cost", logy = True);
# 거래량 max cost가 0이 아닌 것으로 수신자 수, 최대값(로그)로 찍어본 scatter
# 수신자 수는 적은데 최대값은 높게 몰려있음. 적은 수의 수신자가 많은 거래를 한다고 볼 수 있음. 

# COMMAND ----------

# MAGIC %md ## 결론 
# MAGIC ** 우리가 추론할 수 있는것. 
# MAGIC 1. 대부분의 NFT 민팅은 상대적으로 적은 수의 receivers (수신자, to_address와 같은 말)에 의해 이루어진다.
# MAGIC 2. 단일 receivers에 대한 민팅은 상대적으로 적다. 
# MAGIC 3. 대부분의 NFT는 24시간 내에 완전히 등록된다. (완료되는 시간)
# MAGIC 4. 대부분의 이더리움 기반 NFT는 민팅할 때 ETH 부과한다.
# MAGIC 
# MAGIC ** 이것은 대다수의 NFT 프로젝트들이 다음과 같은 바를 알려준다(시사한다).
# MAGIC 1. 토큰 조폐가 허용된 주소의 화이트리스트를 유지하며 각 조폐 작업(민팅)은 ETH(또는 다른 값)와 교환하여 화이트리스트의 구성원이 수행
# MAGIC 2. 임의의 주소를 민팅 값으로 허용해주되, 주소당 발행할 수 있는 토큰 수에 제한을 둠. 
# MAGIC 
# MAGIC 
# MAGIC -- 이러한 NFT 중 얼마나 많은 수가 각 전략을 채택하는지 결정할 만한 충분한 데이터가 없지만, 두 번째 전략은 단일 개인 또는 그룹이 토큰을 청구할 여러 이더리움 주소를 만드는 것이 얼마나 쉬운지 알 수 없기 때문에 가능성이 낮아 보인다.
