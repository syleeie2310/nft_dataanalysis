# Databricks notebook source
# MAGIC %md # Setting

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
sns.set(style = "white", color_codes = True)

%matplotlib inline

# 관련 라이브러리 임포트 
import matplotlib.font_manager as fm

# COMMAND ----------

# MAGIC %md # Data load

# COMMAND ----------

import os
file_list = os.listdir('/dbfs/FileStore/nft/kaggle')
len(file_list), print(file_list)

# COMMAND ----------

# MAGIC %md ## current_owners
# MAGIC 컬럼 : nft_address, token_id, owner (nft주소, 식별자, 소유자 지갑 주소)
# MAGIC * nft_address: 소유권을 나타내는 토큰이 포함된 NFT 컬렉션 주소입니다.
# MAGIC 2. token_id: 소유권을 나타내는 토큰(수집 내)의 ID입니다.
# MAGIC 3. owner: 이 데이터 집합을 구성할 때 토큰을 소유했던 주소입니다.

# COMMAND ----------

co_df = pd.read_csv("/dbfs/FileStore/nft/kaggle/current_owners.csv", low_memory = False)

# COMMAND ----------

# MAGIC %md # 데이터 확인

# COMMAND ----------

co_df.head(5)

# COMMAND ----------

co_df.drop(['Unnamed: 0'], axis = 1)

# COMMAND ----------

co_df.count(axis = 0)

# COMMAND ----------

co_df.isnull().sum()

# COMMAND ----------

# MAGIC %md ## 10개 이상, 10개 미만 데이터 수 확인

# COMMAND ----------

# owner 기준 groupby, count해서 rename했음. 
owner_num_tokens = co_df.groupby(['owner'], as_index = False).size().rename(columns = {'size' : 'num_tokens'})

# COMMAND ----------

owner_num_tokens.sort_values("num_tokens", inplace=True, ascending=False)

# COMMAND ----------

owner_num_tokens.head(5)

# COMMAND ----------

owner_num_tokens.info()

# COMMAND ----------

over_10_num = (owner_num_tokens.num_tokens >= 10)

# COMMAND ----------

# 10개 이상인 데이터 114054개
a = owner_num_tokens.loc[over_10_num, ['num_tokens']].count()
a

# COMMAND ----------

lower_10_num = owner_num_tokens.num_tokens < 10

# COMMAND ----------

# 10개 미만인 데이터 511300개
b = owner_num_tokens.loc[lower_10_num, ['num_tokens']].count()
b

# COMMAND ----------

# MAGIC %md ## 분포 확인

# COMMAND ----------

owner_num_tokens.head(5)

# COMMAND ----------

over = owner_num_tokens.loc[over_10_num]
over.head()

# COMMAND ----------

lower = owner_num_tokens.loc[lower_10_num, 'num_tokens']
lower.head()

# COMMAND ----------

# plt.figure(figsize = (24, 24))
# fig, axs = plt.subplots(1, 2)
# fig.set_size_inches(24, 5, forward = True)
# fig.suptitle('over 10, lower than 10 num_tokens')
# axs[0].hist(over['owner'], density = False, alpha = 0.75, log = False, bins = 20, color = 'orange')
# axs[0].set_title('nft_count_over_10')
# axs[1].hist(lower['owner'], density = False, alpha = 0.75, log = False, bins = 20)
# axs[1].set_title('nft_count_lower_10')
# plt.setp(axs[0], xlabel = 'owner_address', ylabel = 'num_of_nft_address')
# plt.setp(axs[1], xlabel = 'owner_address', ylabel = 'num_of_nft_address')
# print("")

# COMMAND ----------

# 10 미만의 token 소유 수 분포 
plt.hist(lower, density = False, alpha = 0.75, log = False, bins = 10);

# COMMAND ----------

plt.hist(over, density = False, alpha = 0.75, log = False, bins = 20)

# COMMAND ----------

diamonds_df.select("color","price").groupBy("color").agg(avg("price")).display()

# COMMAND ----------

display(owner_num_tokens)

# COMMAND ----------

display(lower)

# COMMAND ----------

# MAGIC %md #다시 천천히 해보자 왜 안되는건지

# COMMAND ----------

# owner 기준 groupby, count해서 rename했음. 
# owner_num_tokens = co_df.groupby(['owner'], as_index = False).size().rename(columns = {'size' : 'num_tokens'})

# COMMAND ----------

owner_num_tokens.describe()
# max값이 10만

# COMMAND ----------

df_small = owner_num_tokens[owner_num_tokens['num_tokens'] < 10]

# COMMAND ----------

df_big = owner_num_tokens[owner_num_tokens['num_tokens'] >= 10]

# COMMAND ----------

plt.figure(figsize=(24,24))
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(24, 5, forward=True)
fig.suptitle('')
axs[0].hist(df_small["num_tokens"], density=False, alpha=0.75, log=False, bins=9, color='orange')
axs[0].set_title('lower than 10')
axs[1].hist(df_big["num_tokens"], density=False, alpha=0.75, log=False, bins=11)
axs[1].set_title('bigger than 10')
# plt.setp(axs[0], xlabel='Number of transactions out', ylabel='Number of addresses')
# plt.setp(axs[1], xlabel='Number of transactions in', ylabel='Number of addresses')
print("")

# 왜 bigger than 10이 자꾸 이상하게 나올까,,

# COMMAND ----------

# 밑에서 로그 안취하니까 안나오는걸 바탕으로,, 다시 해보자 

plt.figure(figsize=(24,24))
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(24, 5, forward=True)
fig.suptitle('')
axs[0].hist(df_small["num_tokens"], density=False, alpha=0.75, log=False, bins=9, color='orange')
axs[0].set_title('lower than 10')
axs[1].hist(df_big["num_tokens"], density=False, alpha=0.75, log=True, bins=15)
axs[1].set_title('bigger than 10, log scale')
# plt.setp(axs[0], xlabel='Number of transactions out', ylabel='Number of addresses')
# plt.setp(axs[1], xlabel='Number of transactions in', ylabel='Number of addresses')
print("")

# COMMAND ----------

# MAGIC %md ## 결론
# MAGIC 
# MAGIC - 한 명의 owner가 nft address를 10개 이상 가지고 있는 (bigger than 10 = log를 해줘야 그래프에서 값이 나온다. )
# MAGIC - 그리고 10개 미만을 가진 owner가 51만개, 10개 이상을 가지고 있는 owner가 11만개. 따라서 10개 미만을 가지고 있는 owner, (특히 한개짜리)가 많다.

# COMMAND ----------

# 그냥 top owners_50해서 뽑아볼까 

# COMMAND ----------

top_owners_df = co_df.groupby(["owner"], as_index=False).size().rename(columns={"size": "num_tokens"})

# COMMAND ----------

top_owners_df.sort_values("num_tokens", inplace=True, ascending=False)

# COMMAND ----------

top_owners_df.head(50)

# COMMAND ----------

plt.xlabel("Number of tokens owned - n")
plt.ylabel("Number of addresses owning n tokens")
_, _, _ = plt.hist(top_owners_df["num_tokens"], bins=100, log=False)

# COMMAND ----------

plt.xlabel("Number of tokens owned - n")
plt.ylabel("Number of addresses owning n tokens")
_, _, _ = plt.hist(top_owners_df["num_tokens"], bins=100, log=True)
# 로그를 안취해주면 안보임 ㄷㄷ

# COMMAND ----------

# MAGIC %md # 히트맵은? 
# MAGIC 
# MAGIC - 이거야말로 랜덤 샘플링해서 돌려야하나 너무 오래걸린다 이것도
# MAGIC - 이건 심지어 10 이상인거 10 이하인거 둘다 안돌아감 ㄷㄷ!

# COMMAND ----------

result = pd.merge(co_df, owner_num_tokens, how='left', on=None)

# COMMAND ----------

result.head()

# COMMAND ----------

result_small = result[result['num_tokens'] < 10]

# COMMAND ----------

result_small.head()

# COMMAND ----------


plt.pcolor(result_small)
plt.xticks(np.arange(0.5, len(result_small.owner), 1), result_small.owner)
plt.yticks(np.arange(0.5, len(result_small.num_tokens), 1), result_small.num_tokens)
plt.title('Heatmap by plt.pcolor()', fontsize=20)
plt.xlabel('owner', fontsize=14)
plt.ylabel('num_tokens', fontsize=14)
plt.colorbar()

plt.show()


# COMMAND ----------

result_big = result[result['num_tokens'] >= 10]

# COMMAND ----------

result_big.head()

# COMMAND ----------

plt.pcolor(result_big)
plt.xticks(np.arange(0.5, len(result_big.owner.unique()), 1), result_big.owner.unique())
plt.yticks(np.arange(0.5, len(result_big.num_tokens), 1), result_big.num_tokens)
plt.title('Heatmap by plt.pcolor()', fontsize=20)
plt.xlabel('owner', fontsize=14)
plt.ylabel('num_tokens', fontsize=14)
plt.colorbar()

plt.show()

# COMMAND ----------

plt.imshow(result_small['num_tokens'], cmap='cool', interpolation='nearest')
plt.show()

# COMMAND ----------


# ax = sns.heatmap(result_small)
# plt.title('heatmap', fontsize=20)
# plt.show() 
