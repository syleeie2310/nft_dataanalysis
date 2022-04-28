# Databricks notebook source
# MAGIC %md # setting

# COMMAND ----------

import pandas as pd
import sqlite3
import numpy as np
from matplotlib.pyplot import figure

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# COMMAND ----------

# MAGIC %md # Data Load

# COMMAND ----------

transfers = pd.read_csv('/dbfs/FileStore/nft/kaggle/transfers.csv')

# COMMAND ----------

# MAGIC %md # NFT Transfers Value_over time

# COMMAND ----------

num_df = (transfers[["transaction_value", "timestamp"]].apply(pd.to_numeric, errors='coerce'))
num_df["timestamp"] = pd.to_datetime(num_df.timestamp, unit='s', errors='coerce')
num_df.set_index("timestamp")
num_df = num_df.resample("1440min", label='right', on='timestamp').sum()

# COMMAND ----------

num_df.describe()

# COMMAND ----------

# Set the width and height of the figure
plt.figure(figsize=(12,6))
# Line chart showing the number of visitors to each museum over time
ax = sns.lineplot(data=num_df, x="timestamp", y="transaction_value")
ax.set(xlabel='timestamp', ylabel='Total value')
plt.title("NFT transfers value over time")
# Add title

# COMMAND ----------

print("number of unique addresses:", transfers["nft_address"].nunique())

# COMMAND ----------

# MAGIC %md # Transactions per address

# COMMAND ----------

#create data frame where group together from_addresses and count size of each group (how many TX each address did in total)
from_series = transfers["from_address"].groupby(transfers["from_address"]).size()
#create data frame where group together from_addresses and count size of each group (how many TX each address did in total)
to_series = transfers["to_address"].groupby(transfers["to_address"]).size()

# COMMAND ----------

df = pd.DataFrame()
df = df.join(to_series.rename("to_count"), how='outer')
df = df.join(from_series.rename('from_count'), how='outer')

# COMMAND ----------

plt.figure(figsize=(24,24))
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(24, 5, forward=True)
fig.suptitle('Numbers of NFT transactions per address')
axs[0].hist(df["from_count"], density=False, alpha=0.75, log=True, bins=20, color='orange')
axs[0].set_title("NFTs Sent from an address")
axs[1].hist(df["to_count"], density=False, alpha=0.75, log=True, bins=20)
axs[1].set_title("NFTs received to an address")
plt.setp(axs[0], xlabel='Number of transactions out', ylabel='Number of addresses')
plt.setp(axs[1], xlabel='Number of transactions in', ylabel='Number of addresses')
print("")

# COMMAND ----------

# MAGIC %md # Transfer Whale 분석

# COMMAND ----------

df_small=df[df["to_count"]<10]
df_small=df_small[df_small["from_count"]<10]

# COMMAND ----------

plt.figure(figsize=(24,24))
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(24, 5, forward=True)
fig.suptitle('Numbers of NFT transactions per address')
axs[0].hist(df_small["from_count"], density=False, alpha=0.75, log=False, bins=9, color='orange')
axs[0].set_title("NFTs Sent from an address")
axs[1].hist(df_small["to_count"], density=False, alpha=0.75, log=False, bins=9)
axs[1].set_title("NFTs received to an address")
plt.setp(axs[0], xlabel='Number of transactions out', ylabel='Number of addresses')
plt.setp(axs[1], xlabel='Number of transactions in', ylabel='Number of addresses')
print("")

# COMMAND ----------

whales_tx=df[df["from_count"]>60000]
whales_rx=df[df["to_count"]>60000]

# COMMAND ----------

whales_rx

# COMMAND ----------

whales_tx

# COMMAND ----------

# MAGIC %md ## Sharks 분석
# MAGIC * 60000 > Sharks > 20000

# COMMAND ----------

sharks_tx=df[df["from_count"]>20e3]
sharks_rx=df[df["to_count"]>20e3]
sharks_tx=sharks_tx[sharks_tx["from_count"]<60e3]
sharks_rx=sharks_tx[sharks_tx["to_count"]<60e3]

plt.figure(figsize=(24,24))
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(24, 5, forward=True)
fig.suptitle('Numbers of NFT transactions per address')
axs[0].hist(sharks_tx["from_count"], density=False, alpha=0.75, log=False, bins=100, color='orange')
axs[0].set_title("NFTs Sent from an address")
axs[1].hist(sharks_rx["to_count"], density=False, alpha=0.75, log=False, bins=9)
axs[1].set_title("NFTs received to an address")
plt.setp(axs[0], xlabel='Number of transactions out', ylabel='Number of addresses')
plt.setp(axs[1], xlabel='Number of transactions in', ylabel='Number of addresses')
print("")

# COMMAND ----------

sharks_tx

# COMMAND ----------

sharks_rx

# COMMAND ----------

# MAGIC %md # Transaction 기준 valuable한 nft

# COMMAND ----------

#transfers not transactions
transactions_per_nft = transfers["nft_address"].groupby(transfers["nft_address"]).size()

# COMMAND ----------

transfers["transaction_value"] = pd.to_numeric(transfers["transaction_value"])
transfers["transaction_value"] = transfers["transaction_value"].fillna(0)
transfers["transaction_value"].head(15)

# COMMAND ----------

total_value_per_nft = transfers[["nft_address", "transaction_value"]].groupby(transfers["nft_address"]).sum()
total_value_per_nft.head(8)

# COMMAND ----------

most_valuable_nfts = total_value_per_nft["transaction_value"].sort_values(ascending=False).head(8)
most_valuable_nfts = most_valuable_nfts.to_frame()
most_valuable_nfts['info'] = None
most_valuable_nfts

# COMMAND ----------

most_valuable_nfts.at['0xa7d8d9ef8D8Ce8992Df33D8b8CF4Aebabd5bD270', 'info'] = 'artblocks'
most_valuable_nfts.at['0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D', 'info'] = 'BAYC Token'
most_valuable_nfts.at['0x60E4d786628Fea6478F785A6d7e704777c86a7c6', 'info'] = 'MAYC Token'
most_valuable_nfts.at['0x7Bd29408f11D2bFC23c34f18275bBf23bB716Bc7', 'info'] = 'Meebits'
most_valuable_nfts.at['0xFF9C1b15B16263C61d017ee9F65C50e4AE0113D7', 'info'] = 'LOOT'
most_valuable_nfts.at['0x3bf2922f4520a8BA0c2eFC3D2a1539678DaD5e9D', 'info'] = '0n1force'
most_valuable_nfts.at['0x059EDD72Cd353dF5106D2B9cC5ab83a52287aC3a', 'info'] = 'Artblocks OLD'
most_valuable_nfts.at['0xBd3531dA5CF5857e7CfAA92426877b022e612cf8', 'info'] = 'pudgypenguins'
most_valuable_nfts

# COMMAND ----------

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.barplot(y=most_valuable_nfts['info'], x=most_valuable_nfts['transaction_value']).set_title('Most valuable NFT conctract chart')

# COMMAND ----------

#get frame with values over time
num_df = pd.DataFrame()
#cast to numeric
num_df = (transfers[["transaction_value", "timestamp"]].apply(pd.to_numeric, errors='coerce'))
#add nft_address column to it
num_df["nft_address"]=transfers["nft_address"]
#filter out only ones that are in most_popular_nft variable
num_df = num_df[num_df.nft_address.isin(list(most_valuable_nfts.index))]
#convert timestamp in to date time
num_df["timestamp"] = pd.to_datetime(num_df.timestamp, unit='s', errors='coerce')
#set index as timestamp
num_df = num_df.set_index("timestamp")

num_df['info'] = None
for i in range(len(most_valuable_nfts)):
    address = most_valuable_nfts.iloc[i].name
    
    num_df.loc[num_df.nft_address == address, 'info'] = most_valuable_nfts.at[address, 'info'] #most_valuable_nfts.at[num_df.iloc[i]['nft_address'], 'info']

num_df

# COMMAND ----------

#group timestamps by day, create column per each nft_address, aggregate transaction value by count and sum
new_df = num_df.groupby([pd.Grouper(freq='d'), 'nft_address', 'info'])['transaction_value'].agg(transaction_value="sum")

# COMMAND ----------

plt.figure(figsize=(24, 12))
# new_df.unstack()
ax = sns.lineplot(data=new_df, x='timestamp', y='transaction_value', hue='info',)

# COMMAND ----------

# MAGIC %md # Transaction 기준 popular nft

# COMMAND ----------

most_popular_nfts = transactions_per_nft.sort_values(ascending=False).head(8)
most_popular_nfts = most_popular_nfts.to_frame()
most_valuable_nfts['info'] = None
most_popular_nfts.at['0x629A673A8242c2AC4B7B8C5D8735fbeac21A6205', 'info'] = 'SOR token'
most_popular_nfts.at['0x57f1887a8BF19b14fC0dF6Fd9B2acc9Af147eA85', 'info'] = 'ENS Base registrar'
most_popular_nfts.at['0xa7d8d9ef8D8Ce8992Df33D8b8CF4Aebabd5bD270', 'info'] = 'BLOCKS Token'
most_popular_nfts.at['0x3B3ee1931Dc30C1957379FAc9aba94D1C48a5405', 'info'] = 'FNDNFT Token'
most_popular_nfts.at['0x06012c8cf97BEaD5deAe237070F9587f8E7A266d', 'info'] = 'CryptoKitties Core'
most_popular_nfts.at['0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D', 'info'] = 'BAYC Token'
most_popular_nfts.at['0x1A92f7381B9F03921564a437210bB9396471050C', 'info'] = ' COOL Token'
most_popular_nfts.at['0xBd3531dA5CF5857e7CfAA92426877b022e612cf8', 'info'] = 'PPG Token'
most_popular_nfts = most_popular_nfts.rename(columns={'nft_address': 'count'})
most_popular_nfts

# COMMAND ----------

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.barplot(y=most_popular_nfts['info'], x=most_popular_nfts['count']).set_title('Most transaction count per NFT conctract chart')

# COMMAND ----------

#get frame with values over time
num_df = pd.DataFrame()
#cast to numeric
num_df = (transfers[["transaction_value", "timestamp"]].apply(pd.to_numeric, errors='coerce'))
#add nft_address column to it
num_df["nft_address"]=transfers["nft_address"]
#filter out only ones that are in most_popular_nfts variable
num_df = num_df[num_df.nft_address.isin(list(most_popular_nfts.index))]
#convert timestamp in to date time
num_df["timestamp"] = pd.to_datetime(num_df.timestamp, unit='s', errors='coerce')
#set index as timestamp
num_df = num_df.set_index("timestamp")

num_df['info'] = None
for i in range(len(most_valuable_nfts)):
    address = most_popular_nfts.iloc[i].name
    
    num_df.loc[num_df.nft_address == address, 'info'] = most_popular_nfts.at[address, 'info'] #most_valuable_nfts.at[num_df.iloc[i]['nft_address'], 'info']

num_df

# COMMAND ----------

#group timestamps by day, create column per each nft_address, aggregate transaction value by count and sum
new_df = num_df.groupby([pd.Grouper(freq='d'), 'nft_address', 'info'])['transaction_value'].agg(transaction_value="count").rename(columns={'transaction_value': 'count'})

# COMMAND ----------

new_df

# COMMAND ----------

plt.figure(figsize=(24, 12))
# new_df.unstack()
ax = sns.lineplot(data=new_df, x='timestamp', y='count', hue='info')

# ax.set(yscale="log")
