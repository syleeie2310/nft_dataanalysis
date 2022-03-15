# Databricks notebook source
# MAGIC %md
# MAGIC # csv 페이지
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ----
# MAGIC -- 마켓 데이터 
# MAGIC - Ether Daily Price (USD) Chart : https://etherscan.io/chart/etherprice
# MAGIC - Ether Market Capitalization Chart : https://etherscan.io/chart/marketcap
# MAGIC - Ether Supply Growth Chart : https://etherscan.io/chart/ethersupplygrowth  
# MAGIC 
# MAGIC ----
# MAGIC -- 블록체인 데이터
# MAGIC - Ethereum Daily Transactions Chart : https://etherscan.io/chart/tx
# MAGIC - Ethereum Unique Addresses Chart : https://etherscan.io/chart/address
# MAGIC - Ethereum Average Block Size Chart : https://etherscan.io/chart/blocksize
# MAGIC - Ethereum Average Block Time Chart : https://etherscan.io/chart/blocktime
# MAGIC - Ethereum Average Gas Price Chart : https://etherscan.io/chart/gasprice
# MAGIC - Ethereum Average Gas Limit Chart : https://etherscan.io/chart/gaslimit
# MAGIC - Ethereum Daily Gas Used Chart : https://etherscan.io/chart/gasused
# MAGIC - Ethereum Daily Block Rewards Chart : https://etherscan.io/chart/blockreward 
# MAGIC - Ethereum Block Count and Rewards Chart : https://etherscan.io/chart/blocks
# MAGIC - Ethereum Uncle Count and Rewards Chart : https://etherscan.io/chart/uncles
# MAGIC - Active Ethereum Addresses : https://etherscan.io/chart/active-address
# MAGIC - Average Transaction Fee Chart : https://etherscan.io/chart/avg-txfee-usd  
# MAGIC 
# MAGIC ----
# MAGIC -- 네트워크 데이터
# MAGIC - Ethereum Network Hash Rate Chart : https://etherscan.io/chart/hashrate
# MAGIC - Ethereum Network Difficulty Chart : https://etherscan.io/chart/difficulty
# MAGIC - Ethereum Network Pending Transactions Chart : https://etherscan.io/chart/pendingtx
# MAGIC - Ethereum Network Transaction Fee Chart : https://etherscan.io/chart/transactionfee
# MAGIC - Ethereum Network Utilization Chart : https://etherscan.io/chart/networkutilization
# MAGIC 
# MAGIC ----
# MAGIC - Ethereum Daily Verified Contracts Chart : https://etherscan.io/chart/verified-contracts
# MAGIC -
# MAGIC -
# MAGIC -

# COMMAND ----------

# MAGIC %md
# MAGIC # 데이터 수집 관련해서 생각난거 -> 이더스캔 stats
# MAGIC 
# MAGIC 데이터브릭스에서는 셀레니움을 어떻게 돌리는지 잘 모르겠는데   
# MAGIC 로컬에서 사용할때는 크롬드라이버 해당 위치에 다운해서 하면 좋을듯.

# COMMAND ----------

# 마켓데이터 csv 다운로드 함수 
def Etherscan_Charts_Statistics():
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    import time
    # options = webdriver.ChromeOptions()
    # options.add_argument("headless")
    # driver 생성
    chromedriver = 'chromedriver' #chromedriver의 위치
    driver = webdriver.Chrome(chromedriver)
    driver.implicitly_wait(3)
    driver.set_window_size(2560, 1440)
#     Ether Daily Price (USD) Chart
    driver.get("https://etherscan.io/chart/etherprice")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
#     Ether Market Capitalization Chart
    driver.get("https://etherscan.io/chart/marketcap")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ether Supply Growth Chart
    driver.get("https://etherscan.io/chart/ethersupplygrowth")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    
    ## 블록체인 데이터
    #     Ethereum Daily Transactions Chart
    driver.get("https://etherscan.io/chart/tx")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Unique Addresses Chart
    driver.get("https://etherscan.io/chart/address")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Average Block Size Chart
    driver.get("https://etherscan.io/chart/blocksize")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Average Block Time Chart
    driver.get("https://etherscan.io/chart/blocktime")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Average Gas Price Chart
    driver.get("https://etherscan.io/chart/gasprice")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Average Gas Limit Chart
    driver.get("https://etherscan.io/chart/gaslimit")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Daily Gas Used Chart
    driver.get("https://etherscan.io/chart/gasused")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Daily Block Rewards Chart
    driver.get("https://etherscan.io/chart/blockreward")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Block Count and Rewards Chart
    driver.get("https://etherscan.io/chart/blocks")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Uncle Count and Rewards Chart
    driver.get("https://etherscan.io/chart/uncles")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Active Ethereum Addresses
    driver.get("https://etherscan.io/chart/active-address")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Average Transaction Fee Chart
    driver.get("https://etherscan.io/chart/avg-txfee-usd")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    
    ## 네트워크 데이터 
    #     Ethereum Network Hash Rate Chart
    driver.get("https://etherscan.io/chart/hashrate")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Network Difficulty Chart
    driver.get("https://etherscan.io/chart/difficulty")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Network Pending Transactions Chart
    driver.get("https://etherscan.io/chart/pendingtx")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    #     Ethereum Network Transaction Fee Chart
    driver.get("https://etherscan.io/chart/transactionfee")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    
    #     Ethereum Network Utilization Chart 
    driver.get("https://etherscan.io/chart/networkutilization")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    
    ########
    #     Ethereum Daily Verified Contracts Chart
    driver.get("https://etherscan.io/chart/verified-contracts")
    driver.find_element_by_link_text('CSV Data')
    time.sleep(1)
    driver.find_element_by_link_text('CSV Data').click()
    time.sleep(10)
    
    

    driver.quit()
    

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


