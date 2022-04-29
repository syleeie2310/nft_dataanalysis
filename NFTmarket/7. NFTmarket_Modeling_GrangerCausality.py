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
# MAGIC ### 미니 EDA
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
# MAGIC ### 미니 시각화
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
# MAGIC ### 데이터 통합

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
# MAGIC ### 미니 상관분석
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

# 주간 2021년 이후 데이터 : nft검색량이 급등한 21년도부터 차이가 분명하다, utility의 상관성이 다시 높아진것에 반면 defi는 낮아짐.
# nft_gt기준 metaverse, collectible, art 순으로 상관성이 가장 높다.
heatmapF(totalW['2021':], 'average_usd')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 상관분석 결과
# MAGIC - 21년도 이후부터 분석하면 될듯
# MAGIC ---
# MAGIC ##### all카테고리, 피처별 상관관계
# MAGIC - 주간 2021년 이후 데이터 : gt데이터가 급등한 21년도부터 상관성이 분명해짐,  avg_usd의 상관성이 약해졌으나 가격류는 유지됨
# MAGIC - **분석 피처 셀렉션 : 총매출, 총판매수, 총사용자수, 총평균가**
# MAGIC   - 상관성이 높고 시장흐름을 이해할 수 있는 주요 피처를 선정
# MAGIC ---
# MAGIC ##### avgusd피처, 카테고리별 상관관계
# MAGIC - 주간 2021년 이후 데이터 : nft검색량이 급등한 21년도부터 차이가 분명하다, utility의 상관성이 다시 높아진것에 반면 defi는 낮아짐. nft_gt기준 metaverse, collectible, art 순으로 상관성이 가장 높다.
# MAGIC - **분석 피처 셀렉션 : metaverse, collectible, art, game**
# MAGIC   - 위에서 선정한 주요피처중에 가장 상관성이 낮아 해석이 용이할 것으로 추정되는 avgusd를 기준으로 다시 상관성과 매출 비중이 높은 주요 카테고리로 선정한다.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 시차상관분석

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
# MAGIC ##### all_flist 결과(2021년 이후 주간기준)
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
# MAGIC ##### avgusd_clist 결과(2021년 이후 주간기준)
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
# MAGIC ### 공적분 검정
# MAGIC - 앵글&그레인저, 주간데이터

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
def URT(data, x1, x2list, Trend):
   
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
URT(totalW, 'nft_gt', all_flist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# 2021년도 이후 주간데이터 기준, nftgt와 모두 장기적 연관성이 없다. -> 기간이 너무 짧은 듯, nft_gt검색량이 급등 전후의 데이터가 충분히 반영되지 않은 듯
URT(totalW['2021'], 'nft_gt', all_flist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### all_flist 결과(CTT기준)
# MAGIC - 2018년도 이후 : 매출, 판매수, 사용자수 모두 장기적 연관성이 있다.
# MAGIC - 2021년도 이후 : 모두 없음

# COMMAND ----------

# 21년도 이후를 보면 1번 2번이 있어보인다. 직접 검정해보자.
x1x2plot(totalW, 'nft_gt', avgusd_clist[1:])

# COMMAND ----------

# 위 그래프를 볼때 2차 기울기 추세임을 알 수 있다. CTT
# 2018년도 이후 주간데이터 기준, art외에 모두 nft_gt와 장기적 연관성이 있다.
URT(totalW, 'nft_gt', avgusd_clist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# 2021년도 이후 주간데이터 기준, 모두 없음
URT(totalW['2021':], 'nft_gt', avgusd_clist[1:], 'ctt') # trend : C(상수), CT(상수&기울기), CTT(상수&기울기2차), NC(추세없음)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### all_flist 결과(CTT기준)
# MAGIC - 2018년도 이후 : game, collectible, metaverse 모두 장기적 연관성이 있다. (art도 있긴하다. pval값이 조금 아쉬움)
# MAGIC - 2021년도 이후 : 모두 없음

# COMMAND ----------

# MAGIC %md
# MAGIC ### 상관분석결과 종합
# MAGIC #### 상관분석
# MAGIC - nft_18년도 이후부터 마켓변수들과 모두 상관성이 있는데, 그중 본격적으로 검색량이 발생하는 21년도부터 각 마켓변수들과의 상관성이 분명함,
# MAGIC   - all카테고리, 세일즈피처별 상관관계 : 분석용 변수 1차 셀렉션 (총매출, 총판매수, 총사용자수, 총평균가)
# MAGIC   - avgusd피처, 카테고리별 상관관계 : 분석용 변수 1차 셀렉션 (metaverse, collectible, art, game)
# MAGIC  
# MAGIC #### 시차상관분석
# MAGIC - 위와 동일하게 21년도 기준 상관성이 분명하게 드러남
# MAGIC - nft_gt와 기존 마켓변수들과 시차상관성을 확인, 상호지연관계로 또다른 제3의 변수가 있지만 시간관계상 pass
# MAGIC - 공적분 검정을 통해 셀렉션한 피처들을 좀더 줄여보자
# MAGIC 
# MAGIC #### 공적분 검정
# MAGIC - 이 경우는 오히려 2018년도까지 포함해야 장기적 관계를 확인할 수 있다.(검색량 유, 무 정보를 모두 포함하기 때문)
# MAGIC   - all카테고리, 세일즈피처별 공적분 검정 : 평균가는 장기적 관계 없음
# MAGIC   - avgusd피처, 카테고리별 공적분 검정 : art의 pval값이 근소함을 고려시 모두 장기적관계가 있다.
# MAGIC ---
# MAGIC #### 그레인저 인과검정을 위한 최종 피처 셀렉션
# MAGIC - 외부 변수 : nft 구글 검색량
# MAGIC - all카테고리, 세일즈 변수 :   총매출, 총판매수, 총사용자수
# MAGIC - avgusd피처, 카테고리별 변수 : metaverse(포함해주자), collectible, art, game

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
# MAGIC ## 정상성 검정
# MAGIC ### 2번 노트북(TSA)에서 전체 변수들 모두 정상성 시차 1을 확인
# MAGIC - 데이터 : 주간 (일/주/월 결과 모두 다름, 외부변수와 함께 비교하려면 주간통일이 편함), 정규화 안함(데이터 특징 우려, )
# MAGIC - 대표칼럼 : avgusd 피처, 카테고리별 검정
# MAGIC - 대표칼럼 : all카테고리, 피처별 검정
# MAGIC ---
# MAGIC #### 1. Augmented Dickey-Fuller("ADF") Test
# MAGIC - ADF 테스트는 시계열이 안정적(Stationary)인지 여부를 확인하는데 이용되는 방법입니다.
# MAGIC - 시계열에 단위근이 존재하는지 검정,단위근이 존재하면 정상성 시계열이 아님.
# MAGIC - 귀무가설이 단위근이 존재한다.
# MAGIC - 검증 조건 ( p-value : 5%이내면 reject으로 대체가설 선택됨 )
# MAGIC - 귀무가설(H0): non-stationary. 대체가설 (H1): stationary.
# MAGIC - adf 작을 수록 귀무가설을 기각시킬 확률이 높다.
# MAGIC 
# MAGIC #### 2. Kwiatkowski-Phillips-Schmidt-Shin (“KPSS”) Test
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

def URT(data, col, dataclass) :
    # 칼럼 분류기 호출
    if dataclass == 'feature':
        col_list = feature_classifier(data, col)
    elif dataclass == 'category':
        col_list = category_classifier(data, col)
    else :
        print('분류기 조건을 확인하세요')
        
    col_list.append('nft_gt')
        
    adf_stats = []
    adf_Pval = []
    kpss_stats = []
    kpss_Pval = []
    total_list = []
    
    for col in col_list:
#         print(f'<<<<{col}>>>>')
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
            total_list.append('ALL Pass')
        elif adf_result[1] <= 0.05 or kpss_result[1] >= 0.05:
            total_list.append('One Pass')
        else :
            total_list.append('fail')
        
    # 테이블 생성
#     col_list.append('total')
    result_df = pd.DataFrame(list(zip(adf_stats, adf_Pval, kpss_stats, kpss_Pval, total_list)), index = col_list, columns=['adf_stats', 'adf_Pval', 'KPSS_stats', 'KPSS_Pval', 'total'])
    
#     # adf stats가 낮은 순으로 정렬
#     result_df.sort_values(sort, inplace=True)
    
    return result_df             

# COMMAND ----------

# MAGIC %md
# MAGIC ### avgusd피처, 카테고리별 검정
# MAGIC -

# COMMAND ----------

dataW_diff = dataW_median.diff(periods=1).dropna()

# COMMAND ----------

# 로그 차분 데이터, 서로 다른 변수의 회귀결과에 대한 분석을하려면 정규화가 필요하다. 
logdata = np.log1p(data)
logdiff = logdata.diff(periods=1).dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC #### [Raw + 주간]
# MAGIC -

# COMMAND ----------

# 로그차분 + 주간중앙 + 전체기간
URT(dataW_diff, 'average_usd', 'feature')

# COMMAND ----------

# MAGIC %md
# MAGIC #### [Log + 주간]
# MAGIC - 

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# 로그차분 + 주간중앙 + 18년 이후
URT(dataW_diff['2018':], 'average_usd', 'feature')

# COMMAND ----------

# 로그차분 + 주간중앙 + 21년 이후
URT(dataW_diff['2021':], 'average_usd', 'feature')

# COMMAND ----------

URT(totalW.diff(periods=1).dropna(), 'average_usd', 'feature')

# COMMAND ----------

URT(totalW['2021':].diff(periods=1).dropna(), 'average_usd', 'feature')

# COMMAND ----------

# MAGIC %md
# MAGIC ### ALL카테고리, 피처별 검정

# COMMAND ----------

# MAGIC %md
# MAGIC #### [주간 기간별 결과]
# MAGIC - avgusd가 정상성이 있다. -> 전체기간을 보려면 avgusd간의 인과분석을 해보자
# MAGIC - 21년도만 볼거면 실패한 피처들은 참고하고 분석해야함

# COMMAND ----------

# 주간중앙 + 차분 + 전체기간
URT(dataW_diff, 'all', 'category')

# COMMAND ----------

# 주간중앙 + 차분 + 18~ 
URT(dataW_diff['2018':], 'all', 'category')

# COMMAND ----------

# 주간중앙+ 차분 + 21~ 
URT(dataW_diff['2021':], 'all', 'category')

# COMMAND ----------

URT(totalW.diff(periods=1).dropna(), 'all', 'category')

# COMMAND ----------

URT(totalW['2021':].diff(periods=1).dropna(), 'all', 'category')

# COMMAND ----------

카테고리내 피처들은 21년도부터 정상성을 찾음
avgusd는 전체기간 정상성을 가짐.. 한결같다고나 할까..

카테고리내 피처들과 비교하려면 2021년도 기준으로 분석해야한다.

이것 끼리 진행하고.

1. all카테고리, 주간 , 21년도기준
'사용자수, 구글트랜드, 판매단가, 판매수, 총매출'

2. avgusd, 주간, 21년도 기준
'all, art, collectible, game, nft_gt'

# COMMAND ----------

# MAGIC %md
# MAGIC ### [결과 종합 ]
# MAGIC - avgusd피처집단과 all카테고리집단들이 모두 검정이 통과한 조건은 **<일간 전체, 일간 18년 이후>** 이다
# MAGIC - 일단 **<일간 + 2018년도 이후>**로 인과검정 진행해보자 
# MAGIC ---
# MAGIC 
# MAGIC #### avgusd피처, 카테고리별 정상성 검정
# MAGIC   - 일간 통과 : 전체, 18년 이후, 21년 이후
# MAGIC   - 주간 통과 : 전체 기간 only
# MAGIC   - 월간 통과 : 21년이후 기간 only
# MAGIC 
# MAGIC #### all카테고리, 피처별 정상성 검정
# MAGIC   - 일간 통과 : 전체, 18년 이후
# MAGIC   - 주간 통과 : 3개 기간 모두 불통
# MAGIC   - 월간 통과 : 3개 기간 모두 불통

# COMMAND ----------

# MAGIC %md
# MAGIC ## 그레인저 인과분석
# MAGIC - 딕셔너리 언패킹을 못해서 시각화못함
# MAGIC - from statsmodels.tsa.stattools import grangercausalitytests [signature](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html)
# MAGIC   - 2개 시계열의 그랜저 비인과성에 대한 4가지 테스트.
# MAGIC   - 현재 일간데이터 길이 기준 maxlag = 15가 최대
# MAGIC   - 2번째 시계열이 1번째 시계열을 유발하는지 테스트(2->1) -> 즉 2번째열이 시차 보행하는 것
# MAGIC     - 그런데 lag를 양수만 입력가능하므로, 이는 X2의 과거lag값임.
# MAGIC     - 결국 X2의 t가 -n일 때의 X1회귀값의 pvalue, 즉, X2의 과거가 x1의 현재값을 통계적으로 유의미하게를 예측할 수 있는지를 본다

# COMMAND ----------

# MAGIC %md
# MAGIC ### 일간+18년이후

# COMMAND ----------

from statsmodels.tsa.stattools import grangercausalitytests

# COMMAND ----------

# collectible -> game, X2 lag 1~2까지 귀무가설 기각
from statsmodels.tsa.stattools import grangercausalitytests
grangercausalitytests(data[['game_average_usd', 'collectible_average_usd']]['2018':].diff(periods=1).dropna(), maxlag=15)

# COMMAND ----------

#  game -> collectible, x2 lag 1~15까지 귀무가설기각
grangercausalitytests(data[['collectible_average_usd', 'game_average_usd']]['2018':], maxlag=15)

# COMMAND ----------

# MAGIC %md
# MAGIC ### collectible_avgusd & game_avgusd
# MAGIC - ***정상성 시차 : 1
# MAGIC - ***그레인저인과 시차 : cg 1 ~ 2, gc1 ~ 15***
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
grangercausalitytests(data[['all_average_usd', 'all_unique_buyers']]['2018':].diff(periods=1).dropna(), maxlag=15)

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


