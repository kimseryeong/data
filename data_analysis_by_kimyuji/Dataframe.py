#!/usr/bin/env python
# coding: utf-8

# # 🫧 기술통계
# ## - 데이터 표본의 특징을 나타내는 요약된 정보.

# ## ✔ describe() : 기술통계 정보를 하나의 테이블로 제공하는 함수 -> include/exclude 활용
# - 결측값을 제외한 데이터 분포의 중심 경향, 분산, 차원의 모양을 요약하는 통계 포함됨
# - 숫자 또는 오브젝트 데이터 타입인 변수의 집합에 관해서도 분석 가능

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[3]:


#시본에서 제공하는 펭귄 데이터셋 로드
#수치형 변수 관련 정보만 출력

df = sns.load_dataset('penguins') 
df.describe()


# In[4]:


#수치형 변수 뿐만 아니라 모든 변수를 포함한 정보 출력

df.describe(include='all')


# In[5]:


#오브젝트 타입만 확인 -> 테이블 인덱스에 데이터 수(count), 고윳값 수(unique), 가장 많이 출연하는 값(top), 빈도(freq)

df.describe(include=[object])


# In[6]:


#숫자 데이터 타입만 출력

df.describe(include=[np.number])


# In[9]:


#카테고리 데이터 타입 제외하고 출력

df.describe(exclude=['category'])


# ## ✔ percentile() : 넘파이로 백분위수 구하는 함수
# 
# - 데이터에 결측값이 있다면 에러가 발생할 수 있음

# In[10]:


df = df.fillna(0) #에러방지 결측값 0으로
point_5 = np.percentile(df['bill_depth_mm'], q=[0, 25, 50, 75, 100])
point_5


# In[12]:


#개별적인 원소에 접근

print(point_5[0])
print(point_5[2])
print(point_5[4])


# ## ✔ quantile() : 판다스로 백분위수 구하는 함수
# 
# - 함수의 ()에 리스트로 원하는 백분위수를 소수점으로 입력하면 해당 값 출력
# - 하나의 백분위수도 입력 가능

# In[13]:


df.quantile([0, .25, .5, .75, 1.0])


# In[14]:


df.quantile(0.25)


# In[15]:


df['bill_length_mm'].quantile(.25)


# ## count() : 데이터 수 파악하기
# - 컬럼이나 로우를 기준으로 결측값이 아닌 모든 데이터의 셀 수를 계산.
# - None, NaN, NaT, numpy.inf 값 등이 결측값으로 인지됨

# In[16]:


df.count() #결측값을 0으로 바꿔서 모든 데이터 344개로 나옴


# In[17]:


df = sns.load_dataset('penguins') 
df.count()


# In[18]:


df['bill_depth_mm'].count()


# In[19]:


df.count(axis='columns') #개별 행에 포함된 컬럼 수


# ## max() : 최댓값 찾기
# - 해당 데이터의 기준 축에서 가장 큰 값 출력

# In[20]:


df.max()


# In[21]:


df['bill_length_mm'].max()


# ## min() : 최솟값 찾기
# - 해당 데이터의 기준 축에서 가장 작은 값 출력
# - 오브젝트 타입의 경우 빈도가 가장 낮은 값 출력

# In[22]:


df.min()


# In[24]:


df['bill_depth_mm'].idxmin() #최솟값이 있는 인덱스 찾기


# ## mean() : 평균값 찾기
# - 숫자 타입 컬럼의 평균값만 계산

# In[25]:


df.mean()


# ## std() : 표준편차 찾기
# - 기본적으로 N-1로 정규화

# In[26]:


df.std()


# ## sum() : 데이터 합계 구하기
# - 문잣값의 경우 문자가 합쳐져서 출력

# In[27]:


df.sum()


# ## 기술통계 시각화 - 막대그래프, 히스토그램, 박스플롯

# ## 막대그래프

# ### catplot() : 시본에서 합계를 표현하는 막대그래프

# In[38]:


#오브젝트 타입이 species / island 2개

sns.catplot(data=df, x="species", kind="count")


# In[39]:


sns.catplot(data=df, x="island", kind="count")


# ### barplot() : 시본에서 평균이나 계산한 결과를 표현하는 막대그래프
# - y축에 숫자 타입 변수 지정

# In[36]:


sns.barplot(data=df, x="species", y="bill_length_mm")


# In[37]:


sns.barplot(data=df, x="island", y="bill_length_mm")


# ## 히스토그램

# In[42]:


sns.histplot(data=df, x="flipper_length_mm")


# In[43]:


sns.histplot(data=df, x="body_mass_g")


# ## 박스플롯
# - 변수의 분포를 표시
# - 숫자와 오브젝트 타입 모두 가능
# - 변수 간 수준 쉽게 비교 가능

# In[44]:


sns.boxplot(data=df)


# In[46]:


sns.boxplot(data=df, x = "species", y="body_mass_g")


# # ---------------------------------------------------------------------------------------------------------------
# # 🫧 고윳값 확인 : unique()
# - 넘파이 배열 형식으로 반환.
# - 숫자 타입과 오브젝트 타입 모두 사용 가능. 주로 오브젝트에서 사용

# In[49]:


df['species'].unique() #species 컬럼의 고윳값 확인


# In[50]:


df['bill_depth_mm'].unique() #숫자 타입 컬럼의 고윳값 확인 -> 큰 의미가 없어서 잘 사용하지 않음


# ## value_counts() : 고윳값과 해당 개수 동시 확인
# - 시리즈와 데이터프레임 모두 사용할 수 있음

# In[54]:


df['species'].value_counts() #시리즈에 적용


# In[55]:


df['species'].value_counts(normalize=True) #비중으로 확인


# ### bins : 고윳값 세는 대신 전체 데이터를 지정한 수 기준으로 인덱스를 나누어 계산
# #### bins를 3개와 5개로 나누어 어느 구간에 데이터가 밀집해 있는지 어느 구간에 어느 정도의 데이터가 존재하는지 정확하게 수치로 구할 수 있음

# In[57]:


df['bill_depth_mm'].value_counts(bins=5)


# In[58]:


df['bill_depth_mm'].value_counts(bins=3)


# In[60]:


df.value_counts(dropna=False) #결측값도 포함해서 고윳값 확인 


# In[56]:


df.value_counts() #데이터 프레임에 적용


# # ---------------------------------------------------------------------------------------------------------------
# # 🫧 현재 컬럼 목록 확인

# ## columns : 데이터셋의 전체 컬럼명 확인 

# In[61]:


df.columns


# ## 컬럼 호출 / 2개 이상의 컬럼 조합해서 데이터프레임 생성

# In[62]:


df['species']


# In[65]:


df.species


# In[64]:


df[['species', 'bill_depth_mm']]


# In[66]:


cols=['species', 'bill_depth_mm']
df[cols]


# In[67]:


new_df = df[cols]
new_df


# ## 새로운 컬럼 생성

# ### 단일 컬럼 생성하기 : df['생성할 컬럼명']

# In[68]:


df['bill_depth_cm'] = df['bill_depth_mm'] / 10
df.head()


# ### 다중 컬럼 동시에 생성하기 : assign()

# In[69]:


df.assign(
    bill_length_cm = df['bill_length_mm'] / 10
    ,bill_depth_cm = df['bill_depth_mm'] / 10
)


# ## 동일한 데이터 타입의 컬럼만 선택하기 : select_dtypes()

# In[71]:


df.select_dtypes(include=['float64']).columns


# In[72]:


df.select_dtypes(include=['object']).columns #string 속성을 지원하지 않으므로 object로 입력


# In[74]:


df.select_dtypes(exclude=['object']).columns #특정 dtype만 제외


# ## 컬럼과 로우 삭제하기 : drop()
# - 불필요한 컬럼이나 로우 삭제
# - axis : 세로 1

# In[75]:


df.drop('species', axis=1).head()


# In[76]:


df.drop(['species', 'flipper_length_mm'], axis=1).head()


# In[77]:


df.drop([0, 3, 5], axis=0).head() #특정 인덱스의 로우 삭제


# In[81]:


df.drop([2]).head() #axis 생략 가능


# ## 컬럼 이름 변경하기

# In[82]:


df.rename(columns={'species':'펭귄종류', 'island':'서식하는 섬', 'bill_length_mm':'부리 길이'}).head()


# In[84]:


df.columns = ['펭귄 종류', '서식하는 섬', '부리 길이', '부리 깊이', '지느러미 길이', '몸무게', '성별', '부리 깊이(cm)']
df.head()


# ## 컬럼 순서/위치 변경하기

# In[88]:


#컬럼명을 원하는 순서대로 재입력하면 컬럼 순서 변경 가능

df = sns.load_dataset('penguins')
df = df[['species', 'island', 'sex', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
df.head()


# In[89]:


#columns 함수 사용하여 순서 지정
#첫번째 컬럼 기준 0

df.columns = df.columns[[0, 1, 6, 2, 3, 4, 5]]
df.head()


# In[90]:


#for문과 if문을 사용해서 일부 컬럼 이동하기

colsToStay = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
cols = colsToStay + [col for col in df if col not in colsToStay]
df = df[cols]
df.head()


# ## 컬럼과 인덱스 교환 : transpose()

# In[94]:


df[:11].transpose()


# In[93]:


df[:11].T


# # ---------------------------------------------------------------------------------------------------------------
# # 🫧 데이터 인덱싱 

# ## 문자형 인덱스 인덱싱하기

# In[95]:


df = sns.load_dataset('penguins')
df = df[:11].transpose()
df


# In[96]:


#하나의 문자형 인덱스에 접근하기 
#loc['인덱스명']

df.loc['bill_length_mm']


# In[97]:


#2개 이상의 문자형 인덱스 접근하기
#loc[['인덱스명', '인덱스명'...]]

df.loc[['bill_length_mm', 'sex']]


# In[98]:


#인덱스와 컬럼의 위치 지정하여 데이터 찾기

df.loc['bill_length_mm', 5]


# In[99]:


#라벨 인덱스의 연속된 구간에 해당하는 복수의 인덱스 선택

df.loc['island':'bill_depth_mm', 7]


# In[100]:


#연속되지 않고 개별 라벨 인덱스 호출

df.loc[['island', 'bill_depth_mm', 'sex'], 7]


# ### 컬럼 기준으로 인덱싱

# In[101]:


df = df.T


# In[102]:


#특정 수치를 기준으로 인덱싱

df.loc[df['bill_length_mm'] > 40]


# In[103]:


df.loc[df['bill_length_mm'] > 40, ['sex']] #특정 컬럼에 해당하는 값만 확인


# In[104]:


df.loc[df['flipper_length_mm'] == 181]


# In[105]:


df.loc[df['sex'] == 'Male']


# ### 값 변경하기
# - 기존 값이 존재하면 변경되고, 기존 값이 없으면 새로운 값 저장됨

# In[106]:


df.loc[[1, 3], ['bill_length_mm']] = 50
df.head()


# In[107]:


#인덱스 전체를 기준으로 특정 컬럼에 해당하는 값만 새롭게 저장하기

df.loc[:, 'bill_length_mm'] = 10
df.head()


# In[108]:


#190을 초과하는 값을 0으로 변경하기

df.loc[df['flipper_length_mm'] > 190] = 0
df.head()


# ## 위치 기반 인덱싱하기

# In[109]:


df = sns.load_dataset('penguins')


# In[111]:


df.loc[[0]]


# In[110]:


df.iloc[[0, 2]]


# In[112]:


df.loc[:3]


# In[113]:


df.iloc[0, 1] #x:0, y:1 위치의 데이터


# In[114]:


df.iloc[[1, 2], [2, 3]] #x:1~2, y:2~3 위치의 데이터


# In[115]:


df.iloc[1:3, 0:3]


# # 컬럼을 인덱스로 만들기, 재정의 하기

# ### 특정 컬럼 값으로 인덱스 만들기 : set_index()

# In[116]:


df.set_index('species', inplace=True)
df


# ### 컬럼을 잘못 지정하여 되돌려야 하는 경우 : reset_index()

# In[117]:


df.reset_index()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




