#!/usr/bin/env python
# coding: utf-8

# # Chapter3 데이터 정제와 응용

# # 3.1 데이터 필터링과 정렬 테크닉

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[5]:


#초보자용 실습에 주로 쓰이는 캐글의 타이타닉 데이터 로드하기

titanic = pd.read_csv('./bumping-into-data-analysis-main/datasets/titanic.csv')
titanic.head()


# ## 📌조건식을 활용한 데이터 필터링

# #### 단일 조건식 활용하여 필터링하기

# In[6]:


#객실등급이 3등급인 데이터만 추출

titanic[titanic.Pclass == 3].head()


# #### 다중 조건식 활용하여 데이터 필터링하기

# In[7]:


#객실등급이 3등급이면서 여성인 데이터만 추출

titanic[(titanic.Pclass == 3) & (titanic.Sex == 'female')].head()


# #### 특정 값을 제외한 데이터 추출하기

# In[8]:


#객실등급이 3등급이지만 여성이 아닌 데이터만 추출

titanic[(titanic.Pclass == 3) & (titanic.Sex != 'female')].head()


# ## 📌loc 조건부 필터링
# - 판다스의 인덱싱 함수인 loc를 활용하여 조건 필터링

# In[9]:


#탑승요금이 300 이상인 데이터 추출
#loc 활용

titanic.loc[titanic.Fare > 300, :]


# In[10]:


#iloc 활용

titanic.iloc[list(titanic.Fare > 300)]


# In[12]:


# 조건 여러 개인 경우 '&' 사용

titanic.loc[
    (titanic.Sex == 'female') & (titanic.Fare > 240),      #다중 조건 입력
    ['Name', 'Age', 'Fare', 'Embarked']    #특정 컬럼만 선택
]


# ## 📌특정 값 포함 여부 필터링 : isin() / 포함하지 않는 경우 '~' 삽입

# #### 단일 컬럼에서 특정 값 포함된 데이터 필터링하기

# In[13]:


#선착장이 S or C 에서 탑승한 승객 데이터

titanic[titanic.Embarked.isin(['S', 'C'])].head()


# In[14]:


titanic[~titanic.Embarked.isin(['S', 'C'])].head()


# #### 다중 컬럼에서 특정 값 포함된 데이터 필터링하기
# - isin()을 사용한 필터 코드를 미리 변수에 담아두고 데이터프레임에서 해당 변수 호출하기

# In[15]:


#성별이 남성이면서 승객 등급이 1, 2등급인 데이터 추출

filter_male = titanic.Sex.isin(['male'])
filter_pclass = titanic.Pclass.isin([1, 2])

titanic[filter_male & filter_pclass].head()


# ## 📌쿼리를 사용하여 데이터 필터링 : query()
# - 데이터 필터링에서 가장 추천하는 방법은 query() 함수 사용하는 것 !
# - 가독성이 높고 간편하다
# - 함수의 인수에 조건식 삽입
# - (주의) 오브젝트 타입의 경우 큰따옴표 사용

# In[17]:


#객실 등급이 1, 2 이면서 탑승 요금이 270 이상인 승객 데이터 추출

titanic.query('Pclass == [1, 2] & Fare > 270')


# In[20]:


#여성이고 객실 등급이 1등급이며 나이가 35살에 해당하는 승객 데이터 추출

titanic.query('Sex == "female" & Pclass == 1 & Age == 35')


# ## 📌결측값을 제외하고 데이터 필터링 : notnull()

# In[21]:


#결측값의 빈도가 가장 높은 Cabin 컬럼에서 결측값 제외한 데이터를 추출

titanic_nn = titanic[titanic.Cabin.notnull()]
titanic_nn.head()


# ## 📌특정 문자가 포함된 데이터 필터링 : contains()

# In[22]:


#승객 이름에 'Catherine'이 포함된 데이터 ㅊ출

titanic[titanic['Name'].str.contains('Catherine')].head()


# In[23]:


#승객 이름에 'Miss'이 포함된 데이터 ㅊ출

titanic[titanic['Name'].str.contains('Miss')].head()


# ## 📌다양한 기준으로 데이터 정렬

# ### sort_values(by=['기준이 되는 컬럼명'])

# In[24]:


#Fare 컬럼을 기준으로 내림차순 

titanic.sort_values(by=['Fare'], ascending=False).head()


# In[25]:


#Fare, Age 컬럼을 기준으로 내림차순

titanic.sort_values(by=['Fare', 'Age'], ascending=False).head()


# ### nlargest(n=개수, columns='정렬기준컬럼') : 내림차순

# In[26]:


#성별이 여성인 데이터 중 Age를 기준으로 내림차순. 데이터 10개  

titanic[titanic.Sex == 'female'].nlargest(n=10, columns='Age')


# ### nsmallest(n=개수, columns='정렬기준컬럼') : 오름차순

# In[27]:


#성별이 여성인 데이터 중 Age를 기준으로 오름차순. 데이터 10개  

titanic[titanic.Sex == 'female'].nsmallest(n=10, columns='Age')


# ## 📌데이터 순서 역순으로 변경하기

# In[28]:


titanic.loc[::-1].head()


# In[29]:


#데이터 역순으로 정렬하고 인덱스 새롭게 적용

titanic.loc[::-1].reset_index().head()


# In[31]:


## 컬럼의 순서를 역순으로 변경

titanic.loc[:, ::-1].head()


# # 3.2 결측값 처리
# - NaN, Null, Na
# - 실제 데이터에서는 결측값이 꽤 많이 존재함
# 
# #### [결측값 처리 방법]
# 1. 결측값이 존재하는 행 삭제 -> 데이터를 분석하는데 큰 정보가 될 수 있는 데이터 포인트를 잃을 가능성이 있음
# 2. 결측값을 다른 값으로 대체 -> 기존의 다른 데이터 포인트를 참조하여 적절한 값을 유추해야 함

# ## 📌결측값 시각화하기

# In[32]:


import matplotlib.pyplot as plt


# ### 시본의 heatmap() 함수 사용

# In[33]:


plt.figure(figsize=(12, 7))                           #그래프 출력 사이즈 지정 12 x 7
sns.heatmap(titanic.isnull(), cbar=False) #결측값이 있는 부분만 활성화해서 보여주기


# #### -> Age, Cabin, Embarked 컬럼에서 결측값 확인할 수 있고, Cabin 컬럼은 절반 이상이 결측값인 것을 확인할 수 있다

# ### missingno 라이브러리 - matrix() 함수 사용

# In[38]:


get_ipython().system('pip install missingno #missingno 설치')


# In[39]:


import missingno as msno
msno.matrix(titanic)


# ## 📌결측값 확인하기 : info()
# - 시각화를 통해 데이터셋에 결측값이 존재함을 확인했으니 정확하게 어느 컬럼에 몇 개의 결측값이 존재하는지 데이터로 확인하기

# In[40]:


titanic.info()


# -> non-null : 정상값을 의미
# -> 인덱스의 총길이는 891 이므로 그에 못 미치는 컬럼은 결측값이 존재하는다는 의미

# In[41]:


#isna()는 결측값 유무를 확인할 수 있음

titanic.isna()


# In[42]:


#sum() 을 통해 결측값 합산

titanic.isna().sum() 


# In[44]:


#컬럼 1개의 결측값을 확인

titanic.Age.isna().sum()


# In[45]:


#정상 데이터 확인하기 : notna()

titanic.notna().sum()


# ## 📌결측값 삭제/제거
# - 결측값을 시각화, 확인까지 마무리 했으면 결측값을 처리하는데 고민을 해야 한다
# 
# ------------------------------------------------------
# 
# ex) '타이타닉 승객의 생존 여부'에 대한 데이터 분석을 한다고 가정할 때, 결측값 처리와 관련한 고민
# 1. 결측값이 가장 많이 있는 Cabin 컬럼은 오히려 있는 것이 데이터 분석에 방해되지 않을까?
# 2. 2개 이상의 컬럼에 결측값이 들어 있는 행은 승객의 생존 여부를 분석하는데 도움이 될까?
# 
# ---
# 
# 데이터 셋에 많은 변수가 있다면 2개 이상의 결측값이 있는 행은 삭제해도 무리가 없을 것. 그러나 결측값이 1개만 있는 경우 그 행이 데이터를 분석하는 데 결정적인 역할을 하는 행이라면 해당 행을 삭제하면 큰 손실이 될 수도 있다. 따라서 본인의 상황에 따라 적합하게 적용해야 한다.

# ### 결측값이 존재하는 컬럼/로우 삭제하기

# In[46]:


#dropna() : 전체 데이터셋을 기준으로 결측값이 1개 이상인 행이 모두 삭제됨

titanic.dropna()


# ▶ 원본데이터의 인덱스가 891인데 183로우만 출력된 것을 보면 상당히 많은 데이터가 삭제된 것!

# In[47]:


#컬럼 축을 기준으로 결측값이 있는 컬럼 삭제 : axis='columns'

titanic.dropna(axis='columns')


# ### 모든 컬럼에서 결측값의 개수가 특정 수치를 넘어가는 행만 삭제하기

# In[48]:


#how - any : 1개 이상의 컬럼에 결측값이 있는 행, 디폴트 값 (생략가능)
#how - all : 모든 컬럼에 결측값이 있는 행

titanic.dropna(how='any')


# #### 결측값 개수의 임곗값을 설정해서 삭제하는 방법

# In[49]:


#2개 이상의 컬럼에 결측값이 존재하는 데이터는 삭제하기

titanic.dropna(thresh=2)


# In[50]:


#컬럼을 지정하여 결측값이 있는 행만 삭제할 수 있음

titanic.dropna(subset=['Age', 'Embarked']) # 둘 중 하나라도 결측값이 존재하는 행은 삭제됨


# ## 📌결측값 대체/보간

# ### 특정 값으로 결측값 채우기

# In[51]:


#Age 컬럼의 결측값을 25로 채우기
#fillna()

titanic.Age.fillna(25)


# In[52]:


#Age 컬럼의 결측값을 25로 채우기
#to_replace()

titanic.Age.replace(to_replace = np.nan, value= 25)


# ### 평균 값으로 결측값 채우기

# In[56]:


titanic.Age.fillna(titanic.Age.mean())


# ▶ 가장 좋은 방법은 아님. 범위를 좁혀서 Pclass(객실등급)를 활용하여 나이를 유추해볼 수 있겠다

# In[55]:


print(titanic[titanic.Pclass == 1].Age.mean())
print(titanic[titanic.Pclass == 2].Age.mean())
print(titanic[titanic.Pclass == 3].Age.mean())


# In[57]:


titanic[titanic.Pclass == 1].Age.fillna(titanic[titanic.Pclass == 1].Age.mean())


# ### 결측값의 전후 값 참조해서 채우기

# In[58]:


titanic.Cabin.fillna(method='ffill') #앞의 데이터 참조 : ffill(or pad) / 뒤의 데이터 참조 : bfill(or backward)


# ### 보간법으로 결측값 채우기

# In[60]:


#결측값을 기준으로 위 아래 중간값으로

titanic.Age.interpolate(method='linear', limit_direction='forward').head(10)


# In[61]:


#결측값을 기준으로 가장 가까운 값

titanic.Age.interpolate(method='nearest', limit_direction='forward').head(10)


# # 3.3 이상값 처리
# - 이상값은 시각화 기법을 활용하여 변수의 분포를 확인 후 처리
# - 변수가 1개인 경우는 박스플롯이나 히스토그램 사용, 2개 이상은 산점도 활용
# - 이상값을 처리하는데는 다양한 방법이 있는데 적절한 상황에 맞는 방법을 고려해야 함

# ## 📌박스플롯

# In[102]:


from IPython.display import Image
Image('./boxplot.png')


# 양끝은 최소와 최대를 나타내며 그 범위를 벗어난 값이 이상값

# #### 타이타닉의 Fare 변수를 대상으로 시본으로 박스플롯 그려서 확인하기

# In[65]:


sns.set_theme(style="whitegrid")


# In[67]:


plt.figure(figsize=(12, 4))
sns.boxplot(x=titanic.Fare)


# In[68]:


plt.figure(figsize=(12, 4))
sns.boxplot(x=titanic.Age)


# ## 📌 IQR 기법으로 이상값 확인하기
# 
# - 시각화 기법을 통해 이상치가 어느 위치에 분포하는지 확인했으니 실제 이상값을 추출해야 함
# 
# #### [IQR 기법 적용하는 순서]
# 1. 1사분위수 Q1을 찾는다
# 2. 3사분위수 Q3를 찾는다
# 3. IQR을 계산한다 (Q3 - Q1)
# 4. 상한값 : Q3 + 1.5 * IQR / 하한값 : Q1 - 1.5 * IQR
# 5. 상한값과 하한값을 벗어난 범위의 값을 이상값으로 정의

# In[72]:


# IQR 기법으로 함수 생성

def outlier_iqr(data, col):
    global lower, upper #하한값, 상한값 변수
    
    q1, q3 = np.quantile(data[col], .25), np.quantile(data[col], .75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    
    print('IQR : ', iqr)
    print('하한값 : ', lower)
    print('상한값 : ', upper)
    
    data1 = data[data[col] > upper]
    data2 = data[data[col] < lower]
    
    return print('이상값의 총 개수 : ', data1.shape[0] + data2.shape[0])


# In[73]:


#타이타닉 데이터셋의 Fare 변수로 이상값 분류하기

outlier_iqr(titanic, 'Fare')


# ## 📌히스토그램
# - matplotlib의 axvspan을 활용하여 이상값의 범위를 강조하자

# In[76]:


plt.figure(figsize=(12, 7))
sns.distplot(titanic.Fare, bins=50, kde=False)

#이상값 영역 박스 그리기
plt.axvspan(xmin=lower, xmax=titanic.Fare.min(), alpha=0.2, color='red')  #하한
plt.axvspan(xmin=upper, xmax=titanic.Fare.max(), alpha=0.2, color='red') #상한


# ## 이상값 삭제하기

# In[77]:


# 하한값 ~ 상한값 사이의 데이터는 정상값

titanic[(titanic['Fare'] < upper) & (titanic['Fare'] > lower)]


# ▶ 상황에 따라 데이터가 많다면 이상값을 제외한 데이터만 사용해서 데이터 분석을 하는 것이 좋은 결과를 가져올 수도 있음 !

# ## 📌이상값 대체하기

# #### 이상값을 평균값으로 대체하기
# ##### 타이타닉의 원본 데이터에서 이상값에 해당하는 인덱스만 선택하여 그 값을 해당 변수의 평균값으로 저장하는 것

# In[78]:


#이상값 선택

outlier = titanic[~(titanic['Fare'] < upper) & (titanic['Fare'] > lower)] #이상값 추출
outlier


# In[79]:


#이상값의 인덱스 추출하여 리스트에 담기

outlier_idx = [outlier.index]
outlier_idx


# In[80]:


#추출한 인덱스의 값을 평균값으로 대체
#Fare 컬럼 

titanic.iloc[outlier_idx, 9] = titanic['Fare'].mean()
titanic.head()


# # 3.4 문자열 데이터 처리
# - Pandas는 'object-dtype',  'StringDtype' 2가지의 문자열 타입이 존재함 (Pandas 공식은 StringDtype을 권고중)

# In[81]:


#타이타닉 데이터에서 필요없는 변수 미리 삭제

titanic.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)


# In[82]:


#문자열 변수인 'Name' 타입 확인

titanic.Name.dtype


# ▶ obejct-dtype 으로 출력됨

# In[83]:


#안정적인 string 타입으로 변환

titanic.Name = titanic.Name.astype("string")
titanic.Name.dtype


# ## 📌문자열 분리하기

# In[86]:


titanic.Name


# ▶ 타이타닉의 Name 변수는 ','로 나누어진 2개 이상의 단어로 구성되어 있음 -> split(pat=",")

# In[87]:


titanic.Name.str.split(pat=",")


# In[88]:


#문자열이 분리되면서 단어 수만큼 개별 컬럼이 생성됨

titanic.Name.str.split(expand=True)


# In[89]:


#분리된 Name에서 '호칭' 접근하기

titanic.Name.str.split().str[1]


# In[90]:


#호칭을 'title' 컬럼에 담기

titanic['Title'] = titanic.Name.str.split().str[1]


# ## 📌문잣값 교체하기
# - 데이터에 불필요한 기호가 포함되어 있을 때 주로 활용
# - replace() 사용

# In[92]:


titanic.Title.head()


# In[94]:


titanic.Title.str.replace(".", "", regex=False) #regex - True : 정규표현식, False : 입력한 문자 그대로


# In[95]:


titanic['Title'] = titanic.Title.str.replace(".", "", regex=False)
titanic.head()


# In[96]:


#Title 고윳값 확인

titanic.Title.value_counts()


# In[97]:


#Title 호칭 중 Mlle -> Miss, Ms -> Miss, Mme -> Mrs 과 같은 의미이므로 변경해주기

titanic['Title'] = titanic['Title'].str.replace('Mlle', 'Miss', regex=False)
titanic['Title'] = titanic['Title'].str.replace('Ms', 'Miss', regex=False)
titanic['Title'] = titanic['Title'].str.replace('Mme', 'Mrs', regex=False)


# In[100]:


#비중이 적은 별 의미 없는 호칭들은 'Rare'로 변경
#복수의 값을 변경할 때는 리스트 타입으로 묶어서 판다스의 replace 사용

rareName = ['Dr', 'Rev', 'y', 'Planke,', 'Impe,', 'Gordon,', 'Col', 'Major', 'Melkebeke,', 'Jonkheer', 'Shawah,', 'the', 'Velde,', 'Capt'
            , 'Messemaeker,', 'Carlo,', 'Steen,', 'Mulder,', 'Pelsmaeker,', 'Walle,', 'der', 'Billiard,', 'Don', 'Cruyssen,']
titanic['Title'] = titanic['Title'].replace(rareName, 'Rare', regex=False)


# In[101]:


titanic.Title.value_counts()


# ## 📌정규표현식
# - 문자의 패턴을 인식하고 검색하여 필요한 정보를 쉽게 식별하거나 추출하는데 사용
# - 문자 그대로를 입력하여 찾을 수 있지만 원하지 않는 문자도 포함되는 경우가 있음 -> 정규표현식 사용하는 것이 좋음
# ---
# 정규 표현식 연습 사이트 :  https://regexr.com/ 
# 
# 
# #### [정규표현식 수행할 수 있는 경우]
# 1. 문자열에서 특정 단어를 검색하는 경우
# 2. 문자열에서 특정 패턴에 부합하는 단어를 검색하는 경우
# 3. 문자열에서 특정 단어와 기호를 변경하거나 교체하는 경우
# ...

# ### 정규표현식 필수 문법

# In[103]:


Image('./정규표현식.png')


# ### 판다스 정규 표현식 기초 활용

# In[105]:


#Name에서 호칭만 분리하기
# ([A-Za-z]+)\.

titanic['Title2'] = titanic.Name.str.extract(r' ([A-Za-z]+)\.' , expand=False)
titanic.head()


# In[106]:


#Name이 'Z'로 시작하는 승객 데이터 추출하기

titanic[titanic.Name.str.count(r'(^Z.*)') == 1]


# In[107]:


titanic.Name.str.count(r'(^Z.*)').sum()


# In[108]:


#Name이 'Y'로 시작하는 승객 데이터 추출하기

titanic[titanic.Name.str.match(r'^Y.*') == True]


# ## 📌문자 수 세기

# In[109]:


#타이타닉 Name 컬럼에서 공백을 포함한 모든 문자 수 세기

titanic['Name'].str.count('')


# In[110]:


#타이타닉 Name 컬럼에서 단어 수 세기

titanic['Name'].str.count(' ') + 1


# In[111]:


#타이타닉 Name 컬럼에서 특정 문자 수 세기

titanic['Name'].str.count('a')


# # 3.5 카테고리 데이터 처리
# - 타이타닉 데이터셋에서 'Fare', 'Age', 'SibSp' 컬럼을 제외한 모든 컬럼은 카테고리 타입에 해당됨
# - 카테고리 데이터 == 고윳값이 있는 데이터
# 
# #### [2가지의 카테고리 데이터]
# 1. 범주형(명목형) - 사람의 혈액형이나 주소, 직업, 취미 등의 값의 높고 낮음의 순위가 없는 범주형
# 2. 순서형 - 객실등급(Pclass)과 같이 크기나 순서 등의 고유 순위가 존재하는 순서형

# ## 📌숫자 타입 데이터를 카테고리 타입으로 만들기
# - 숫자 타입의 연속형이나 이산형 데이터는 그 자체로 활용하기도 하지만 범주형으로 변환해서 사용하기도 함
# - 일정 구간(구간형) 또는 비율(비율형)으로 변환하면 데이터 파악이 수월 

# ### 구간형으로 범주화하기

# In[112]:


print(titanic.Age.min())
print(titanic.Age.max())


# In[113]:


# 5개의 연령대로 나이 컬럼 범주화하기

bins = [0, 9, 18, 40, 60, 81] #나이 구분 포인트 지정하기
labels = ['어린이', '청소년', '2030대', '4050대', '60대 이상'] #bins 보다 1개 더 적게 정의


# In[114]:


titanic['Age_band_cut'] = pd.cut(titanic['Age'], bins=bins, labels=labels)
titanic.head()


# In[115]:


#생성한 범주 고유 개수 확인

titanic.Age_band_cut.value_counts()


# ### 비율형으로 범주화하기

# In[116]:


labels = ['어린이', '청소년', '2030대', '4050대', '60대 이상']
titanic['Age_band_qcut'] = pd.qcut(titanic.Age, q=5 ,labels=labels)
titanic.head()


# ▶ 모든 구간이 일정 비율로 나누어졌기 때문에 cut() 함수와 다른 결과가 나옴

# In[119]:


# 각 구간의 시작과 끝 확인하기

print(titanic[titanic.Age_band_qcut == '어린이'].min()['Age'])
print(titanic[titanic.Age_band_qcut == '어린이'].max()['Age'])


# ## 📌카테고리 데이터에 순서 만들기
# - 카테고리 데이터에 순서를 지정하는 것 == 고유한 범줏값에 각각 다른 가중치를 부여하는 것
# - 데이터 분석 시 가중치가 높은 범줏값에 더 높은 점수 또는 더 높은 순위를 부여한 데이터를 분석 모델링에 사용하면 결과에 더 큰 영향을 줄 수 있음
# - 변수 특성에 따라 순서가 적합한 경우도 있음(아닌 경우도 있음)

# In[120]:


pd.Categorical([1, 2, 3, 1, 2, 3, 'a', 'b', 'c', 'a', 'b', 'c', np.nan])


# In[121]:


#카테고리 데이터에 codes 활용하여 각 데이터의 범주 코드를 확인

pd.Categorical([1, 2, 3, 1, 2, 3, 'a', 'b', 'c', 'a', 'b', 'c', np.nan]).codes


# In[122]:


#카테고리 데이터에 순서 부여
#categories 작은 순으로 입력

pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'], ordered=True, categories=['c', 'b', 'a'])


# In[124]:


#타이타닉 데이터의 객실등급(Pclass) 컬럼을 순서형으로 변경하기
#승객의 생존에 객실 등급의 높고 낮음이 영향을 끼칠 수 있다고 판단

titanic['Pclass'] = titanic.Pclass.astype('category')
titanic['Pclass'] = titanic.Pclass.cat.set_categories([3, 2, 1], ordered=True)
titanic.Pclass.sort_values() #작은 데이터 순으로 정렬


# In[ ]:





# In[ ]:




