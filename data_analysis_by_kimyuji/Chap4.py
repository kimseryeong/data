#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image


# # 4.1 데이터 병합
# - 데이터는 여러 파일에 분산, 데이터베이스에 보관/기록 등 다양한 형태로 존재함
# - 흩어져 있는 데이터를 병합하는 작업이 중요

# ## 📌 데이터 위 아래로 연결하기
# 1. concat()
# 2. append()

# ### concat() : 공통 컬럼 없이도 가능

# In[3]:


# Series 병합

menu1 = pd.Series(['파스타', '라멘', '냉면'], index=[1, 2, 3])
menu2 = pd.Series(['돈가스', '피자', '치킨'], index=[4, 5, 6])
pd.concat([menu1, menu2])


# In[2]:


# Dataframe 병합

data1 = pd.DataFrame({'음식명' : ['돈가스', '치킨', '피자', '초밥', '라멘', '냉면'],
                                         '카테고리' : ['일식', '한식', '양식', '일식', '일식', '한식']
                     })
data1


# In[3]:


data2 = pd.DataFrame({'음식명' : ['갈비탕', '완탕면', '마라탕', '파스타', '쫄면', '우동'],
                                         '카테고리' : ['한식', '중식', '중식', '양식', '한식', '한식']
                     })
data2


# In[7]:


pd.concat([data1, data2], ignore_index=True) #ignore_index : 기존 인덱스 무시하고 새롭게 정의


# In[8]:


pd.concat([data1, data2], keys=['data1', 'data2']) #keys : 데이터 출처 삽입


# In[4]:


data3 =  pd.DataFrame({'음식명' : ['갈비탕', '완탕면', '마라탕', '파스타', '쫄면', '우동'],
                                         '판매인기지역' : ['전주', '서울', '서울', '부산', '제주', '인천']
                     })
data3


# In[10]:


# 서로 다른 컬럼이 존재하는 데이터 합치는 경우
# 합집합인 outer join 으로 적용되어 컬럼에 존재하지 않는 값은 NaN으로 출력됨

pd.concat([data1, data3], ignore_index=True)


# In[11]:


# 서로 다른 컬럼이 존재하는 데이터 합치는 경우
# 교집합인 부분만 얻고 싶다면, join='inner' 변수 적용

pd.concat([data1, data3], ignore_index=True, join='inner')


# ### append() : concat 만큼 자주 사용되지는 않음 (매개변수가 적은편)

# In[12]:


data1.append(data2, ignore_index=True)


# In[13]:


# 서로 다른 컬럼이 존재하는 데이터 합치는 경우
# 합집합인 outer join 으로 적용되어 컬럼에 존재하지 않는 값은 NaN으로 출력됨

data1.append(data3, ignore_index=True)


# ## 📌 컬럼을 기준으로 데이터 병합 (좌우연결)
# 1. concat()
# 2. metge()
# 3. join()

# ### 데이터 좌우 연결하기

# ### concat()

# In[14]:


#concat() 의 axis=1 지정 시 컬럼을 기준으로 데이터 연결됨
#데이터에 공통 컬럼이 존재하지 않아도 옆으로 병합 가능

pd.concat([data1, data2], axis=1)


# ### 특정 컬럼 기준으로 데이터 병합

# In[5]:


print(data1)
print(data2)
print(data3)
data4 = pd.concat([data1, data2], ignore_index=True)
print(data4)


# In[23]:


data3


# In[20]:


data4


# ### merge()

# In[22]:


# merge() : 일반적으로 데이터 병합 시 자주 사용되는 함수
# key : 기준이 되는 컬럼과 행 데이터
# inner join연산

pd.merge(data3, data4)


# ▶ 공통 컬럼인 '음식명'을 기분으로 데이터를 합치고 그 컬럼이 key가 됨

# In[6]:


data5 = data4.merge(data3)
data5


# In[8]:


data6 = data4.merge(data3, how='outer')
data6


# In[10]:


#data6의 키 값과 동일한 값을 가진 또 다른 데이터 생성
data7 = pd.DataFrame({'판매인기지역': ['서울', '부산', '전주', '인천'],
                                         '오픈시간' : ['10시', '11시', '12시', '13시']
                     })
data7


# In[11]:


pd.merge(data6, data7)


# ### join()

# In[12]:


#join() : 겹치는 컬럼이 있다면 suffix를 지정해주어야 함

data6.join(data7, lsuffix='_left_key', rsuffix='_right_key')


# In[15]:


#기준이 되는 컬럼을 on 변수에 정의

data6.join(data7.set_index('판매인기지역'), on='판매인기지역')


# ## 📌key 컬럼에 같은 데이터 값이 여러 개 있는 경우 데이터 병합하기

# In[16]:


menu_price = pd.DataFrame({
    '음식명' : ['돈가스', '돈가스', '파스타', '파스타', '파스타']
    , '가격' : ['11000', '15000', '19000', '18000', '16000']
})
menu_price


# In[17]:


menu_location = pd.DataFrame({
    '음식명' : ['돈가스', '파스타', '파스타', '피자', '피자']
    , '매장위치' : ['광화문', '을지로', '명동', '여의도', '가로수길']
})
menu_location


# In[18]:


#키 값이 중복되는 데이터가 있는 경우 가능한 경우의 수를 조합하여 병합

pd.merge(menu_price, menu_location)


# ## 📌동일한 컬럼명이 존재하지만, key 컬럼이 되면 안 되는 경우

# In[19]:


menu1 = pd.DataFrame({
    '음식명': ['초밥', '짜장면']
    ,'판매날짜' : ['2024-04-01', '2024-03-12']
    ,'메모' : ['20000', '15000']
})
menu1


# In[20]:


menu2 = pd.DataFrame({
    '음식명': ['초밥', '짜장면']
    ,'메모' : ['일식', '중식']
})
menu2


# In[21]:


pd.merge(menu1, menu2)


# ▶ 동일한 이름의 컬럼 중 어느 것을 key로 병합할지 정해주지 않았으므로 어떠한 데이터도 출력되지 않음

# In[22]:


pd.merge(menu1, menu2, on='음식명')


# ## 📌서로 다른 key 컬럼을 가진 데이터 병합
# - left_on / right_on 을 사용하여 key 명시해주기

# In[23]:


menu_price = pd.DataFrame({
    '음식명': ['초밥', '짜장면', '초밥']
    ,'가격' : ['20000', '8000', '16000']
})
menu_price


# In[24]:


menu_score = pd.DataFrame({
    '메뉴': ['초밥', '짜장면', '짜장면']
    ,'단가' : ['20000', '8000', '7000']
})
menu_score


# In[26]:


pd.merge(menu_price, menu_score, left_on = '음식명', right_on='메뉴')


# ▶ '음식명'과 '메뉴'의 컬럼의 값은 동일하므로 drop()을 통해 삭제가능

# In[27]:


pd.merge(menu_price, menu_score, left_on = '음식명', right_on='메뉴').drop('메뉴', axis=1)


# ## 📌인덱스를 기준으로 데이터 병합

# In[29]:


#음식 이름이 인덱스인 데이터 2개 생성

menuD1 = pd.DataFrame(
    [20000, 15000, 12000, 13000, 15000]
    ,index=['초밥', '초밥', '갈비탕', '갈비탕', '갈비탕']
    ,columns=['가격']
)

menuD1


# In[30]:


menuD2 = pd.DataFrame(
     [12000, 7000, 8000, 9000, 25000]
    ,index=['갈비탕', '짜장면', '짜장면', '짜장면', '탕수육']
    ,columns=['가격']
)

menuD2


# In[31]:


#위 2개의 데이터프레임을 그대로 병합하면 기존의 인덱스는 사라짐
#기존 인덱스를 살리기 위해서는 left_index & right_index를 True로 지정

pd.merge(menuD1, menuD2, how='outer', left_index=True, right_index=True)


# ## 📌인덱스가 겹치는 데이터 병합 시 결측값 처리

# In[37]:


data1 = pd.DataFrame({
    '음식명' : ['돈가스', np.nan, '초밥', '치킨', np.nan]
    ,'카테고리' : ['일식', '양식', np.nan, '양식', '중식']
    ,'판매인기지역' : [np.nan, '부산', '제주', '서울', '제주']
})
data1


# In[38]:


data2 = pd.DataFrame({
    '음식명' : [np.nan, '냉면', '초밥', '치킨', '탕수육']
    ,'카테고리' : ['일식', np.nan, '양식', '양식', np.nan]
})
data2


# In[39]:


data1.combine_first(data2)


# ▶ data2에 존재하지 않는 컬럼의 결측값은 그대로 유지됨

# # 4.2 데이터 재형성과 피벗

# ## 📌컬럼을 인덱스로 교환

# In[40]:


#싱글인덱스의 경우

coffee_size_data = pd.DataFrame(
    [[10, 28], [8, 22]]
    ,index=['스타벅스', '커피빈']
    ,columns=['테이블수', '매장규모(평)']
)

coffee_size_data


# In[41]:


#데이터의 컬럼을 인덱스로 교환

coffee_size_data.stack()


# In[43]:


#인덱스 교환 원 상태로 되돌리기

coffee_size_data.stack().unstack()


# ## 📌데이터 피벗하기
# - 데이터를 변형하여 다양한 관점에서 볼 수 있음
# 1. pivot()
# 2. pivot_table()

# In[44]:


#주식 데이터 로드

data_stock = pd.read_csv('./bumping-into-data-analysis-main/datasets/stock_data.csv')
data_stock.head()


# In[45]:


#기업명(symbol) 컬럼을 인덱스로 하여 날짜별 거래량 확인하기

data_stock.pivot(index='symbol', columns='date', values='volume')


# ▶ 2019년 3월 1일의 애플 거래량이 가장 높은 것을 알 수 있음!

# In[46]:


#날짜(date) 컬럼을 인덱스로 하여 기업별 종가 확인하기

data_stock.pivot(index='date', columns='symbol', values='close')


# In[47]:


pd.pivot_table(data_stock, values='close', index='date', columns='symbol')


# ▶ pivot() 함수와 pivot_table() 함수 모두 동일한 결과를 얻을 수 있음

# ## 📌복합 개체를 개별 개체로 분리하기
# - 데이터에 복합적인 값이 포함되어 있다면 분석이 힘들기 때문에 분리해주는 것이 좋음

# In[49]:


data_e = pd.DataFrame({
    '재료' : [['밀가루', '설탕', '계란'], '밀가루', [], ['버터', '생크림']]
    ,'시간' : 10
    ,'방식' : [['굽기', '볶기'], np.nan, [], ['볶기', '섞기']]
})

data_e


# In[50]:


#explode() : 기본적으로 1개의 컬럼을 기준으로 데이터 분리
#'재료' 컬럼 분리하기

data_e.explode('재료')


# In[51]:


#추가적으로 분리하고 싶다면 explode() 중복해서 사용

data_e.explode('재료').explode('방식')


# # 4.3 데이터 병합 후 처리하기

# ## 📌병합 후 중복 행 확인 및 처리

# In[52]:


d = pd.DataFrame({
    '패션아이템' : ['팬츠', '팬츠', '자켓', '자켓', '자켓']
    ,'스타일' : ['캐주얼', '캐주얼', '캐주얼', '비즈니스룩', '비즈니스룩']
    ,'선호도(5점)' : [4, 4, 3.5, 4.2, 2.7]
})

d


# In[53]:


d.duplicated(keep='first') #keep : 중복 행이 있을 때, 첫행/마지막행 중 어느것에 체크할 것인지 여부 (first, last, False)


# In[54]:


d.duplicated(keep='first').value_counts()


# In[58]:


#특정 컬럼 중복 데이터 확인하기

d.duplicated(subset=['스타일']).value_counts()


# In[59]:


#중복 행 중 1개만 남기고 삭제

d.drop_duplicates(subset=['패션아이템', '스타일'], keep='last')


# ## 📌2개 데이터 비교하여 다른 부분 파악
# - 동일한 컬럼으로 구성된 데이터 프레임을 병합할 때는 어느 값이 다른지 파악하는 것이 중요
# - 일정 기간이 지난 후 데이터 업데이트 또는 어떠한 상황으로 기존 값이 변경 등등의 상황이 있음

# ### 데이터 길이가 동일한 데이터 비교

# In[62]:


fashion = pd.DataFrame({
    '패션아이템' : ['팬츠', '팬츠', '자켓', '자켓', '팬츠']
    ,'선호도' : [1.0, 2.0, 3.0, np.nan, 5.0]
    ,'평점' : [1.0, 2.0, 3.0, 4.0, 5.0]
})

fashion


# In[63]:


#데이터프레임 복제 후 일부 값 변경

fashion2 = fashion.copy()
fashion2.loc[0, '패션아이템'] = '스커트'
fashion2.loc[2, '평점'] = 4.0
fashion2


# In[64]:


#비교 
#차이가 없는 경우 NaN 출력

fashion.compare(fashion2)


# In[65]:


#비교 
#차이가 없는 경우 원본 데이터를 보고 싶은 경우 -> keep_equal=True

fashion.compare(fashion2, keep_equal=True)


# In[66]:


#비교 
#결과를 세로축으로 변경

fashion.compare(fashion2, align_axis=0)


# In[67]:


#전체 데이터 사이즈 유지한 채 비교한다면
#keep_shape=True

fashion.compare(fashion2, keep_shape=True)


# ### 데이터 길이가 다른 데이터 비교

# In[68]:


d1 = pd.DataFrame({
    '패션아이템' : ['팬츠', '스커트', '자켓', '티셔츠', '블라우스', '베스트']
    ,'평점' : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
})
d1


# In[69]:


d2 = pd.DataFrame({
    '패션아이템' : ['팬츠', '스커트', '자켓', '티셔츠', '블라우스', '베스트', '패딩']
    ,'평점' : [1.0, 5.0, 3.0, 3.0, 5.0, 6.0, 1.5]
})
d2


# In[70]:


#길이가 서로 다르기 때문에 가장 긴 길이를 기준으로 

d1.eq(d2)


# In[72]:


d2[d1.eq(d2).all(axis=1) == False]


# In[ ]:





# In[ ]:




