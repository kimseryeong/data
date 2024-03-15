#!/usr/bin/env python
# coding: utf-8

# ## 기본 개념
# 
# ###### csv.reader() : CSV 파일에서 데이터 읽어오는 함수
# 
# ###### csv.writer() : CSV 파일에 데이터를 저장하는 함수
# 

# In[1]:


import csv

#weather.csv 파일을 읽어보되 windows  한글 인코딩 방식(cp949)으로 읽어오라는 의미
f = open('weather.csv', 'r', encoding='cp949') 

#읽어온 csv 파일을 콤마(,)를 기준으로 분리해서 저장하라는 의미
data = csv.reader(f, delimiter=',')


#print(data)

#data 한줄씩 출력
for row in data:
    print(row)
    
f.close()


# ## 헤더 생성

# In[3]:


f = open('weather.csv', 'r', encoding='cp949') 

data = csv.reader(f)
header = next(data) #next() : data의 첫번째 행 읽어오기
print(header)


# In[4]:


f = open('weather.csv', 'r', encoding='cp949') 
data = csv.reader(f)

header = next(data)

for row in data:
    print(row)
    
f.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




