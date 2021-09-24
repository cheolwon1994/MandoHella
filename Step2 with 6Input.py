
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv
import json
import csv
import pandas as pd
from operator import eq

model = tf.global_variables_initializer()
data = pd.read_csv('ForTest5.csv',sep=',')

#다음에 해당되는 data의 열만 가져온다
df2 = data[["AccelerometerX", "AccelerometerY", "AccelerometerZ","GyroscopeX", "GyroscopeY", "GyroscopeZ","Label","SegmentID"]]
df3 = data[["AccelerometerX", "AccelerometerY", "AccelerometerZ","GyroscopeX", "GyroscopeY", "GyroscopeZ"]]

#각 data의 값을 불러온다
xy2 = np.array(df2[0:]["AccelerometerX"],dtype=np.float32)
xy3 = np.array(df2[0:]["AccelerometerY"],dtype=np.float32)
xy4 = np.array(df2[0:]["AccelerometerZ"],dtype=np.float32)
xy5 = np.array(df2[0:]["GyroscopeX"],dtype=np.float32)
xy6 = np.array(df2[0:]["GyroscopeY"],dtype=np.float32)
xy7 = np.array(df2[0:]["GyroscopeZ"],dtype=np.float32)

j=0
k=0

#파일을 만들어서, 첫 행에 다음과 같은 형태를 기입한다.
f = open('TestInpput6_5.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['','GyroscopeX25','GyroscopeY25','GyroscopeZ25','GyroscopeZ50','GyroscopeX75','GyroscopeZ75','Label1','Label2','Label3','Label4'])

#eq(dd)부분은 label의 값을 1000,0100,..형태로 바꾸기 위한 작업
#dd == 2 : <<이 부분은 data 100개 단위마다 'GyroscopeX25','GyroscopeY25','GyroscopeZ25','GyroscopeZ50','GyroscopeX75','GyroscopeZ75' 값을 기입
for d in df2["SegmentID"] :
    dd =(df2["Label"][j].split('.')[0])
    if eq(dd,"1"):
        dd1=1
        dd2=0
        dd3=0
        dd4=0
    elif eq(dd,"2") :
        dd1=0
        dd2=1
        dd3=0
        dd4=0
    elif eq(dd,"3") :
        dd1=0
        dd2=0
        dd3=1
        dd4=0
    elif eq(dd,"4") :
        dd1=0
        dd2=0
        dd3=0
        dd4=1
        
    if d == 2 :       
        j+=1
        if j%100 ==0 :
            csv_writer.writerow([k,np.percentile(xy5[j-100:j],25),np.percentile(xy6[j-100:j],25),np.percentile(xy7[j-100:j],25), np.mean(xy7[j-100:j]),np.percentile(xy5[j-100:j],75),np.percentile(xy7[j-100:j],75),dd1,dd2,dd3,dd4])
            k+=1
    elif d==3 :
        if df2["SegmentID"][j-1]!=d :
            start = j
        j+=1
        if (j-start)%100 ==0 :
            csv_writer.writerow([k,np.percentile(xy5[j-100:j],25),np.percentile(xy6[j-100:j],25),np.percentile(xy7[j-100:j],25), np.mean(xy7[j-100:j]),np.percentile(xy5[j-100:j],75),np.percentile(xy7[j-100:j],75),dd1,dd2,dd3,dd4])
            k+=1
            
    elif d==4 :
        if df2["SegmentID"][j-1]!=d :
            start = j
        j+=1
        if (j-start)%100 ==0 :
            csv_writer.writerow([k,np.percentile(xy5[j-100:j],25),np.percentile(xy6[j-100:j],25),np.percentile(xy7[j-100:j],25), np.mean(xy7[j-100:j]),np.percentile(xy5[j-100:j],75),np.percentile(xy7[j-100:j],75),dd1,dd2,dd3,dd4])
            k+=1
            
    elif d==5 :
        if df2["SegmentID"][j-1]!=d :
            start = j
        j+=1
        if (j-start)%100 ==0 :
            csv_writer.writerow([k,np.percentile(xy5[j-100:j],25),np.percentile(xy6[j-100:j],25),np.percentile(xy7[j-100:j],25), np.mean(xy7[j-100:j]),np.percentile(xy5[j-100:j],75),np.percentile(xy7[j-100:j],75),dd1,dd2,dd3,dd4])
            k+=1
            
    elif d==6 :
        if df2["SegmentID"][j-1]!=d :
            start = j
        j+=1
        if (j-start)%100 ==0 :
            csv_writer.writerow([k,np.percentile(xy5[j-100:j],25),np.percentile(xy6[j-100:j],25),np.percentile(xy7[j-100:j],25), np.mean(xy7[j-100:j]),np.percentile(xy5[j-100:j],75),np.percentile(xy7[j-100:j],75),dd1,dd2,dd3,dd4])
            k+=1
            
            
    elif d==7 :
        if df2["SegmentID"][j-1]!=d :
            start = j
        j+=1
        if (j-start)%100 ==0 :
            csv_writer.writerow([k,np.percentile(xy5[j-100:j],25),np.percentile(xy6[j-100:j],25),np.percentile(xy7[j-100:j],25), np.mean(xy7[j-100:j]),np.percentile(xy5[j-100:j],75),np.percentile(xy7[j-100:j],75),dd1,dd2,dd3,dd4])
            k+=1
            
            
    elif d==8 :
        if df2["SegmentID"][j-1]!=d :
            start = j
        j+=1
        if (j-start)%100 ==0 :
            csv_writer.writerow([k,np.percentile(xy5[j-100:j],25),np.percentile(xy6[j-100:j],25),np.percentile(xy7[j-100:j],25), np.mean(xy7[j-100:j]),np.percentile(xy5[j-100:j],75),np.percentile(xy7[j-100:j],75),dd1,dd2,dd3,dd4])
            k+=1
            
    elif d==9 :
        if df2["SegmentID"][j-1]!=d :
            start = j
        j+=1
        if (j-start)%100 ==0 :
            csv_writer.writerow([k,np.percentile(xy5[j-100:j],25),np.percentile(xy6[j-100:j],25),np.percentile(xy7[j-100:j],25), np.mean(xy7[j-100:j]),np.percentile(xy5[j-100:j],75),np.percentile(xy7[j-100:j],75),dd1,dd2,dd3,dd4])
            k+=1
            
f.close()

