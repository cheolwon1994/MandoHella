
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv
import json
import csv

#json형태의 file을 data2에 저장한다
with open('Cw_test1 2019-02-15 16_12_45.sdcl', encoding="utf-8") as data_file:    
    data2 = json.load(data_file)

model = tf.global_variables_initializer()

#csv파일을 data에 저장한다
data = read_csv('Cw_test1 2019-02-15 16_12_45.csv',sep=',')

#상태를 확인하기 위한 코드로, file에는 영향끼치지 않음
for d in data2['DetectedSegments']:
    a= d['SegmentWindow']['StartSample']
    b = d['SegmentWindow']['EndSample']
    print("<" + str(a)+", "+ str(b)+">")

    for d5 in d['LabelData']:
        c = d5['value']
        e = d5['name']
        if 'Label' == str(e) :
            print(c)

    print()

#file을 만들어서, label된 data만 저장
f = open('ForTrainInput6_1.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f) 
csv_writer.writerow(['','AccelerometerX','AccelerometerY','AccelerometerZ','GyroscopeX','GyroscopeY','GyroscopeZ','Label','SegmentID','Subject'])
j=0
k=2
for d in data2['DetectedSegments']:
    a= d['SegmentWindow']['StartSample']
    b = d['SegmentWindow']['EndSample']

    for d5 in d['LabelData']:
        c = d5['value']
        e = d5['name']
        if 'Label' == str(e) :
            g = c
    for i in range(0,14000) :
        if data.sequence[i]>b :
            k+=1
            break
        elif a<=data.sequence[i] :
            csv_writer.writerow([j,data.AccelerometerX[i],data.AccelerometerY[i],data.AccelerometerZ[i],data.GyroscopeX[i],data.GyroscopeY[i],data.GyroscopeZ[i],g,k,'CW'])
            j+=1
f.close()

