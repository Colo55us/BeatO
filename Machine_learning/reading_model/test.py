# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:00:32 2018

@author: BeatO
"""
'''
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

df = pd.read_csv('D:/reversing_slope/slope_userid.csv',sep=',')

df1 = df.iloc[:500]
first = np.mean(np.array(list(df1['slope'])))

df2 = df.iloc[530:]
last = np.mean(np.array(list(df2['slope'])))

print('first:',first,'    last :',last)

x = np.array(list([i for in range(0,len(list(df1['slope'])))]))
y = np.array(df['slope'])


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:/Data_For Sugar_Educators-HOD Dashboard_2018_01_01.csv',sep=',')
df = df.drop(['name','email','phone','deviceid','appversion','os','readingdevice'],axis=1)
#print(df[df.userid.contains(57783)==True])
df = df[df.readingtime.str.contains('Fasting')==True]
#print(df[['userid','reading','entrydate']].head(50))

print(df.loc[df['userid']==57409].reading,df.loc[df['userid']==57409].entrydate)#,df.loc[df['userid']==49416].entrydate,sep='\n')
df = df[df.readingtime.str.contains("Fasting")== True]
df = df[df.eduname.str.contains("Anshika Gupta")==True]
#print(df)
df = df.drop(['email','phone','deviceid','appversion','os','readingdevice','firstentrydate','lastentrydate'],axis=1)
#df = df.iloc[0:200]

df = df.loc[df.duplicated(subset='reading',keep=False),:]
df = df[df.readingtime.str.contains("Fasting")== True]
li_userid = list(set(list(df['userid'])))
df = df.sort_values(['userid'],ascending=[True])
print(df.head(40))

li_m = []
li_c = []

li_poly = []

df_length = int((df.shape[0]))

margin_id=int(df.iloc[int(df_length/2),0])



df1 = pd.DataFrame(df.iloc[0:int((df_length)/2)])
df2 = pd.DataFrame(df.iloc[int((df_length/2)):df_length])

counter = 0
for uid in li_userid:
    li_helper = []
    if uid<margin_id:
        for index,row in df1.iterrows():
            if row['userid']==uid:
                li_helper.append(row['reading'])
                #df1 = df1.drop([index], inplace=True)
    else:
        for index,row in df2.iterrows():
            if row['userid']==uid:
                li_helper.append(row['reading'])
                #df2 = df2.drop([index],inplace=True)
    l = np.array(np.arange(0,len(li_helper)))
    z = np.polyfit(l,np.array(li_helper),1)
    x = np.array(z).tolist()
    if x[0]<=1E308:
        li_m.append(x[0])
        li_c.append(x[1])
        df_result = df_result.append({'userid':uid,'slope':slope})


        print('slope: ',x[0],'     Y-intercept:',x[1], counter,' out of ',len(li_userid),'user_id:',uid)
    else:
        pass
    counter = counter+1

m_final = np.mean(np.array(li_m))
c_final = np.mean(np.array(li_c))

Y = []
X = []
for i in range(1,11):
    y_loc = m_final*i + c_final
    X.append(i)
    Y.append(y_loc)
print('Mean slope=',m_final,'    Mean Y-intercept=',c_final)

fig = plt.figure()
df_result.to_csv('slope_final.csv')
plt.plot(X,Y)
fig.suptitle('Anshika',fontsize=25)
plt.xlabel('frequency',fontsize=18)
plt.ylabel('Glucose Level',fontsize=18)
plt.show()
fig.savefig('graph/Anshika.jpg')


'''
from sys import argv
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
df_query = pd.read_csv('D:/Machine_learning/ml_reading/final.csv',sep=',')

script, userid = argv

model = pickle.load(open('D:/Machine_learning/ml_reading/reading_picle.sav','rb'))

user = np.int64(userid)

i = df_query.index[df_query.userid==user]

d = pd.DataFrame(df_query.iloc[i])
d.drop(['userid','slope','flag'],1,inplace=True)

poly = preprocessing.PolynomialFeatures(interaction_only=True, include_bias=False)
ui = np.array(poly.fit_transform(d))

result = model.predict(ui)
print("the output is :", result)


