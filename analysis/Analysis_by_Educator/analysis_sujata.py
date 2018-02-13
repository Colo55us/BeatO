# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:00:34 2018

@author: BeatO
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('D:/Data_For Sugar_Educators-HOD Dashboard_2018_01_01.csv',sep=',')
#print(df['reading'].count())

df = df.loc[df.duplicated(subset='reading',keep=False),:]
#df = df.iloc[0:200]


#print(df.sort_values(['name'],ascending=1))
#print(df[df['name']=='NaN'])

df = df[df.readingtime.str.contains("Fasting")== True]
df = df[df.eduname.str.contains("Sujata")== True]
df = df.drop(['email','phone','deviceid','appversion','os','readingdevice','firstentrydate','lastentrydate'],axis=1)

li_userid = list(set(list(df['userid'])))

print(df.head(40))

li_m = []
li_c = []




print(len(li_userid))
counter = 0
for uid in li_userid:
    li_helper = []
    
    for index,row in df.iterrows():
        if row['userid']==uid:
            if row['reading']>30 and row['reading']<300:
                li_helper.append(row['reading'])
                
                
                df1 = df.drop([index], inplace=True)

                
    if len(li_helper)>3:
        l = np.array(np.arange(0,len(li_helper)))
        z = np.polyfit(l,np.array(li_helper),1)
        x = np.array(z).tolist()
        if x[0]<=1E308:
            li_m.append(x[0])
            li_c.append(x[1])
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
plt.plot(X,Y)
fig.suptitle('sujata',fontsize=25)
plt.xlabel('frequency',fontsize=18)
plt.ylabel('Glucose Level',fontsize=18)
plt.show()
fig.savefig('graph/sujata.jpg')




    




    


    
    
    
        
    
    