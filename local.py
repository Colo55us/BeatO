import pandas as pd
import numpy as np
from sys import argv

'''
df = pd.read_excel('book1.xlsx')
df = pd.DataFrame(list(set(list(df['userid']))))

df.to_csv('users.csv',sep=',')

'''

df = pd.read_csv('final.csv',sep=',')
df_query = df
#df.dropna(0,inplace=True)

#df.to_csv('user_info1.csv',sep=',')
'''
for index,row in df.iterrows():
    if 0 in df['yob']:
        df = df.drop(index)

df.to_csv('user_info1.csv',sep=',')
df = pd.read_csv('user_info1.csv',sep=',')
for index,row in df.iterrows():
    if row['yob']>2000:
        df = df.drop(index)

df.to_csv('user_info1.csv')


for index,row in df.iterrows():
    if row['bmr']<=0:
        df = df.drop(index)

df.to_csv('user_info1.csv',sep=',')
'''
'''
df_reading = pd.read_csv('raw_data/User_sugar.csv',sep=',')
df_reading = df_reading[df_reading.readingtime.str.contains("Fasting")==True]
li_user = list(df['userid'])
df_reading.to_csv('fasting_reading.csv',sep=',')
df_reading = pd.read_csv('fasting_reading.csv')
df_slope = pd.DataFrame(columns = ['userid','slope','intercept','read_count','flag'])
print(df_reading.head())

for user in li_user:
    li = []
    if user in df_reading['userid'].values:
        li = df_reading.index[df_reading.userid==user].tolist()
        #print(li)
    
        if len(li)>=3:
        #print(len(li),'   ----    ',user)
            li_reading = []
            for index in li:
                li_reading.append(df_reading.iloc[index].reading)
                
            l = np.array(np.arange(0,len(li_reading)))
            z = np.polyfit(l,np.array(li_reading),1)
            x = np.array(z).tolist()

            if x[0]<1E308:
                flag = 0
                if x[0]<0:
                    flag = 1

                df_slope = df_slope.append({'userid':user,'slope':x[0],'intercept':x[1],'read_count':len(li),'flag':flag},ignore_index=True)


df_slope.to_csv('user_slope.csv',sep=',')
'''

from sklearn import preprocessing,svm,neighbors
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV,SelectKBest,f_classif
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier

df = df.drop(['userid','slope'],1)
#print(df.head())
def increase_dimensionality(df):
    combination = list(combinations(df.columns.tolist(), 2))
    col_names = list(df.columns)+['_'.join(x) for x in combination]

    poly = preprocessing.PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = col_names

    col_noint = [i for i,x in enumerate(list((df==0).all())) if x]
    df = df.drop(df.columns[col_noint],axis=1)

    return df

X = (df.drop(['flag'],1))
X = np.array(increase_dimensionality(X))
print(X.shape)
#print(X)
y = np.array(df['flag'])
X = preprocessing.scale(X)
kbest = SelectKBest(f_classif,k=20)
#X = kbest.fit_transform(X,y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X.shape)
clf = svm.SVC(C = 0.01,kernel='linear')

#clf = RFECV(RandomForestClassifier(n_estimators=15,max_features=None,n_jobs=-1),scoring='accuracy')
clf.fit(X_train,y_train)

score = clf.score(X_test,y_test)
print(score)
'''
print('Enter BMI,  BMR,  Gender(Male-0/Female-2),  height(inch),  weight,  Year of Birth,   First Reading,  Reading Count')
ui = pd.DataFrame([input().split()],dtype=np.int64)
poly = preprocessing.PolynomialFeatures(interaction_only=True, include_bias=False)
ui = np.array(poly.fit_transform(ui))
#ui = increase_dimensionality(ui)
#ui = kbest.transform(ui)
print(ui.shape)
result = clf.predict(ui)
print("the output is :",result,'with accuracy:',score)

'''

import pickle
pickle.dump(clf,open('reading_picle.sav','wb'))
'''
s, userid = argv
user = np.int64(userid)

i = df_query.index[df_query.userid==user]

d = pd.DataFrame(df_query.iloc[i])
d.drop(['userid','slope','flag'],1,inplace=True)

poly = preprocessing.PolynomialFeatures(interaction_only=True, include_bias=False)
ui = np.array(poly.fit_transform(d))

result = clf.predict(ui)
print("the output is :", result,'with accuracy',score)
'''