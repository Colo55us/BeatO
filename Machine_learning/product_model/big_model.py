import numpy as np 
import pandas as pd 

#df = pd.read_csv('merged.csv',sep=',')

#print(len(user))
df_merged = pd.read_csv('merged.csv')
df_sugar = pd.read_csv('user_sugar.csv',sep=',')
df_user = pd.read_csv('C:/Users/BeatO/Documents/garbage_merged_sugar.csv',sep=',')
'''
li_user = list(set(list(df_user['userid'])))

df_vinegar = df_merged[df_merged.service.str.contains('Vinegar')==True]
df_socks = df_merged[df_merged.service.str.contains('Socks')==True]
df_atta = df_merged[df_merged.service.str.contains('Atta')==True]
df_juice = df_merged[df_merged.service.str.contains('Juice')==True]
df_oats = df_merged[df_merged.service.str.contains('Oats')==True]
df_strips = df_merged[df_merged.service.str.contains('Strips')==True]
df_strips = df_strips[df_strips.service.str.contains('with')==False]
df_strips_25 = df_strips[df_strips.service.str.contains('25')==True]
df_strips_50 = df_strips[df_strips.service.str.contains('50')==True]
df_strips_100 = df_strips[df_strips.service.str.contains('100')==True]

print(len(li_user))
df_result = pd.DataFrame(columns = ['userid','vinegar','socks','atta','juice','oats','strips_25','strips_50','strips_100'])
for user in li_user:
    vinegar,socks,atta,juice,oats,strips_25,strips_50,strips_100 = 0,0,0,0,0,0,0,0
    if user in df_vinegar['userid'].values:
        li = df_vinegar.index[df_vinegar.userid==user].tolist()
        vinegar = len(li)

    if user in df_socks['userid'].values:
        li = df_socks.index[df_socks.userid==user].tolist()
        socks = len(li)
   
    if user in df_atta['userid'].values:
        li = df_atta.index[df_atta.userid==user].tolist()
        atta = len(li)

    if user in df_juice['userid'].values:
        li = df_juice.index[df_juice.userid==user].tolist()
        juice = len(li)

    if user in df_oats['userid'].values:
        li = df_oats.index[df_oats.userid==user].tolist()
        oats = len(li)

    if user in df_strips_25['userid'].values:
        li = df_strips_25.index[df_strips_25.userid==user].tolist()
        strips_25 = len(li)
    
    if user in df_strips_50['userid'].values:
        li = df_strips_50.index[df_strips_50.userid==user].tolist()
        strips_50 = len(li)
 
    if user in df_strips_100['userid'].values:
        li = df_strips_100.index[df_strips_100.userid==user].tolist()
        strips_100 = len(li)
    
    if vinegar == 0 & socks == 0 & atta == 0 & juice == 0 & oats == 0 & strips_25 == 0 & strips_50 == 0 & strips_100 == 0:
        pass
    else:
        df_result = df_result.append({'userid':user,'vinegar':vinegar,'socks':socks,'atta':atta,'juice':juice,'oats':oats,'strips_25':strips_25,'strips_50':strips_50,'strips_100':strips_100},ignore_index=True)
df_result.to_csv('finaldata.csv',sep=',')



df_result = pd.read_csv('finaldata.csv')

li_user = (df_result['userid'])
df_count = pd.DataFrame(columns = ['userid','count'])
for user in li_user:
    li = df_sugar.index[df_sugar.userid==user].tolist()
    df_count= df_count.append({'userid':user,'count':len(li)},ignore_index=True)


df_count.to_csv('usercount.csv',sep=',')
#/////////////////////////////////////////////////////////////////////////////////////////////////////
'''
'''

df_result = pd.read_csv('result_no_1.csv')

#print(df_result.head())

li_user = (df_result['userid'])

df_slope = pd.DataFrame(columns = ['userid','improvement'])
for user in li_user:
    li = df_sugar.index[df_sugar.userid==user].tolist()
    li_reading = []
    for index in li:
        li_reading.append(df_sugar.iloc[index].reading)

    l = np.array(np.arange(0,len(li_reading)))
    z = np.polyfit(l,np.array(li_reading),1)
    x = np.array(z).tolist()
    if x[0]<1E308:
        flag = 0
        if x[0] < 0:
            flag =1

        df_slope = df_slope.append({'userid':user,'improvement':flag},ignore_index=True)

df_slope.to_csv('userid_impro.csv',sep=',')
'''

#//////////////////////////////////////////////////////////////////////////////////////////////////////////
'''
df_result = pd.read_csv('finaldata.csv')

li_user = (df_result['userid'])

for index,row in df_result.iterrows():
    if row['count'] == 1:
        df_result = df_result.drop(index)



df_result.to_csv('result_no_1.csv',sep=',')
'''
#//////////////////////////////////////////////////////////////////////////////////////////////////

from sklearn import preprocessing,svm,neighbors
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV,SelectKBest,f_classif,chi2,mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
df = pd.read_csv('result_no_2.csv')
print(df.head(10))
df.drop(['userid'],1,inplace=True)
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



#df = pd.to_numeric(df)
#print(df.head())
X = (df.drop(['improvement'],1).astype(float))
y = np.array(df['improvement'])
X = increase_dimensionality(X)
print(X.shape)
X = np.array(preprocessing.scale(pd.DataFrame(X)))
kbest = SelectKBest(f_classif,k=7)
X = kbest.fit_transform(X,y)

print(X.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

#clf = svm.SVC(C = 0.01,kernel='linear')
#clf = LogisticRegression(C=0.001)
clf = RFECV(RandomForestClassifier(n_estimators=15,max_features=None,n_jobs=-1),scoring='accuracy')
#clf = neighbors.KNeighborsClassifier(n_neighbors = 2)
clf.fit(X_train,y_train)


score = (clf.score(X_test,y_test))
print(score)
print('''

        Please enter following details in order:-
        1)Enter vinegar count - 
        2)Enter socks count - 
        3)Enter atta count-
        4)Enter Juice count-
        5)Enter Oats count-
        9)Enter Reading Count-

    ''')

input1 = pd.DataFrame([input().split()],dtype = np.int64)

poly = preprocessing.PolynomialFeatures(interaction_only=True, include_bias=False)
input1 = poly.fit_transform(input1)
#input1 = increase_dimensionality(input1)
input1 = kbest.transform(input1)
result = clf.predict(input1)
if score <50.0:
    score = 100 - (score * 100)
    result = 1 - result

print("the ooutput is :",result,'with accuracy:',score)



