import pandas as pd 
import numpy as np 
from sklearn import preprocessing,model_selection,svm,decomposition
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.feature_selection import RFECV,SelectKBest,f_classif
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import random
import pickle

'''
df = pd.read_csv('data/user_recall.csv',sep=',')

df = df.drop(['id','bonustype','bonustype','classifaction','comorbidity','anymedicalcondition','bgtrend','targetbgtrend','diabeticcomplications','whatexercse','dayexercise','duartionexcercise','anyconcernkey','anyconcernshoulder','anyconcernother','familiarfootcare','emailfootcare','knowallergen','listallergen','oil','oilconsumption','familymember','artificialsweet','alcoholoften','smokeoften','insulinemail','prescription','additionalqueries','feedbacks','followuprequired','followupdate','earlymorning','breakfast','lunch','evesnacks','dinner','bedtime','cereallike','cerealdaily','pulselike','greenlike','fruitlike','milklike','dessertlike','nonveglike','nonvegdaily','nutlike','junklike','kidney','eye','nerve','heart','coliver','colipid','coheart','cokidney'],1)
#df.dropna(inplace=True)

def converting_strings(df):
    
    columns = df.columns.values

    for column in columns:
        string_index_dict = {}

        def helper_converter(val):
            return string_index_dict[val]
        

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            
            column_list = df[column].values.tolist() 

            unique_val_list = set(column_list)


            allocator = 0

            for unique_val in unique_val_list:

               if unique_val not in string_index_dict:
                    string_index_dict[unique_val]=allocator
                    allocator += 1


            
            df[column] = list(map(helper_converter,df[column])) 

    return df
def converting_nans(df):
    columns = df.columns.values
    
    for column in columns:
        if df[column].dtype == np.float64 or df[column].dtype == np.int64:
            df[column].fillna(df[column].mean(),inplace=True)


    return df
df = converting_strings(df)
df = converting_nans(df)
print(df.head())

df_final = pd.read_csv('data/final.csv',sep=',')
df_final1 = pd.DataFrame(columns=['userid','rank'])
for index,row in df_final.iterrows():
	if row['slope'] > 3.5:
		df_final1 = df_final1.append({'userid':row['userid'],'rank':10},ignore_index=True)

	elif row['slope'] <=3.5 and row['slope']>2.5:
		df_final1 = df_final1.append({'userid':row['userid'],'rank':9},ignore_index=True)

	elif row['slope'] <=2.5 and row['slope']>1.5:
		df_final1 = df_final1.append({'userid':row['userid'],'rank':8},ignore_index=True)

	elif row['slope'] <=1.5 and row['slope']>0.5:
		df_final1 = df_final1.append({'userid':row['userid'],'rank':7},ignore_index=True)

	elif row['slope'] <=0.5 and row['slope']>0.0:
		df_final1 = df_final1.append({'userid':row['userid'],'rank':6},ignore_index=True)

	elif row['slope'] <=0.0 and row['slope']> -0.5:
		df_final1 = df_final1.append({'userid':row['userid'],'rank':5},ignore_index=True)

	elif row['slope'] <= -0.5 and row['slope']> -1.5:
		df_final1 = df_final1.append({'userid':row['userid'],'rank':4},ignore_index=True)

	elif row['slope'] <= -1.5 and row['slope']> -2.5:
		df_final1 = df_final1.append({'userid':row['userid'],'rank':3},ignore_index=True)

	elif row['slope'] <= -2.5 and row['slope']> -3.5:
		df_final1 = df_final1.append({'userid':row['userid'],'rank':2},ignore_index=True)

	elif row['slope'] < -3.5:
		df_final1 = df_final1.append({'userid':row['userid'],'rank':1},ignore_index=True)


print(df_final1.head(20))
#df_final1.to_csv('rank.csv',sep=',')
df.to_csv('data/user_recall_final.csv',sep=',')




'''

df = pd.read_csv('data/book1.csv',sep=',')
df.dropna(0,inplace=True)
#ssprint(df)
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
'''
def converting_nans(df):
    columns = df.columns.values
    
    for column in columns:
        if df[column].dtype == np.float64 or df[column].dtype == np.int64:
            df[column].fillna(df[column].mode(),inplace=True)
'''


X = df.drop(['userid','rank','rank4','slope','flag'],1)
X = increase_dimensionality(X)
X = preprocessing.scale(X)
print(X.shape)
y = df['rank4']
kbest = SelectKBest(f_classif,k=50)
X = kbest.fit_transform(X,y)

pca = decomposition.PCA()
X = pca.fit_transform(X) 
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.20,random_state=random.randint(0,42))
print(X.shape)
#clf = RFECV(RandomForestClassifier(n_estimators=20,n_jobs=-1),scoring='accuracy')
#clf= GradientBoostingClassifier()
#clf = XGBClassifier()
clf = LogisticRegression(solver='saga')
#clf = svm.SVC(C=0.01)
#clf = GaussianNB()
#clf = KNeighborsClassifier(n_neighbors=4)
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)

print(accuracy)

pickle.dump(clf,open('model.sav','wb'))

