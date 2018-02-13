import numpy as np 
import pandas as pd 

#df = pd.read_csv('C:/Users/BeatO/Documents/garbage_merged_sugar.csv')
'''
df_final = pd.read_csv('merged.csv')
df_final = df_final[['userorderid','service','collectiondate','userid']]

df_usersugar = pd.read_csv('user_sugar.csv',sep=',')

df_final = df_final[df_final.service.str.contains('Vinegar')==True]

#df_final.to_csv('Vinegar.csv',sep=',')

df = df.iloc[28530:]
df = df[df.readingtime.str.contains('Fasting')==True]
#print(df.head(10))
df = df[['userid','entrydate','reading']]
df.to_excel('user_sugar.xlsx',index=False)
print(df)


li_user = list(set(list(df['userid'])))
df_count = pd.DataFrame(columns=['userid','count'])
for user in li_user:
    li_readings = []
    li = df_usersugar.index[df_usersugar.userid==user].tolist()
    df_count = df_count.append({'userid':user,'count':len(li)},ignore_index=True)
    for l in li:
        li_readings.append(df_usersugar.iloc[l].reading)
    




print(df_count)
df_count.to_csv('vineger_reading_count.csv',sep=',')

#for user in li_user:
'''

'''
user = list(set(list(df['userid'])))
#print(len(user))
df_merged = pd.read_csv('merged.csv')
df_vinegar = df_merged[df_merged.service.str.contains('Vinegar')==True]
df_socks = df_merged[df_merged.service.str.contains('Socks')==True]
df_atta = df_merged[df_merged.service.str.contains('Atta')==True]
df_juice = df_merged[df_merged.service.str.contains('Juice')==True]
df_oats = df_merged[df_merged.service.str.contains('Oats')==True]



print(df_merged.groupby('service').size().to_string())



frames = [df_vinegar,df_socks,df_atta,df_juice,df_oats]
'''

df = pd.read_csv('usercount.csv')
for index,row in df.iterrows():
    if row['cont'] == 1:
        df = df.drop(index)

df_count = pd.DataFrame(columns = ['userid','cont'])

for index,row in df.iterrows():
    if row['cont'] <=8:
        df_count = df_count.append({'userid':row['userid'],'cont':1},ignore_index=True)
    elif row['cont']>8 and row['cont']<=16:
        df_count = df_count.append({'userid':row['userid'],'cont':2},ignore_index=True)
    elif row['cont'] >16 and row['cont']<=30:
        df_count = df_count.append({'userid':row['userid'],'cont':3},ignore_index=True)
    elif row['cont'] >30 and row['cont'] <= 45:
        df_count = df_count.append({'userid':row['userid'],'cont':4},ignore_index=True)
    elif row['cont'] > 46 and row['cont'] <=65:
        df_count = df_count.append({'userid':row['userid'],'cont':5},ignore_index=True)
    else:
        df_count = df_count.append({'userid':row['userid'],'cont':6},ignore_index=True)


df_count.to_csv('usercountbagged.csv',sep=',')