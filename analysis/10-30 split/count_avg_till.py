import numpy as np
import pandas as pd
import datetime
'''
df = pd.read_csv('D:/Data_For Sugar_Educators-HOD Dashboard_2018_01_01.csv',sep=',')
#df = df.drop(['name','email','phone','created','deviceid','appversion','os','entrydate','readingdevice','firstentrydate','lastentrydate','name','readingtime'],axis=1)
df = df[['userid']]
df_slope = pd.read_csv('slope_userid.csv')

df_slope = (df_slope.sort_values('slope'))
#
df_first = df_slope.iloc[0:500]

li_user = list(set(list(df_first['userid'])))
df_final = pd.DataFrame(columns=['userid','reading-count'])
df_edu = pd.DataFrame(columns=['userid','count'])
df['freq'] = (df.groupby('userid')['userid'].transform('count'))
df = df.drop_duplicates()
c = 0
print(df)
print(len(li_user))
for user in li_user:
    c+=1

    for i,row in df.iterrows():
        if row['userid']==user:

            df_final = df_final.append({'userid':user,'reading-count':row['freq']},ignore_index=True)
            print(row['freq'],'  appended')
    
  
print(df_final)
df_final.to_csv('count_top_users.csv',sep=',')

#print(df.head(100))

df = pd.read_csv('count_top_users.csv',sep=',')
df = df[['reading-count']]

df['reading_count_most'] = (df.groupby('reading-count')['reading-count'].transform('count'))
df = df.drop_duplicates()

print(df.sort_values('reading_count_most',ascending = False))
print(np.mean(np.array(df['reading-count'])))
'''

df = pd.read_csv('performance-measure.csv',sep=',')
df = df[df.performance.str.contains('Very') == True]
#df = df.iloc[0:200]
print(df.head())


df_date = pd.read_csv('D:/Data_For Sugar_Educators-HOD Dashboard_2018_01_01.csv',sep=',')
df_date = df_date.iloc[21:]

df_date = df_date[df_date.readingtime.str.contains('Fasting')==True]
df_date = df_date[['userid','reading','entrydate']]
df_date = df_date[df_date.entrydate != '2560-09-20']
#li_user = list(set(list(df['userid'])))
df_date_user_good = pd.DataFrame(columns = ['userid','avg_reading_count'])
df_date_user_bad = pd.DataFrame(columns = ['userid','avg_reading_count'])
end_date = pd.to_datetime('2017-12-31')
for i,per_row in df.iterrows():
    li_loc = []
    for index,row in df_date.iterrows():
        if row['userid'] == per_row['userid']:
            li_loc.append(row['entrydate'])
            df_date=df_date.drop(index)

            

    if len(li_loc)>1:
        start_date = pd.to_datetime(li_loc[0])
        
        diff = ((end_date - start_date)).days/30
        
        avg_reading = len(li_loc)/diff
        print('userid: ',per_row['userid'],' took average reading of ',avg_reading,' till 31st december ,2017')
        if per_row['performance'] == 'Very Good':
            df_date_user_good = df_date_user_good.append({'userid':per_row['userid'],'avg_reading_count':avg_reading},ignore_index=True)
        elif per_row['performance'] == 'Very Bad':
            df_date_user_bad = df_date_user_bad.append({'userid':per_row['userid'],'avg_reading_count':avg_reading},ignore_index=True)


#df_reading_count = pd.read_csv('count_top_users.csv',sep=',')
'''
li_final_user = list(set(list(df_date_user['userid'])))

for user in li_final_user:
    for index,row in df_reading_count.iterrows():
        if row['userid'] == user:
            cnt = row['reading-count']
            

'''

df_date_user_good.to_csv('final_avg_count_good.csv',sep=',')
print('Very Good users mean',np.mean(np.array(df_date_user_good['avg_reading_count'])))

df_date_user_bad.to_csv('final_avg_count_bad.csv',sep=',')
print('Very Bad users mean',np.mean(np.array(df_date_user_bad['avg_reading_count'])))












