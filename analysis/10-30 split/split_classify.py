import numpy as np
import pandas as pd
'''
df = pd.read_csv('Data_For Sugar_Educators-HOD Dashboard_2018_01_01.csv')
df = df[df.entrydate != '2560-09-20']
df = df[df.readingtime.str.contains('Fasting')==True]
df = df[['userid','reading','eduname']]
#df = df.iloc[0:150]
li_user = list(set(list(df['userid'])))

df_100 = pd.DataFrame(columns=['userid','reading','eduname'])
df_150 = pd.DataFrame(columns=['userid','reading','eduname'])
df_200 = pd.DataFrame(columns=['userid','reading','eduname'])
df_250 = pd.DataFrame(columns=['userid','reading','eduname'])
df_300 = pd.DataFrame(columns=['userid','reading','eduname'])
df_300 = pd.DataFrame(columns=['userid','reading','eduname'])

df_per_chng = pd.DataFrame(columns = ['userid','first_reading','last_reading','per_chng','inc/dec','flag'])
user_count = len(li_user)
count = 0
for user in li_user:
    count += 1
    li_loc = []
    for index,row in df.iterrows():
        if row['userid'] == user:
            li_loc.append(row['reading'])
            df = df.drop(index)

    if len(li_loc)>5:
        len_li_first = (np.int64(0.1*len(li_loc)))+1
        li_first = np.mean(np.array(list(li_loc[:len_li_first])))
        len_li_last = int(len(li_loc)*0.3)+1
        li_last = np.mean(np.array(li_loc[-len_li_last:]))

        flag = 0
        per_chng = 0
        if li_first > li_last:
            flag =1
            diff = abs(li_first-li_last)
            per_chng = (diff/li_first) * 100

        else:
            diff = abs(li_first-li_last)
            per_chng = (diff/li_first) * 100

        dec = 'decrease'
        inc = 'increase'
        inc_dec = ' '
        if flag == 0:
            inc_dec = inc
        else:
            inc_dec = dec

        df_per_chng = df_per_chng.append({'userid':user,'first_reading':li_first,'last_reading':li_last,'per_chng':per_chng,'inc/dec':inc_dec,'flag':flag},ignore_index=True)
        print('Userid:',user,'    first_reading: ',li_first,'    last_reading:',li_last,'     per_chng:',per_chng,'   ',count,' out of ',user_count)


df_per_chng.to_csv('percentage-change.csv',sep=',')
''' 
'''
df = pd.read_csv('percentage-change.csv',sep=',')
df_perf = pd.DataFrame(columns=['userid','first_reading','performance',])
df['first_reading'] = df['first_reading'].astype(int)
for index,row in df.iterrows():
    if row['first_reading'] <= 120 :
        print('uid ',row['userid'],row['first_reading'])
        if row['flag'] == 1:
            df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Very Good'},ignore_index=True)
        else:
            df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Very Bad'},ignore_index=True)
    
    elif row['first_reading'] >120 & row['first_reading'] <=150:
        print('uid ',row['userid'],row['first_reading'])
        if row['flag'] == 1:
            if row['per_chng'] > 20 :
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Very Good'},ignore_index=True)
            else:
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Good'},ignore_index=True)
        else:
            if row['per_chng'] < 20 :
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Bad'},ignore_index=True)
            else:
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Very Bad'},ignore_index=True)

    elif row['first_reading'] >150 & row['first_reading'] <=200:
        print('uid ',row['userid'],row['first_reading'])
        if row['flag'] == 1:
            if row['per_chng'] > 25 :
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Very Good'},ignore_index=True)
            else:
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Good'},ignore_index=True)
        else:
            if row['per_chng'] < 25 :
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Bad'},ignore_index=True)
            else:
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Very Bad'},ignore_index=True)

    elif row['first_reading'] >200 & row['first_reading'] <=250:
        print('uid ',row['userid'],row['first_reading'])
        if row['flag'] == 1:
            if row['per_chng'] > 30 :
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Very Good'},ignore_index=True)
            else:
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Good'},ignore_index=True)
        else:
            if row['per_chng'] < 30 :
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Bad'},ignore_index=True)
            else:
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Very Bad'},ignore_index=True)
    elif row['first_reading']>250: 
        print('uid ',row['userid'],row['first_reading'])
        if row['flag'] == 1:
            if row['per_chng'] > 30 :
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Very Good'},ignore_index=True)
            else:
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Good'},ignore_index=True)
        else:
            if row['per_chng'] < 30 :
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Bad'},ignore_index=True)
            else:
                df_perf = df_perf.append({'userid':row['userid'],'first_reading':row['first_reading'],'performance':'Very Bad'},ignore_index=True)




df_perf.to_csv('performance-measure.csv',sep=',')



'''
df_perf = pd.read_csv('performance-measure.csv',sep=',')
df_perf = df_perf[df_perf.performance.str.contains('Very') == True]

print(df_perf.head(100))
