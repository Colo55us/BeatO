import numpy as np
import pandas as pd

df = pd.read_csv('D:/Data_For Sugar_Educators-HOD Dashboard_2018_01_01.csv',sep=',')
df_slope = pd.read_csv('slope_userid.csv')

df_slope = (df_slope.sort_values('slope'))
#
df_first = df_slope.iloc[0:700]

li_user = list(set(list(df_first['userid'])))

df_edu = pd.DataFrame(columns=['userid','eduname'])
print(len(li_user))

for user in li_user:
	for ind,row in df.iterrows():
		if row['userid']==user:
			if row['eduname']:
				df_edu = df_edu.append({'userid':user,'eduname':row['eduname']},ignore_index=True)
				print(row['eduname'] ,'  inserted')
				break


df_edu.to_csv('userid_edu_top.csv',sep=',',header=True)

