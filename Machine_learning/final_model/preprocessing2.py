import pandas as pd 
import numpy as np 

df = pd.read_csv('data/book1.csv',sep=',')

df_f = pd.DataFrame(columns =['userid','rank4','flag'])


for i,row in df.iterrows():
	if row['slope'] > 2.5:
		df_f = df_f.append({'userid':row['userid'],'rank4':4,'flag':0},ignore_index=True)

	elif row['slope'] <=2.5 and row['slope']>0:
		df_f = df_f.append({'userid':row['userid'],'rank4':3,'flag':0},ignore_index=True)

	elif row['slope'] <=0 and row['slope']> -2.5:
		df_f = df_f.append({'userid':row['userid'],'rank4':2,'flag':1},ignore_index=True)

	elif row['slope'] < -2.5:
		df_f = df_f.append({'userid':row['userid'],'rank4':1,'flag':1},ignore_index=True)



df_f.to_csv('rank4.csv',sep=',')