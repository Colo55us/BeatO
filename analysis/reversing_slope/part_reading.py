import numpy as np 
import pandas as pd 


df = pd.read_csv('D:/Data_For Sugar_Educators-HOD Dashboard_2018_01_01.csv',sep=',')

li_userid = list(set(list(df['userid'])))

for uid in li_userid:
	for index,row in df.iterrows():
		li_loc_read = []
		
		 
