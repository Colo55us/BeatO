import numpy as np
import pandas as pd
'''
df_userorder = pd.read_csv('UserOrder.csv',low_memory=False)
df_orderdetail = pd.read_csv('User_Order_Details.csv',low_memory=False)


df_orderdetail = df_orderdetail.iloc[6900:]
df_userorder = df_userorder.iloc[3500:]


df_userorder = df_userorder[['id','userid','collectiondate']]

df_orderdetail = df_orderdetail[['userorderid','service']]

print(df_orderdetail.head())
print('////////////////////////////////////////////////////////////////////////////////////////////')
print(df_userorder.head())

df_userorder.to_excel('userorder-cleaned.xlsx',index=False)
df_orderdetail.to_excel('orderdetail-cleaned.xlsx',index=False)


'''
df_final = pd.read_csv('merged.csv')
df_final = df_final[['userorderid','service','collectiondate','userid']]



df_final = df_final[df_final.service.str.contains('Vinegar')==True]

df_final.to_csv('Vinegar.csv',sep=',')


user = list(set(list(df_final['userid'])))
print(len(user))