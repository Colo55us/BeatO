import numpy as np
import pandas as pd

df = pd.read_csv('userid_edu_top.csv')
df.dropna(axis=0,inplace=True)
#print(df)
df = df[df.eduname.str.contains('Demo')==False]
print(df.groupby('eduname').size())

