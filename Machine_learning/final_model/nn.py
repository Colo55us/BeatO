import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder,StandardScaler,scale,LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import model_selection
from keras.utils import to_categorical

df = pd.read_csv('data/book1.csv',sep=',')


#df.drop(['userid','slope','flag','rank'],1,inplace=True)

X = np.array(df.drop(['userid','rank','rank4','slope','flag'],1))
y = np.array(df['rank4'])
print(y.size)
#X = scale(X)
print(X.shape)
ohe = OneHotEncoder(categorical_features = [1])
X = ohe.fit_transform(X)
#X = X[:,1:]

sc = StandardScaler(with_mean=False)
#X = scale(X,mean=True)
X = sc.fit_transform(X)
encoder = LabelEncoder()
y = encoder.fit_transform(y)
#y = encoder.transform(y)
y = to_categorical(y,num_classes=None)

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2)
clf = Sequential()

print (X.shape)

clf.add(Dense(output_dim=50, init='uniform', activation='relu', input_dim=79))
clf.add(Dense(output_dim=50, init='uniform', activation='relu'))
clf.add(Dense(output_dim=4, init='uniform', activation='softmax'))

clf.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

clf.fit(np.array(X_train),np.array(y_train),batch_size=5,epochs=150)

y_pred = clf.predict(X_test,y_test)

print(y_pred)

