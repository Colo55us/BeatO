import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder,StandardScaler,scale,LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import model_selection
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier


df = pd.read_csv('data/book1.csv',sep=',')


#df.drop(['userid','slope','flag','rank'],1,inplace=True)

X = np.array(df.drop(['userid','rank','rank4','slope','flag'],1))
y = np.array(df['rank4'])
print(y.size)
print(X.size)
X = scale(X)
print(X.shape)
#ohe = OneHotEncoder(categorical_features = [1])
#X = ohe.fit_transform(X)
#X = X[:,1:]

sc = StandardScaler(with_mean=False)
#X = scale(X,mean=True)
X = sc.fit_transform(X)
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y,num_classes=None)
print(y.shape)
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2)

def model():
    clf = Sequential()

    clf.add(Dense(100, kernel_initializer='uniform', activation='relu', input_dim=28))
    clf.add(Dense(60, kernel_initializer='uniform', activation='relu'))
    clf.add(Dense(30, kernel_initializer='uniform', activation='relu'))
    clf.add(Dense(output_dim=6, kernel_initializer='uniform', activation='softmax'))

    clf.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return clf

estimator = KerasClassifier(build_fn=model,epochs=300,batch_size=5,verbose=0)
#clf.fit(np.array(X_train),np.array(y_train),batch_size=5,epochs=150)
kfold = model_selection.KFold(n_splits=10,shuffle=True,random_state=0)
#y_pred = clf.predict(X_test,y_test)
results = model_selection.cross_val_score(estimator,X,y,cv=kfold)
print(results.mean())


