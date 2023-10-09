import pandas as pd 
df=pd.read_csv('data')
df.head()

df=df.dropna()

x=df.drop('label',axis=1)
y=df['label']

import tensorflow as tf 

from tensorflow.keras.layers import LSTM 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequencial
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot 

voc_size=5000

message=x.copy()
message.reset_index(inplace=True)

import nltk 
import re 
from nltk.corpus import stopwards 


from nltk.stem.porter import PorterStemmer 
ps=PorterStemmer()
corpus=[]
for i in range(0,len(message)):
    print(i)
    review=re.sub('[a-zA-Z]',' ',message['title'][i])
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word)for word in review if not word in stopwards.words('english')]
    review=' '.join(review)
    corpus.append(review)


onehot_rep=[one_hot(words,voc_size)for words in corpus]
print(onehot_rep)

sent_len=20

embedded_docs=pad_sequences(onehot_rep,padding='pre',maxlen=sent_len)
print(embedded_docs)

embedding_vector_features=40
model=Sequencial()
model.add(Embedding(voc_size,embedding_vector_features,input_len=sent_len))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1,activation='sigmoid'))
model.complie(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

import numpy as np 
x_final=np.array(embedded_docs)
y_final=np.array(y)

from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64)

y_pred=model.predict_classes(x_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

##############

from tensorflow.keras.layers import Dropout 

embedding_vector_features=40    
model=Sequencial()
model.add(Embedding(voc_size,embedding_vector_features,input_len=sent_len))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

