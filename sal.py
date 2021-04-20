import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import sklearn 
print("libraries imported")

df = pd.read_csv('hike.csv')


df['Experience'].fillna(0,inplace=True)
df['Test_score'].fillna(df['Test_score'].mean(),inplace=True)

X = df.iloc[:,:3]


def convert_int(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':7,'ten':10,'eleven':11,0:0}
    return word_dict[word]

X['Experience'] = X['Experience'].apply(lambda x: convert_int(x))

Y = df.iloc[:,-1]

from sklearn.linear_model import LinearRegression
regressor  = LinearRegression()

regressor.fit(X,Y)

pickle.dump(regressor,open('sal_model.pkl','wb'))



print(X)

model = pickle.load(open('sal_model.pkl','rb'))
result = model.predict([[2,9,6]])

print(result)