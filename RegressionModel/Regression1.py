# Multiple Linear Regression

# Importing the libraries
import numpy as np
#import matplot.lib as mo
import pandas as pd
 #importing tyhe dataset
dataset=pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Encoding Categorical data 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
#print(x)

#Splitting the data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#trainiing the model

from sklearn.linear_model import LinearRegression
r=LinearRegression()
r.fit(x_train,y_train)

#Prediction

y_pred=r.predict(x_test)

np.set_printoptions(precision=2)
print("Predicted-       Actual-       ")
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#predicting new values

#y_new=r.predict([[1.0,0.0 ,0.0, 118763, 543, 65555]])
#print(y_new)

#printing the coefficients

#print(r.coef_)
#print(r.intercept_)







