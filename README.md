# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
   
2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for the marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
*/
```

## Output:
#### df.head()
![image](https://github.com/mathes6112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477782/8933eaf7-cb60-4fd5-8133-8536237ca128)
#### df.tail()
![image](https://github.com/mathes6112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477782/21fe3232-3431-4b82-85d5-a07388a8d422)
#### Values of X:
![image](https://github.com/mathes6112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477782/947c396a-7e1f-4cb0-8557-a40722d466ca)
#### Values of Y:
![image](https://github.com/mathes6112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477782/2dbfc20d-abc2-49d1-9394-efeac2e9ebce)
#### Values of Y prediction:
![image](https://github.com/mathes6112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477782/875cce27-8ee8-4710-9188-8d7630797ec3)
#### Values of Y test:
![image](https://github.com/mathes6112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477782/083c7c55-1b83-48c7-a4bb-2df59d8f9d9a)
#### Training set graph:
![image](https://github.com/mathes6112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477782/169b3119-00a7-41ef-9f7a-db65a2b787b2)
#### Test set graph:
![image](https://github.com/mathes6112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477782/f091f5d8-11f5-485c-bf6a-4041ac375b11)
#### Value of MSE,MAE & RMSE:
![image](https://github.com/mathes6112004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477782/161e3e52-a2a6-4abc-ab20-0904490efbb2)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
