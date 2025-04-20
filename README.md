# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: DEEPAK JG
RegisterNumber:  21224220019
*/
```
```

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
DATA HEAD:

![image](https://github.com/user-attachments/assets/46badc12-3275-41d4-83ca-b7e56481eec4)

DATA INFO:

![image](https://github.com/user-attachments/assets/d8d11ad3-250d-48f8-884f-4a583c381c97)

ISNULL() AND SUM():

![image](https://github.com/user-attachments/assets/5c80437e-18a9-47de-a38f-a8adeb9991c6)

DATA HEAD FOR SALARY:

![image](https://github.com/user-attachments/assets/ba37a025-67ab-4f68-b46a-72d3afd6fdb6)


MEAN SQUARED ERROR:

![image](https://github.com/user-attachments/assets/3bca6650-5276-4357-9433-1d6bf7db822d)

R2 VALUE:

![image](https://github.com/user-attachments/assets/09cc6adb-ba65-4133-acaf-b881d958c9ab)

DATA PREDICTION:

![image](https://github.com/user-attachments/assets/285c46e4-1746-40b5-98e4-d9a1c7de87b0)





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
