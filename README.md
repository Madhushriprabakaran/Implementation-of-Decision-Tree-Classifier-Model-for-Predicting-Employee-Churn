# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start the program.
2.Import pandas module and import the required data set.
3.Find the null values and count them.
4.Count number of left values.
5.From sklearn import LabelEncoder to convert string values to numerical values.
6.From sklearn.model_selection import train_test_split.
7.Assign the train dataset and test dataset.
8.From sklearn.tree import DecisionTreeClassifier.
9.Use criteria as entropy.
10.From sklearn import metrics.
11.Find the accuracy of our model and predict the require values.
12.End the program.
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Madhushri 
RegisterNumber:212224040178
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee (1).csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
Data.head():

<img width="588" height="424" alt="image" src="https://github.com/user-attachments/assets/d1bb255b-ade4-4baa-84dd-3cc8fc30854f" />

Data.info():

<img width="587" height="425" alt="image" src="https://github.com/user-attachments/assets/4094496c-cecc-4fcb-a626-337f5804c244" />

isnull() and sum():

<img width="352" height="278" alt="image" src="https://github.com/user-attachments/assets/3d43120f-bb78-4acc-b2f3-9e21809479d4" />

Data Value Counts():


<img width="294" height="90" alt="image" src="https://github.com/user-attachments/assets/b6768b50-f486-4ce8-bb46-b17c68012158" />

Data.head() for salary:

<img width="946" height="174" alt="image" src="https://github.com/user-attachments/assets/339680fb-b8bc-4244-a9c5-e39845f817a1" />

x.head:

<img width="921" height="166" alt="image" src="https://github.com/user-attachments/assets/d62770eb-f240-473a-868c-7c5a217563f5" />

Accuracy Value:

<img width="92" height="41" alt="image" src="https://github.com/user-attachments/assets/05ea88cc-6ce2-46e2-92aa-7ba5f96b33b1" />

Data Prediction:

<img width="554" height="74" alt="image" src="https://github.com/user-attachments/assets/a21c31c7-8eb6-4a76-8500-404acf4119f1" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
