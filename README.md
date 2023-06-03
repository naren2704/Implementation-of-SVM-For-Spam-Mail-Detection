## Implementation-of-SVM-For-Spam-Mail-Detection
## AIM:

To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:

1.Hardware – PCs

2.Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.

2. Import the dataset to operate on.

3. Split the dataset.

4. Predict the required output.

5. End the program.

## Program:
Program to implement the SVM For Spam Mail Detection..

Developed by: B NARENDRAN

RegisterNumber: 212222240069

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

## Output:
## Data Head:
![image](https://github.com/naren2704/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118706984/9bf86a7f-fc89-45fd-a8be-684c05162ed1)

## Data Info:
![image](https://github.com/naren2704/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118706984/bd09aa8c-0929-4170-9680-93c3904f1ae0)

## Data isnull():
![image](https://github.com/naren2704/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118706984/8ee12770-c47e-4727-8f09-ac8a15f36a2b)

## y_pred:
![image](https://github.com/naren2704/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118706984/f03c34e9-64ce-41ca-8c29-05740708c15e)

## Accuracy:
![image](https://github.com/naren2704/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118706984/75720f69-562c-4f43-a673-b4e950a946c0)

## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
