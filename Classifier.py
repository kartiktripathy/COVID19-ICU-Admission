'''COVID 19 ICU Admission'''

#importing the libraries
import numpy as np
import pandas as pd

#importing the dataset
dataset = pd.read_excel('Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
dataset = pd.DataFrame(dataset)

#Removing the unnecessary columns
dataset.drop(["AGE_PERCENTIL"], axis = 1, inplace = True)
dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='_MEAN')))]
dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='_MIN')))]
dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='_MAX')))]
dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='_DIFF')))]

#encoding the range like parameters with numerals
'''we have encoded:
0-2 as 1
2-4 as 2 
4-6 as 3
6-12 as 4
ABOVE_12 as 5'''
dataset["WINDOW"].replace({"0-2": "1", "2-4": "2","4-6": "3","6-12": "4","ABOVE_12": "5"}, inplace=True)

#Splitting into dependant and independant variable sets
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,54]

#Removing the infinite and null values
X = X.replace([np.inf, -np.inf],np.nan) #removing the infinite values
X = X.replace(np.nan,9) #replacing the null values with 9(any random number)

#Filling the missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=9, strategy='mean') #now treat 9 as missing values
imputer = imputer.fit(X.iloc[:,11:53])
X.iloc[:,11:53] = imputer.transform(X.iloc[:,11:53])

#Splitting the dataset into test and train set
from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Fitting Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500  ,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

#Predicting The Test Results
y_pred = classifier.predict(x_test)

#Making the confusion MAtrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
ac = accuracy_score(y_test,y_pred)
print("The Confusion Matrix :")
print(cm)
print("Accuracy Percentage : ",ac*100,"%")

#Pickle Dumping the classifier
import pickle
pickle.dump(classifier , open('ICU_Admission.pkl','wb'))

