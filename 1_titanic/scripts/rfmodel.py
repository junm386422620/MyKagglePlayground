# -*- coding: utf-8 -*-

# Read in train data
import pandas as pd
import numpy as np
df = pd.read_csv('../data/train.csv', header = 0)
df_test = pd.read_csv('../data/test.csv', header = 0)

# Map Sex to 0 1
df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)
df_test['Gender'] = df_test['Sex'].map({'female':0, 'male':1}).astype(int)

df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked_m'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})
df_test['Embarked_m'] = df_test['Embarked'].map({'S':0, 'C':1, 'Q':2})

# Fill missing ages
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) &\
        (df['Pclass'] == j+1)]['Age'].dropna().median()
df['AgeFill'] = df['Age']
df_test['AgeFill'] = df_test['Age']
for i in range(0,2):
    for j in range(0, 3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & \
        (df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]
        df_test.loc[(df_test.Age.isnull()) & (df_test.Gender == i) & \
        (df_test.Pclass == j+1), 'AgeFill'] = median_ages[i, j]
# Feature engineering
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test.Fare.median()

df['FamilySize'] = df['SibSp'] + df['Parch']
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']


df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
df = df.drop(['Age'], axis=1)
passid_test = df_test.PassengerId
df_test = df_test.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
df_test = df_test.drop(['Age'], axis=1)

data_in_nparray = df.values
data_test = df_test.values

# Build Random Forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 500)
forest = forest.fit(data_in_nparray[0::, 1::], data_in_nparray[0::, 0])

print forest.score(data_in_nparray[0::, 1::], data_in_nparray[0::, 0])

output = forest.predict(data_test)

# Write output
import csv
pred_file = open("../output/rfpred.csv", "wb")
pred_obj = csv.writer(pred_file)

pred_obj.writerow(["PassengerId", "Survived"])
for test_cnt in range(len(output)):
    pred_obj.writerow([int(passid_test[test_cnt]), int(output[test_cnt])])

pred_file.close()