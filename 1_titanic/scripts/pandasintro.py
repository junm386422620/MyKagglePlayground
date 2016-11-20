# -*- coding: utf-8 -*-

import csv
import numpy as np
with open('../data/train.csv', 'rb') as f:
    csv_obj = csv.reader(f)
    header = csv_obj.next()
    data = []
    for row in csv_obj:
        data.append(row)
data = np.array(data)

import pandas as pd
df = pd.read_csv('../data/train.csv', header = 0)

import pylab as P
df['Age'].dropna().hist(bins = 20, range=(0,100), alpha = 0.5)
P.show()

df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)

median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) &\
        (df['Pclass'] == j+1)]['Age'].dropna().median()
df['AgeFill'] = df['Age']
for i in range(0,2):
    for j in range(0, 3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & \
        (df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]
        #df[(df.Age.isnull()) & (df.Gender == i) & \
        #(df.Pclass == j+1)].AgeFill = median_ages[i, j]
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
df = df.drop(['Age'], axis=1)

data_in_nparray = df.values