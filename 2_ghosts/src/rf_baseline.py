# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# Read data
df_train = pd.read_csv('../data/train.csv', header = 0)
train_len = len(df_train)
df_test = pd.read_csv('../data/test.csv', header = 0)

# Prepare data
df_train['color_id'] = df_train['color'].map({'white':0, 'green':1, \
    'black':2, 'blue':3, 'blood':4, 'clear':5}).astype(int)
df_test['color_id'] = df_test['color'].map({'white':0, 'green':1, \
    'black':2, 'blue':3, 'blood':4, 'clear':5}).astype(int)
 
df_train['type_id'] = df_train['type'].map({'Ghoul':0, 'Goblin':1, \
    'Ghost':2}).astype(int)
    
df_train = df_train.drop(['type', 'color'], axis=1) 
df_test = df_test.drop(['color'], axis=1) 

# Learn
from sklearn.ensemble import RandomForestClassifier

train_data = df_train.values
test_data = df_test.values

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[:, 1:-1], train_data[:, -1])

print 'Train accuracy:'
print forest.score(train_data[:, 1:-1], train_data[:, -1])

# Pred output
output = forest.predict(test_data[:, 1:])

# Write output
import csv
pred_file = open("../output/rfpred.csv", "wb")
pred_obj = csv.writer(pred_file)

obj_list = ['Ghoul', 'Goblin', 'Ghost']
pred_obj.writerow(["id", "type"])
for test_cnt in range(len(output)):
    pred_obj.writerow([int(df_test.id[test_cnt]), obj_list[int(output[test_cnt])]])

pred_file.close()