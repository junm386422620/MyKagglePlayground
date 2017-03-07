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

df_train['hair_soul_bone'] = df_train['hair_length']*df_train['bone_length'] \
    *df_train['has_soul']
df_test['hair_soul_bone'] = df_test['hair_length']*df_test['bone_length'] \
    *df_test['has_soul']
df_train['hair_bone_flesh'] = df_train['hair_length']*df_train['bone_length'] \
    *df_train['rotting_flesh']
df_test['hair_bone_flesh'] = df_test['hair_length']*df_test['bone_length'] \
    *df_test['rotting_flesh']
#df_train['soul_flesh'] = df_train['has_soul']*df_train['rotting_flesh']    
#df_test['soul_flesh'] = df_test['has_soul']*df_test['rotting_flesh']  

type_id = df_train.type_id
df_train = df_train.drop(['type', 'type_id', 'color', 'color_id'], axis=1) 
df_test = df_test.drop(['color', 'color_id'], axis=1) 

# Learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

train_data = df_train.values
test_data = df_test.values
train_target = type_id.values

#Splitting data for validation
Xtrain, Xtest, ytrain, ytest = train_test_split(train_data[:, 1:], train_target, \
    test_size = 0.2, random_state = 36)


parameter_grid = {'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
                  'multi_class' : ['ovr', 'multinomial'],
                  'C' : [0.005, 0.01, 1, 10, 100, 1000],
                  'tol': [0.0001, 0.001, 0.005]
                 }

logreg = GridSearchCV(LogisticRegression(), param_grid=parameter_grid, cv=StratifiedKFold(5))

logreg.fit(Xtrain, ytrain)

print 'val accuracy:'
print logreg.score(Xtest, ytest)

# Pred output
output = logreg.predict(test_data[:, 1:])

# Write output
import csv
pred_file = open("../output/logpred.csv", "wb")
pred_obj = csv.writer(pred_file)

obj_list = ['Ghoul', 'Goblin', 'Ghost']
pred_obj.writerow(["id", "type"])
for test_cnt in range(len(output)):
    pred_obj.writerow([int(df_test.id[test_cnt]), obj_list[int(output[test_cnt])]])

pred_file.close()