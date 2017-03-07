# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# Read data
df_train = pd.read_csv('../data/train.csv', header = 0)
df_test = pd.read_csv('../data/test.csv', header = 0)

# Preprocess
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

le = LabelEncoder()
le.fit(df_train['species'])
train_label = le.transform(df_train['species'])

train_feat = df_train.drop(['id', 'species'], axis = 1)
test_id = df_test['id']
test_feat = df_test.drop(['id'], axis = 1)

scaler = StandardScaler().fit(train_feat)
train_feat = scaler.transform(train_feat)
scaler = StandardScaler().fit(test_feat)
test_feat = scaler.transform(test_feat)

# Train Val Split
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(test_size=0.2, random_state=1)
for train_index, val_index in sss.split(df_train, train_label):
    X_train, X_val = train_feat[train_index], train_feat[val_index]
    y_train, y_val = train_label[train_index], train_label[val_index]

# Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print lda.score(X_val, y_val)

lda.fit(train_feat, train_label)
test_pred = lda.predict_proba(test_feat)

sub = pd.DataFrame(test_pred, columns = le.classes_)
sub.insert(0, 'id', test_id)
sub.to_csv('../output/ldapred.csv', index = False)