# -*- coding: utf-8 -*-

# This script is a copy from the kaggle tutorial

import csv
import numpy as np
with open('../data/train.csv', 'rb') as f:
    
    csv_obj = csv.reader(f)
    header = csv_obj.next()
    
    data = []
    for row in csv_obj:
        data.append(row)

data = np.array(data)
number_passengers = np.size(data[0::, 1])
number_survivors = np.sum(data[0::, 1].astype(np.float))
survive_ratio = number_survivors/number_passengers 

women_stats = data[(data[0::, 4] == "female"), 0::]

test_file = open('../data/test.csv', 'rb')
test_obj = csv.reader(test_file)
test_header = test_obj.next()

# Part 1. Predict based on gender
 
pred_file = open("../output/genderpred.csv", "wb")
pred_obj = csv.writer(pred_file)

pred_obj.writerow(["PassengerId", "Survived"])
for row in test_obj:
    if row[3] == "female":
        pred_obj.writerow([row[0], '1'])
    else:
        pred_obj.writerow([row[0], '0'])

test_file.close()
pred_file.close()

# Part 2. Predict based on gender, class, and ticket price

fare_ceiling = 40
data[data[0::, 9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0 

fare_b_size = 10
num_of_bins = fare_ceiling / fare_b_size
num_of_cls = 3

survival_table = np.zeros((2, num_of_cls, num_of_bins))

for i in xrange(num_of_cls):
    for j in xrange(num_of_bins):
        women_survive_stats = data[
            (data[0::, 4] == 'female') \
            &(data[0::, 2].astype(np.float) == i+1) \
            &(data[0::, 9].astype(np.float) >= j*fare_b_size)\
            &(data[0::, 9].astype(np.float) < (j+1)*fare_b_size)\
        , 1]
        men_survive_stats = data[
            (data[0::, 4] != 'female') \
            &(data[0::, 2].astype(np.float) == i+1) \
            &(data[0::, 9].astype(np.float) >= j*fare_b_size)\
            &(data[0::, 9].astype(np.float) < (j+1)*fare_b_size)\
        , 1]
        survival_table[0,i,j] = np.mean(women_survive_stats.astype(np.float)) \
            if women_survive_stats.size > 0 else 0
        survival_table[1,i,j] = np.mean(men_survive_stats.astype(np.float)) \
            if men_survive_stats.size > 0 else 0


