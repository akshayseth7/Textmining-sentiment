# -*- coding: utf-8 -*-


import pandas as pd

training = pd.read_csv("train.csv")
testing = pd.read_csv("test.csv")

train_pos = training[(training.Sentiment == 'positive')]
train_neg = training[(training.Sentiment == 'negative')]



train_pos_list = []
for i,t in train_pos.iterrows():
    train_pos_list.append([t.text.lower(), 1])

train_neg_list = []
for i,t in train_neg.iterrows():
    train_neg_list.append([t.text.lower(), -1])



test_pos = testing[(testing.Sentiment == 'positive')]
test_neg = testing[(testing.Sentiment == 'negative')]

test_pos_list = []
for i,t in test_pos.iterrows():
    test_pos_list.append([t.text.lower(), 1])

test_neg_list = []
for i,t in test_neg.iterrows():
    test_neg_list.append([t.text.lower(), -1])



# Create the train set and the test set by attaching labels to text to form a 
# list of tupes (sentence, label). Labels are 1 for positive, -1 for negatives
trainset = train_pos_list + train_neg_list
testset = test_pos_list + test_neg_list
