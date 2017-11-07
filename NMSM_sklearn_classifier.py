# -*- coding: utf-8 -*-


import sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer #as vectorizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import chi2


# Create your Vectorizer function
vectorizer = TfidfVectorizer()
#vectorizer = TfidfVectorizer(ngram_range=(1,2))

# Create datasets without labels to transform to TF-IDF Vectorizer
train_nolab = [t[0].decode('latin-1').encode("utf-8") for t in trainset]
test_nolab = [t[0].decode('latin-1').encode("utf-8") for t in testset]

# Create datasets with only labels to serve as "Y" 
train_lab = [t[1] for t in trainset]
test_lab = [t[1] for t in testset]

# Create TF-IDF Vectorizer from your training features
train_vectors = vectorizer.fit_transform(train_nolab)

# Transform your test features to fit your already trained TF-IDF
test_vectors = vectorizer.transform(test_nolab)

# Check the size of your features and documents for your training and test sets
train_vectors.shape
test_vectors.shape

#cm = ConfusionMatrix(pred, test_lab)


# Create a function which will train the SVM Classifier
def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=5000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm

# Train the SVM
sv = train_svm(train_vectors, train_lab)

# Predict on the Test set using the trained SVM
predSVM= sv.predict(test_vectors)
pred = list(predSVM)

# Create confusion Matrix
cm = ConfusionMatrix(pred, test_lab)

# Print Confusion Matrix
print cm

# Print Accuracy Score
print accuracy_score(pred, test_lab)

# Print a classification report
print classification_report(pred,  test_lab)

# Can also create a Naive Bayes classifier using sklearn

clf = MultinomialNB().fit(train_vectors, train_lab)
predNB = clf.predict(test_vectors)
pred = list(predNB)
cm = ConfusionMatrix(pred, test_lab)
print cm
print accuracy_score(pred, test_lab)
print classification_report(pred,  test_lab)


# What if you want to select only the K best features and train your models?
ch21 = SelectKBest(chi2, k=15000)

# Transform your training and testing datasets accordingly
train_Kbest = ch21.fit_transform(train_vectors, train_lab)
test_Kbest = ch21.transform(test_vectors)

# Train your SVM with the k best selected features
sv = train_svm(train_Kbest, train_lab)
predSVM= sv.predict(test_Kbest)
pred = list(predSVM)
cm = ConfusionMatrix(pred, test_lab)
print cm
print accuracy_score(pred, test_lab)
print classification_report(pred,  test_lab)



clf = MultinomialNB().fit(train_Kbest, train_lab)
predNB = clf.predict(test_Kbest)
pred = list(predNB)
cm = ConfusionMatrix(pred, test_lab)
print cm
print accuracy_score(pred, test_lab)
print classification_report(pred,  test_lab)

# View the selected features
selected_features = list(np.array(vectorizer.get_feature_names())[ch21.get_support()])
print selected_features






