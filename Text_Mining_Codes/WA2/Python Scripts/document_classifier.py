# -*- coding: utf-8 -*-
"""
Created on Wed Sep 09 12:24:01 2015

@author: aditya
"""


# Import all packages
import random
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import metrics

# Import the file which we will use for classification
amnesty=pd.read_csv("amnesty-related.csv")


# Create the list which you would like to filter on
desk=['Foreign Desk', 'Foreign']

# Create the list which we would consider to be dirty articles
dirty_list=['Obituary; Biography','Blog','Biography','Obituary','Web Log']

# Create a clean data frame
amnesty_clean=amnesty[~amnesty.type.isin(dirty_list)]

# Separate out the data frame into the two separate data frames, one each for each of the classes we want labelled

foreign=amnesty_clean[amnesty_clean.desk.isin(desk)]
non_foreign=amnesty_clean[~amnesty_clean.desk.isin(desk)]


# Convert all the different values in the "desk" column to one, uniform value, "foreign"

foreign['desk'] = foreign['desk'].apply(lambda x: "foreign")
print pd.unique(foreign.desk.ravel())

# Check the unique values for the non_foreign dataframe's "desk" column and convert it to one standard value - "non-foreign"

print pd.unique(non_foreign.desk.ravel())
non_foreign['desk'] = non_foreign['desk'].apply(lambda x: "non_foreign")
print pd.unique(non_foreign.desk.ravel())


# Create one big data frame by joining the two dataframes of foreign and non-foreign

onebig = foreign.append(pd.DataFrame(data = non_foreign))

# Check the column values and retain only those of interest, namely "desk" and "abstract"

print onebig.columns.values 
col_list = ['desk', 'abstract']
onebig = onebig[col_list]
print onebig.columns.values 

# Remove all rows with no abstract

onebig=onebig.dropna()

# Create a list which has "labelled" instances of Foreign and Non_foreign alongwith the raw text

labelled=[]
for row in onebig.iterrows():
    index, data = row
    labelled.append(data.tolist())

print labelled

# Create a function which will store the TF-IDF for each of the abstracts which we will then use as features to train upon
def create_tfidf_training_data(docs):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list. 

    The function returns both the class label vector (y) and 
    the corpus token/feature matrix (X).
    """
    # Create the training data class labels
    y = [d[0] for d in docs]

    # Create the document corpus list
    corpus = [d[1] for d in docs]

    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    return X, y
    

# Separate out the X (predictors) and the y (response)    
X, y = create_tfidf_training_data(labelled)
# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Number of features in the Training Set 
len(X_train.data)

# Number of features in the Test Set
len(X_test.data)

# Number of documents in Training set
len(y_train)


# Number of documents in Training set
len(y_test)

# Number of documents of the 'foreign' category in the training set

y_train.count('foreign')

# Number of documents of the 'non_foreign' category in the training set

y_train.count('non_foreign')


# Create a function which will train a Support Vector Classifier

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm

# Train the SVM - using all the data

svm = train_svm(X_train, y_train)



# Predict on the Test set
pred = svm.predict(X_test)

# Print the classification rate
print(svm.score(X_test, y_test))

labels = list(set(y_train))

# Print the confusion matrix
import matplotlib.pyplot as plt
import pylab as pl
cm = confusion_matrix(y_test, pred, labels)
print(cm)

# Make a prettier Confusion Matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
pl.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
pl.xlabel('Predicted')
pl.ylabel('True')
pl.show()

# Printing Metrics

print(metrics.classification_report(y_test, pred, target_names=list(set(y_test))))

# Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)

# Predict on test data
pred = clf.predict(X_test)

# Print the classification rate
print(clf.score(X_test, y_test))

# Print the confusion matrix
import matplotlib.pyplot as plt
import pylab as pl
cm = confusion_matrix(y_test, pred, labels)
print(cm)

# Make a prettier Confusion Matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
pl.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
pl.xlabel('Predicted')
pl.ylabel('True')
pl.show()

# Printing Metrics

print(metrics.classification_report(y_test, pred, target_names=list(set(y_test))))

