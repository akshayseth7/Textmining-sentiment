# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:08:25 2015

@author: aditya
"""
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
    
    
X, y = create_tfidf_training_data(labelled)

# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating functions for different classifiers

# Decision Tree Classifier
def train_dtc(X, y):
    """
    Create and train the Decision Tree Classifier.
    """
    dtc = DecisionTreeClassifier()
    dtc.fit(X, y)
    return dtc

# K-Nearest Neighbour Classifier
def train_knn(X, y, n, weight):
    """
    Create and train the k-nearest neighbor.
    """
    knn = KNeighborsClassifier(n_neighbors = n, weights = weight, metric = 'cosine', algorithm = 'brute')
    knn.fit(X, y)
    return knn

# SVM Classifier
def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm

# Naive Bayes Classifier
def train_nb(X, y):
    """
    Create and train the Naive Baye's Classifier.
    """
    clf = MultinomialNB().fit(X, y)
    return clf

# Logistic Regression Classifier
def train_lr(X, y):
    """
    Create and train the Naive Baye's Classifier.
    """
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr

# Start the predictions

# Decision Tree Classifier
dt = train_dtc(X_train, y_train)
predDT = dt.predict(X_test)

# Print the classification rate
print(dt.score(X_test, y_test))

# K-Nearest Neighbours Classifier
kn = train_knn(X_train, y_train, 5, 'distance')
predKN = kn.predict(X_test)

# Print the classification rate
print(kn.score(X_test, y_test))

# SVM Classifier
sv = train_svm(X_train, y_train)
predSVM= sv.predict(X_test)

# Print the classification rate
print(sv.score(X_test, y_test))

# Naive Bayes Classifier
nb = train_nb(X_train, y_train)
predNB = nb.predict(X_test)

# Print the classification rate
print(nb.score(X_test, y_test))

# Logistic Regression Classifier
lr = train_lr(X_train, y_train)
predLR = lr.predict(X_test)

# Print the classification rate
print(lr.score(X_test, y_test))

# Create a Dataframe with all predictions in it
columns = ['DecisionTree', 'NearestNeighbor', 'SVM','NaiveBayes','Logistic']
ensDF = pd.DataFrame({'DecisionTree': predDT, 'NearestNeighbor': predKN, 'SVM': predSVM, 'NaiveBayes': predNB, 'Logistic': predLR}, columns = columns)
print ensDF

# Create another dataframe with the counts of each predicted class
majority = ensDF.apply(pd.Series.value_counts, axis=1)[['foreign','non_foreign']].fillna(0)
print majority

# Create a column called "decision" which will hold the majority voted class for each observation
majority['decision'] = ""  
for i in range(0, majority.shape[0]):
    if majority['foreign'][i]>majority['non_foreign'][i]:
        majority['decision'][i] = 'foreign'
    else:
        majority['decision'][i] = 'non_foreign'

# View the data frame
print majority

# View the accuracy of the manority-voting-ensemble method
accuracy = accuracy_score(y_test, majority['decision'])
print accuracy