# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 23:51:09 2015

@author: aditya
"""

# Suppose you have a scenario in which you have documents belonging to overlapping classes
# We can build multiple binary classifiers which will predict the document's membership in each of the classes
# We can then build a Matrix which shows which of the classes the document belongs to
# We select documents with three overlapping "subjects": Terror, Civil War and Capital Punishment

# Create a list of fields to keep
col_list = ['subjects', 'abstract']

# Terror
amnesty_terror = amnesty_clean[amnesty_clean['subjects'].str.contains("TERROR")]
amnesty_terror['subjects'] = amnesty_terror['subjects'] .apply(lambda x: "Terror")
amnesty_terror = amnesty_terror[col_list]

# Remove NA Values

amnesty_terror = amnesty_terror.dropna()

# Create a list with "Terror" Documents and their abstracts
labelled_terror=[]
for row in amnesty_terror.iterrows():
    index, data = row
    labelled_terror.append(data.tolist())
    
# Civil War
amnesty_cv = amnesty_clean[amnesty_clean['subjects'].str.contains("CIVIL WAR")]
amnesty_cv['subjects'] = amnesty_cv['subjects'] .apply(lambda x: "Civil_War")
amnesty_cv= amnesty_cv[col_list]

# Remove NA Values
amnesty_cv = amnesty_cv.dropna()

# Create a list with "Civil War" Documents and their abstracts
labelled_cv=[]
for row in amnesty_cv.iterrows():
    index, data = row
    labelled_cv.append(data.tolist())
    
# Capital Punishment

amnesty_cp = amnesty_clean[amnesty_clean['subjects'].str.contains("CAPITAL PUNISHMENT")]
amnesty_cp['subjects'] = amnesty_cp['subjects'] .apply(lambda x: "Capital_Punishment")
amnesty_cp= amnesty_cp[col_list]


# Remove NA Values
amnesty_cp = amnesty_cp.dropna()

# Create a list with "Capital Punishment" Documents and their abstracts
labelled_cp=[]
for row in amnesty_cp.iterrows():
    index, data = row
    labelled_cp.append(data.tolist())

# Create one list with all the documents and their abstracts
labelled = labelled_terror+labelled_cv+labelled_cp

# Choose the classifier and set up the training function

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm
# Create tf-idfs of all the abstracts and separate out the X (predictors) and the y (response)
X_maj, y_maj = create_tfidf_training_data(labelled)

# As before, we split our tf-idf and y values into training and testing sets
Xmaj_train, Xmaj_test, y_majtrain, y_majtest = train_test_split(X_maj, y_maj, test_size=0.2, random_state=42)

# Now we must create binary classifiers for each class, which will predict whether the document belongs
# to that class or not
# The trick to build such a classifier is to split the data into roughly balanced data sets and 
# then train each classifier on such a dataset
# For example, for a Terror or NoTerror classifier, we train the SVM classifier on the "Terror" documents
# and label all the other documents as "NoTerror". We also take care to create a balanced set so that the 
# classifier is not biased towards the majority class in its predictions

# Terror
# Separate out all instances where the y value was "Terror" and not "Terror"
y_train_terror=[item if item== "Terror" else "NoTerror" for item in y_majtrain]
y_test_terror=[item if item== "Terror" else "NoTerror" for item in y_majtest]


# We select the entire set of tf-idf documents to use as training set for "Terror"
X_train_terror = Xmaj_train
X_test_terror=Xmaj_test
# Train the SVM classifier to predict Terror or No_Terror
svm = train_svm(X_train_terror, y_train_terror)



# Predict on the Test set
pred = svm.predict(X_test_terror)

# Print the classification rate
print(svm.score(X_test_terror, y_test_terror))
##### Write your own code to train an SVM Classifier to predict Terror or NoTerror #####

# Predict on test set

##### Write your own code to predict on test set #####


# Similarly, let's create our test and train sets for Capital Punishment
labelled_cp = [(label, text) if label== "Capital_Punishment" else ("NoCP", text) for (label, text) in labelled]

i_cp_n = [i for i, (label, text) in enumerate(labelled_cp) if label == "NoCP"]
i_cp_y = [i for i, (label, text) in enumerate(labelled_cp) if label == "Capital_Punishment"]

# We select 70 observations as No_Capital_Punishment because there were only about 68 instances of Capital_Punishment labels
i_cp_all = random.sample(i_cp_n, 70) + i_cp_y

data_cp = [labelled_cp[i] for i in i_cp_all]

X_cp, y_cp = create_tfidf_training_data(data_cp)

# We now split out tf-idf matrix to be roughly about the same size as the size of our response set

X_train_cp, X_test_cp, y_train_cp, y_test_cp = train_test_split(X_cp, y_cp, test_size=0.2, random_state=42)

# Train the SVM classifier to predict Capital_Punishment or No_Capital_Punishment
svm_cp = train_svm(X_train_cp, y_train_cp)
##### Write your own code to train an SVM Classifier to predict Capital_Punishment or No_Capital_Punishment #####

# Predict on test set
print(svm_cp.score(X_test_cp, y_test_cp))
##### Write your own code to predict on test set #####

# Similarly, let's create our test and train sets for Civil War
labelled_cv = [(label, text) if label== "Civil_War" else ("NoCV", text) for (label, text) in labelled]

i_cv_n = [i for i, (label, text) in enumerate(labelled_cv) if label == "NoCV"]
i_cv_y = [i for i, (label, text) in enumerate(labelled_cv) if label == "Civil_War"]


# We select 110 observations as No_Civil_War because there were only about 110 instances of Capital_Punishment labels
i_cv_all = random.sample(i_cv_n, 110) + i_cv_y

data_cv = [labelled_cv[i] for i in i_cv_all]

X_cv, y_cv = create_tfidf_training_data(data_cv)
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_cv, y_cv, test_size=0.2, random_state=42)

# Train the SVM classifier to predict Civil_War or No_Civil_War
svm_cv = train_svm(X_train_cv, y_train_cv)



##### Write your own code to train an SVM Classifier to predict Civil_War or No_Civil_War #####

# Predict on test set
print(svm_cv.score(X_test_cv, y_test_cv))

##### Write your own code to predict on test set #####

# Now let's select 4 documents at random from the entire test set and check their class membership.
# Let these documents be document number 20, 60, 71 and 42
checklist = [20,60,71,42]

# Let's predict their class membership through each binary classifier


# Now to create the final matrix, we first digitize all the class labels and create the matrix

# The Matrix

print ## The Matrix ##





