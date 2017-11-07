# -*- coding: utf-8 -*-


from nltk.metrics import ConfusionMatrix
from nltk.classify import NaiveBayesClassifier
from nltk.classify import MaxentClassifier
from nltk.tokenize import word_tokenize as wt

# Create tokenized pair of reviews and labels to be used further
train_tokenized = [[wt(x), c] for x,c in trainset]
test_tokenized = [[wt(x), c] for x,c in testset]

# Creating a function which will take the tokenized words and  return a dictionary of the form "word: True".
# This is done in order to comply with the classifier's requirements in nltk
def word_feats(words):
    return dict([(word, True) for word in words])


# Transforming all the words in the train and test sets to a form acceptable to the nltk classifier
train_featureset = [(word_feats(d), c) for (d,c) in train_tokenized]
test_featureset = [(word_feats(d), c) for (d,c) in test_tokenized]

# Creating a Naive Bayesian Classifier and evaluating it
classifier_nb = nltk.NaiveBayesClassifier.train(train_featureset)

# Accuracy of the Naive Bayesian Classifier
print(nltk.classify.accuracy(classifier_nb, test_featureset))

# Create one empty list which will hold all the predictions
predictions_nb = []

# Create one list holding all the actuals
actuals = [t[1] for t in test_featureset]

# Create a test set with no labels
test_nolab = [t[0] for t in test_featureset]

# Append to the empty list holding all predictions
for t in test_nolab:
    predictions_nb.append(classifier_nb.classify(t))

# Create and view Confusion Matrix   
cm = ConfusionMatrix(predictions_nb, actuals)
print cm

# View the most important words for classification
classifier_nb.show_most_informative_features(20)


#### Repeating the above steps for creating a Maximum Entropy Classifier

classifier_me = nltk.MaxentClassifier.train(train_featureset, algorithm="IIS", max_iter=5)
print(nltk.classify.accuracy(classifier_me, test_featureset))

predictions_me = []


for t in test_nolab:
    predictions_me.append(classifier_me.classify(t))

cm = ConfusionMatrix(predictions_me, actuals)
print cm

classifier_me.show_most_informative_features(20)

