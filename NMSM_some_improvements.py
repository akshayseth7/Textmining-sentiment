# -*- coding: utf-8 -*-

import itertools
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist
from sklearn.metrics import classification_report
from nltk.metrics.association import BigramAssocMeasures

""" Create a function which will evaluate the classifier under different conditions.
 We will see how"""
 
def evaluate_classifier(classifier, feature_function, train_set, test_set):
    
    train_tokenized = [[wt(x), c] for x,c in train_set]
    test_tokenized = [[wt(x), c] for x,c in test_set]    
    
    train_featureset = [(feature_function(d), c) for (d,c) in train_tokenized]
    test_featureset = [(feature_function(d), c) for (d,c) in test_tokenized]
    
    if classifier == "NaiveBayesClassifier":
        trained_classifier = nltk.NaiveBayesClassifier.train(train_featureset)
    else:
        trained_classifier = nltk.MaxentClassifier.train(train_featureset, algorithm="IIS", max_iter=5)
    
    predictions = []
    actuals = [t[1] for t in test_featureset]
    test_nolab = [t[0] for t in test_featureset]

    for t in test_nolab:
        predictions.append(trained_classifier.classify(t))
    
    print "======================Accuracy============================="
    print(nltk.classify.accuracy(trained_classifier, test_featureset))
    print "==========================================================="

    print "\n\n"
    print "===================Classification Report===================" 
    print classification_report(predictions, actuals)
    print "==========================================================="
    print "\n\n"
    trained_classifier.show_most_informative_features(20)

""" Moving on, the next way to select features is to only take the n most informative features —
 basically, the features that convey the most information
 We first need to find the information gain of each word. To do this we must find the 
 Frequencies of each word and conditional frequencies of each word"""


word_fd = FreqDist()
cond_word_fd = ConditionalFreqDist()

# Find a list of words in the Positive sentences
posWords = [wt(x) for x in poslines]
negWords = [wt(x) for x in neglines]

posWords = list(itertools.chain(*posWords))
negWords = list(itertools.chain(*negWords))

# Now fill up distributions
for word in posWords:
    word_fd.inc(word.lower())
    cond_word_fd['pos'].inc(word.lower())
for word in negWords:
    word_fd.inc(word.lower())
    cond_word_fd['neg'].inc(word.lower())

""" The next thing we need to find the highest-information features is the count of words in positive reviews, 
 words in negative reviews, and total words:"""

pos_word_count = cond_word_fd['pos'].N()
neg_word_count = cond_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count

"""The last thing we need to do is use a chi-squared test test (also from NLTK) to score the words.
We find each word’s positive information score and negative information score, add them up, 
and fill up a dictionary correlating the words and scores, which we then return out of the function. 
Chi-squared tests is a great way to see how much information a given input conveys."""

word_scores = {}
for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score

#best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]


""" We then make another function that finds the best n words, given a set of scores and n"""


def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

""" Finally, we can make a feature selection mechanism that returns ‘True’ for a word only 
if it is in the best words list:"""

def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

numbers_to_test = [10, 100, 1000, 10000, 15000]

for num in numbers_to_test:
    print "============================================================================="
    print "\n"
    print 'evaluating best %d word features' % (num)
    best_words = find_best_words(word_scores, num)
    evaluate_classifier("NaiveBayesClassifier", best_word_features, trainset, testset)

