# -*- coding: utf-8 -*-


# Importing the data to build our topic models on. We will use the test data
# we used in the previous class for our topic modeling

import string
import pandas as pd
import numpy as np
import lda
import lda.datasets
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


testing = pd.read_csv(r'C:\\Users\\A0136039H\\Downloads\\NMSM\\Day3\\test.csv')

test_pos = testing[(testing.Sentiment == 'positive')]
test_neg = testing[(testing.Sentiment == 'negative')]

test_pos_list = []
for i,t in test_pos.iterrows():
    test_pos_list.append([t.text, 1])

test_neg_list = []
for i,t in test_neg.iterrows():
    test_neg_list.append([t.text, -1])



# Create the train set and the test set by attaching labels to text to form a 
# list of tupes (sentence, label). Labels are 1 for positive, -1 for negatives
testset = test_pos_list + test_neg_list

# We are only interested in the reviews, hence we create another dataset with only reviews
fortopic = [x[0] for x in testset]

# Create a variable which will hold the stopwords list
stop = stopwords.words('english')

# Extend the stopwords list with some tokens that we do not need.

stop.extend(['wa','ha','ve'])

# Create a Count Vectorizer which will create the DTM which we need for Topic Modeling
vectorizer = CountVectorizer(stop_words=stop, min_df=5)

# Create the variable which will hold the WordNetLemmatizer function
wnl = WordNetLemmatizer()

# Create a list of lemmatized reviews
fortopic_lem = []
for rev in fortopic:
    temp = []
    for sent in sent_tokenize(rev):
        for word in word_tokenize(sent):
            temp.append(wnl.lemmatize(word))
    fortopic_lem.append(temp)

# Create a list of lowercase, punctuation removed reviews
fortopic_lower = [[t.lower() for t in rev if t not in string.punctuation] for rev in fortopic_lem ]

# Merge them back to form proper strings - to be passed to the vectorizer        
fortopic_merge = [" ".join(t) for t in fortopic_lower]

# Create the vectorizer
dtm_rev = vectorizer.fit_transform(fortopic_merge).toarray()
vocab = np.array(vectorizer.get_feature_names())

# Check the size of the resulting dtm
dtm_rev.shape

# Initiate the lda model
model = lda.LDA(n_topics=5, n_iter=1500, random_state=1)

# Fit the lda model
%time model.fit(dtm_rev)

# Print out the topics and the top words in each topic
topic_word = model.topic_word_
n_top_words = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# Earlier we treated reviews as docs. Now we treat each sentence as a document

# Carrying out the preprocessing



test_sent=[]

for i in fortopic:
    for sent in sent_tokenize(i):
        test_sent.append(sent)
        
# Creating lemmatized sentences          
sent_lem = []
for rev in test_sent:
    temp = []
    for word in word_tokenize(rev):
        temp.append(wnl.lemmatize(word))
    sent_lem.append(temp)

# Create a list of lowercase, punctuation removed sentences
sent_lower = [[t.lower() for t in rev if t not in string.punctuation] for rev in sent_lem ]

# Merge them back to form proper strings - to be passed to the vectorizer        
sent_merge = [" ".join(t) for t in sent_lower]

# Create the vectorizer
dtm_sent = vectorizer.fit_transform(sent_merge).toarray()
vocab = np.array(vectorizer.get_feature_names())

# Initiate the lda model
model = lda.LDA(n_topics=5, n_iter=1500, random_state=1)

# Fit the lda model
model.fit(dtm_sent)

# Print out the topics and the top words in each topic
topic_word = model.topic_word_
n_top_words = 10
vocab = np.array(vectorizer.get_feature_names())

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

