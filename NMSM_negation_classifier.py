# -*- coding: utf-8 -*-


import re

# Create a function which will tag all words occurring between
# a negative word and a puntuation as NEG_{word}

def neg_tag(text):
    transformed = re.sub(r"\b(?:never|nothing|nowhere|noone|none|not|haven't|hasn't|hasnt|hadn't|hadnt|can't|cant|couldn't|couldnt|shouldn't|shouldnt|won't|wont|wouldn't|wouldnt|don't|dont|doesn't|doesnt|didn't|didnt|isnt|isn't|aren't|arent|aint|ain't|hardly|seldom)\b[\w\s]+[^\w\s]", lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)), text, flags=re.IGNORECASE)
    return(transformed)

# Check out the tagger

text = "I don't like that place you keep calling awesome."

print neg_tag(text)

      
# Create a training list which will now contain reviews with Negatively tagged words and their labels
train_set_neg = []

# Append elements to the list
for doc in trainset:
    trans = neg_tag(doc[0])
    lab = doc[1]
    train_set_neg.append([trans, lab])

# Create a testing list which will now contain reviews with Negatively tagged words and their labels
test_set_neg = []

# Append elements to the list
for doc in testset:
    trans = neg_tag(doc[0])
    lab = doc[1]
    test_set_neg.append([trans, lab])

train_tokenized = [[wt(x), c] for x,c in train_set_neg]
test_tokenized = [[wt(x), c] for x,c in test_set_neg]

def word_feats(words):
    return dict([(word, True) for word in words])


train_featureset = [(word_feats(d), c) for (d,c) in train_tokenized]
test_featureset = [(word_feats(d), c) for (d,c) in test_tokenized]




#train_set_neg, test_set_neg = featuresets[2000:], featuresets[:2000]
classifier = nltk.NaiveBayesClassifier.train(train_featureset)
print(nltk.classify.accuracy(classifier, test_featureset))
classifier.show_most_informative_features(100)

predictions_nb = []
actuals = [t[1] for t in test_featureset]

test_nolab = [t[0] for t in test_featureset]


for t in test_nolab:
    predictions_nb.append(classifier.classify(t))
    
cm = ConfusionMatrix(predictions_nb, actuals)
print cm


classifier1 = nltk.MaxentClassifier.train(train_featureset, algorithm="IIS", max_iter=25)
print(nltk.classify.accuracy(classifier1, test_featureset))

predictions_me = []


for t in test_nolab:
    predictions_me.append(classifier1.classify(t))

cm = ConfusionMatrix(predictions_me, actuals)
print cm

train_nolab = [t[0].decode('latin-1').encode("utf-8") for t in train_set_neg]
test_nolab = [t[0].decode('latin-1').encode("utf-8") for t in test_set_neg]

train_lab = [t[1] for t in train_set_neg]
test_lab = [t[1] for t in test_set_neg]

vectorizer = TfidfVectorizer()


train_vectors = vectorizer.fit_transform(train_nolab)
test_vectors = vectorizer.transform(test_nolab)

train_vectors.shape
test_vectors.shape

cm = ConfusionMatrix(pred, test_lab)


# SVM Classifier
def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=10000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm

sv = train_svm(train_vectors, train_lab)
predSVM= sv.predict(test_vectors)
pred = list(predSVM)
cm = ConfusionMatrix(pred, test_lab)
cm

def train_dtc(X, y):
    """
    Create and train the Decision Tree Classifier.
    """
    dtc = DecisionTreeClassifier()
    dtc.fit(X, y)
    return dtc

dt = train_dtc(train_vectors, train_lab)
predDT = dt.predict(test_vectors)
pred = list(predDT)
cm = ConfusionMatrix(pred, test_lab)
cm

def train_knn(X, y, n, weight):
    """
    Create and train the k-nearest neighbor.
    """
    knn = KNeighborsClassifier(n_neighbors = n, weights = weight, metric = 'cosine', algorithm = 'brute')
    knn.fit(X, y)
    return knn

kn = train_knn(train_vectors, train_lab, 20, 'distance')
predKN = kn.predict(test_vectors)
pred = list(predKN)
cm = ConfusionMatrix(pred, test_lab)
cm

clf = MultinomialNB().fit(train_vectors, train_lab)
predNB = clf.predict(test_vectors)
pred = list(predNB)
cm = ConfusionMatrix(pred, test_lab)

ch21 = SelectKBest(chi2, k=8000)
train_Kbest = ch21.fit_transform(train_vectors, train_lab)
test_Kbest = ch21.transform(test_vectors)

sv = train_svm(train_Kbest, train_lab)
predSVM= sv.predict(test_Kbest)
pred = list(predSVM)
cm = ConfusionMatrix(pred, test_lab)
cm


clf = MultinomialNB().fit(train_Kbest, train_lab)
predNB = clf.predict(test_Kbest)
pred = list(predNB)
cm = ConfusionMatrix(pred, test_lab)
cm

selected_features = list(np.array(vectorizer.get_feature_names())[ch21.get_support()])
print selected_features





