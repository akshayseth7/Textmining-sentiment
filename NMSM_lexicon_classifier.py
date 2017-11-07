# -*- coding: utf-8 -*-


poslines = open(r'rt-polarity.pos','r').read().splitlines()
neglines = open(r'rt-polarity.neg','r').read().splitlines()

# There is a total of 5331 positives and negatives
# Lets take the first N as training set, and leave the rest for validation
N=4800
poslinesTrain = poslines[:N]
neglinesTrain = neglines[:N]
poslinesTest = poslines[N:]
neglinesTest = neglines[N:]

# Create the train set and the test set by attaching labels to text to form a 
# list of tupes (sentence, label). Labels are 1 for positive, -1 for negatives
trainset = [(x,1) for x in poslinesTrain] + [(x,-1) for x in neglinesTrain]
testset = [(x,1) for x in poslinesTest] + [(x,-1) for x in neglinesTest]

# Count the number of occurrences of each word in positives and negatives
poswords = {} # this dictionary will stopr counts for every word in positives
negwords = {} # and negatives
for line, label in trainset: # for every sentence and its label
	for word in line.split(): # for every word in the sentence
		# increment the counts for this word based on the label
		if label == 1: poswords[word] = poswords.get(word,0) + 1
		else: negwords[word] = negwords.get(word, 0) + 1
		
# Evaluate the test set
wrong = 0 # will store the number of misclassifications
pred_list = []
actual = []
for line, label in testset:
	totpos, totneg = 0.0,0.0
	for word in line.split():
		# Get the (+1 smooth'd) number of counts this word occurs in each class
		#smoothing is done in case this word isn't in train set, so that there
		# is no danger in dividing by 0 later when we do a/(a+b)
		a = poswords.get(word, 0.0) + 1.0
		b = negwords.get(word, 0.0) + 1.0
		#increment our score counter for each class, based on this word
		totpos+=a/(a+b)
		totneg+=b/(a+b)
	#create prediction based on counter values
	prediction=1
	if totneg>totpos: prediction = -1
	pred_list.append(prediction)
	actual.append(label)
	if prediction!=label:
		wrong+=1
		print 'ERROR: %s posscore = %.2f negscore=%.2f' % (line, totpos, totneg)
	else:
		print 'CORRECT: %s posscore=%.2f negscore=%.2f' % (line, totpos, totneg)

print 'error rate is %f' % (1.0*wrong/len(testset),)

# Create Series objects for actuals and predicted - compatible with the Crosstab function from pandas that we will be using 

actuals = pd.Series(actual)
predicted = pd.Series(pred_list)

# Print the confusion matrix
pd.crosstab(actuals, predicted, rownames=['Actuals'], colnames=['Predicted'], margins=True)

# Can also print the classification report which comes with sklearn's metrics pack
target_names = ['Positive', 'Negative']
print(classification_report(actual, predicted, target_names=target_names))
