# -*- coding: utf-8 -*-

import sentiwordnet as swn
swn = swn.SentiWordNetCorpusReader("Sentiwordnet.txt")
import scipy
from scipy import mean

for syn in swn.senti_synsets('slow'):
    print str(syn) 

# Write a function to derive the maximum positive and negative scores of each
# word in the sentiwordnet
       
def get_pos_neg_score(word, metric):
    posi=[]
    negi=[]
    sw = swn
    synsets = sw.senti_synsets(word)
    if len(synsets) == 0:
        return 0,0
    for syn in synsets:
        posi.append(syn.pos_score)
        negi.append(syn.neg_score)
    if metric == "Mean":
        pos = mean(posi)
        neg = mean(negi)
    else:
        pos = max(posi)
        neg = max(negi)
    return pos, neg



# Split every review into words and score each word 
# based on the sentiment score acquired from Sentiwordnet 

# Create empty lists which will hold predictions and actuals
pred = []
actual = []
for line, label in testset:
    pos_rev = neg_rev = 0 
    for word in wt(line):
        pos, neg = get_pos_neg_score(word, "Mean")
        pos_rev+=pos
        neg_rev+=neg
    if pos_rev>neg_rev:
        lab=1
    else:
        lab=-1
    pred.append(lab)
    actual.append(label)
    
actuals = pd.Series(actual)
predicted = pd.Series(pred)

# Confusion Matrix
pd.crosstab(actuals, predicted, rownames=['Actuals'], colnames=['Predicted'], margins=True)

        
