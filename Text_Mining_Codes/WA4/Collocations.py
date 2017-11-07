# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:48:43 2015

@author: aditya
"""

import nltk.collocations
import nltk.corpus
import collections
import csv
import Orange
import math
from math import log

amnesty = pd.read_csv('amnesty-related.csv')

# Create the list which you would like to filter on
desk=['Foreign Desk', 'Foreign']

# Create two separate data sets, one each for Foreign desk publications and for other desks publication

foreign_unclean=amnesty[amnesty.desk.isin(desk)]
non_foreign_unclean=amnesty[~amnesty.desk.isin(desk)]

# Now, we must merge all the abstracts together to make one large string

# For articles from Foreign desk

foreign_unclean_abstract=""

for i in foreign_unclean.index:
    foreign_unclean_abstract=foreign_unclean_abstract+str(foreign_unclean.abstract[i])
    
# For articles from other desks
    
non_foreign_unclean_abstract=""

for i in non_foreign_unclean.index:
    non_foreign_unclean_abstract=non_foreign_unclean_abstract+str(non_foreign_unclean.abstract[i]) 

# First, let's remove all those components which are not articles, like Obituaries, Biographies, Blogs etc.
dirty_list=['Obituary; Biography','Blog','Biography','Obituary','Web Log']
amnesty_clean=amnesty[~amnesty.type.isin(dirty_list)]

# Now, let's create two different data sets one each for Foreign desk publications and for other desks publication

foreign=amnesty_clean[amnesty_clean.desk.isin(desk)]
non_foreign=amnesty_clean[~amnesty_clean.desk.isin(desk)]

# Now, we must merge all the abstracts together to make one large string

# For articles from Foreign desk
foreign_abstract=""

for i in foreign.index:
    foreign_abstract=foreign_abstract+str(foreign.abstract[i])

# For articles from other desks    
non_foreign_abstract=""

for i in non_foreign.index:
    non_foreign_abstract=non_foreign_abstract+str(non_foreign.abstract[i])

add=['say', 'year', 'international', 'may', 'u', 'amnesty', 'ago', 'photo', 'human', 'right', 'rights','photos', 'states','state','prime','min','oped','article','articles']


text_nopunc=foreign_abstract.translate(string.maketrans("",""), string.punctuation)
text_lower=text_nopunc.lower()
stop = stopwords.words('english')
stop.extend(add)
more_stop = open("long_stopwords.txt").read().splitlines()
stop.extend(more_stop)
text_nostop=" ".join(filter(lambda word: word not in stop, text_lower.split()))
tokens = word_tokenize(text_nostop)
wnl = nltk.WordNetLemmatizer()
text_lem=" ".join([wnl.lemmatize(t) for t in tokens])
textlem_nostop = " ".join(filter(lambda word: word not in stop, text_lem.split()))
tokens_lem = word_tokenize(textlem_nostop)

bgm    = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_words(tokens_lem,window_size=4)
scored = finder.score_ngrams( bgm.jaccard  )

# Group bigrams by first word in bigram.                                        
prefix_keys = collections.defaultdict(list)
for key, scores in scored:
   prefix_keys[key[0]].append((key[1], scores))

# Sort keyed bigrams by strongest association.                                  
for key in prefix_keys:
   prefix_keys[key].sort(key = lambda x: -x[1])

print 'iraq', prefix_keys['iraq'][:5]
print 'saddam', prefix_keys['saddam'][:5]

# Calculate Unigram frequencies
uni_freq = [(item, tokens_lem.count(item)) for item in sorted(set(tokens_lem))]

# Calculate Bigram frequencies

bgs = nltk.bigrams(tokens_lem)
fdist = nltk.FreqDist(bgs)
bi_freq = fdist.items()

# Some minor bigram pre-processing for ease of use later
temp1 = []
for i in bi_freq:
    big = i[0][0] + " " + i[0][1]
    freq = i[1]
    L = [big, freq]
    temp1.append(L)


# Convert into dict for ease of access in the function for pmi
big_freq = dict(temp1)
uni_freq = dict(uni_freq)

# Create a function for pmi - just to calculate for bigrams. Consideration is that you need 
# a list of unigram frequencies and bigram frequencies
def pmi(word1, word2, uni_freq, bi_freq):
    prob_word1 = uni_freq[word1]/float(sum(uni_freq.values()))
    prob_word2 = uni_freq[word2]/float(sum(uni_freq.values()))
    prob_word1_word2 = bi_freq[" ".join([word1,word2])]/float(sum(bi_freq.values()))
    return log(prob_word1_word2/float(prob_word1*prob_word2),2)
    

pmi("saddam", "hussein", uni_freq, big_freq)
pmi("nigerian", "government", uni_freq, big_freq)
# Remove NAs from the dataset
amnesty_nona = amnesty.dropna(subset=['abstract'])

# Create a list with all tokenized abstracts
article_amnesty = []
for i in amnesty_nona.index:
    text = amnesty_nona.abstract[i]
    text_nopunc=text.translate(string.maketrans("",""), string.punctuation)
    text_lower=text_nopunc.lower()
    text_nostop=" ".join(filter(lambda word: word not in stop, text_lower.split()))
    tokens = word_tokenize(text_nostop)
    wnl = nltk.WordNetLemmatizer()
    text_lem=" ".join([wnl.lemmatize(t) for t in tokens])
    textlem_nostop = " ".join(filter(lambda word: word not in stop, text_lem.split()))
    tokens_lem = word_tokenize(textlem_nostop)
    tok=list(set(tokens_lem))
    article_amnesty.append(tok)
    

# Write the list into a csv
with open("mba.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(article_amnesty)
    
# Make sure you "Save As" the csv file to a .basket file ("mba.basket" in my case)

data = Orange.data.Table("mba.basket")

rules = Orange.associate.AssociationRulesSparseInducer(data, support = 0.03)
print "%4s %4s  %s" % ("Supp", "Conf", "Rule")
for r in rules[:5]:
    print "%4.1f %4.1f  %s" % (r.support, r.confidence, r)