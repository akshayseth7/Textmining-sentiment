# -*- coding: utf-8 -*-
"""

@author: akshay
"""

from nltk import pos_tag


print('The nltk version is {}.'.format(nltk.__version__))

folder_path = 'C:\\Users\\aditya\\Documents\\Python Scripts\\Data'

# Open the JSON data file
data =  open(os.path.join(folder_path, "Article_99.json"), "r")

# Load in the JSON object in the file
jdata = json.load(data)

# Extract the URL and the Text from within the JSON object
url=jdata['URL']
url

text=jdata['Text']
text

# Remove all the encoding, escape and special characters
ntext=unicodedata.normalize('NFKD', text).encode('ascii','ignore')
print ntext

# Remove all the punctuations from the text
text_nopunc=ntext.translate(string.maketrans("",""), string.punctuation)
print text_nopunc

# Convert all characters to Lower case
text_lower=text_nopunc.lower()
print text_lower

# Create a stopword list from the standard list of stopwords available in nltk
stop = stopwords.words('english')
print stop

# Remove all these stopwords from the text
text_nostop=" ".join(filter(lambda word: word not in stop, text_lower.split()))
print text_nostop

# Convert the stopword free text into tokens to enable further processing
tokens = word_tokenize(text_nostop)
print tokens

# Now, for Lemmatization, which converts each word to it's corresponding lemma, use the Lemmatizer provided by nltk
wnl = nltk.WordNetLemmatizer()
text_lem=" ".join([wnl.lemmatize(t) for t in tokens])
print text_lem

# Try the default tagger with the Processed Text
pos = pos_tag(word_tokenize(text_lem)) 
print pos


# Does it work well? Why not?
# The value of the Capitalization and stopwords in finding out the Parts of Speech for a text is lost. 
# If you must, remove the stopwords etc., only after PoS tagging
# Let's check the accuracy of the PoS tagger if we run it on the Pre-Processed text

pos_preProcess = pos_tag(word_tokenize(ntext)) 
print pos_preProcess



# This worked because we already had a tagger already trained on news articles
# Let's check out a way to train our own tagger and evaluate its accuracy
# The first step is to find an annotated corpus of text which is the most closely associated with 
# the problem at hand
# To see an example, let's use a corpus which comes pre-installed with Python NLTK
# The corpus is the famous Brown Corpus which was the first 1 Million word annotated Corpus
# Let's split the corpus into training and testing set for evaluation purposes
# We assign 90% of the corpus for the tagger training

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])
unigram_tagger.evaluate(brown_tagged_sents)

# Now that we are training a tagger on some data, we must be careful not to test it on
# the same data, as we did in the previous example. A tagger that simply memorized its
# training data and made no attempt to construct a general model would get a perfect
# score, but would be useless for tagging new text. Instead, we should split the data,
# training on 90% and testing on the remaining 10%
size = int(len(brown_tagged_sents) * 0.9)
size
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

unigram_tagger = nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)

# Although the score is worse, we now have a better picture of the usefulness of this
# tagger, i.e., its performance on previously unseen text.

# Now we try to see if Bigrams work better

bigram_tagger = nltk.BigramTagger(train_sents)
bigram_tagger.tag(brown_sents[2007])

# We see that the bigram tagger has performed very well indeed, tagging all the words in a sentence which it had encountered before
# Let's evaluate its performance on an unseen sentence

unseen_sent = brown_sents[4203]
bigram_tagger.tag(unseen_sent)

# Notice that the bigram tagger manages to tag every word in a sentence it saw during
# training, but does badly on an unseen sentence. As soon as it encounters a new word
# (i.e., 13.5), it is unable to assign a tag. It cannot tag the following word (i.e., million),
# even if it was seen during training, simply because it never saw it during training with
# a None tag on the previous word. Consequently, the tagger fails to tag the rest of the
# sentence. Its overall accuracy score is very low:

bigram_tagger.evaluate(test_sents)

# As n gets larger, the specificity of the contexts increases, as does the chance that the
# data we wish to tag contains contexts that were not present in the training data. This
# is known as the sparse data problem, and is quite pervasive in NLP. As a consequence,
# there is a trade-off between the accuracy and the coverage of our results (and this is
# related to the precision/recall trade-off in information retrieval)

# Let's also see a Trigram tagger
trigram_tagger = nltk.TrigramTagger(train_sents)
trigram_tagger.tag(brown_sents[2007])

unseen_sent = brown_sents[4203]
trigram_tagger.tag(unseen_sent) 

trigram_tagger.evaluate(test_sents)



# The Trigram Tagger is even worse

# We can also create a Regular Expressions Tagger

patterns = [
(r'.*ing$', 'VBG'),
(r'.*ed$', 'VBD'),
(r'.*es$', 'VBZ'),
(r'.*ould$', 'MD'),
(r'.*\’s$', 'NN$'),
(r'.*s$', 'NNS'),
(r'(The|the|A|a|An|an)$', 'AT'),
(r'.*able$', 'JJ'),
(r'.*ly$', 'RB'),
(r'.*s$', 'NNS'),
(r'.*', 'NN')]

regexTagger = nltk.RegexpTagger(patterns)
regexTagger.evaluate(test_sents)

# The Regex Tagger performs better than the Bigram and the Trigram Taggers

# One way to address the trade-off between accuracy and coverage is to use the more
# accurate algorithms when we can, but to fall back on algorithms with wider coverage
# when necessary. For example, we could combine the results of a bigram tagger, a
# unigram tagger, and a default tagger, as follows

# 1. Try tagging the token with the trigram  tagger.
# 2. If the trigram tagger is unable to find a tag for the token, try the bigram tagger.
# 3. If the bigram tagger is also unable to find a tag, use the unigram tagger.
# 4. If the unigram tagger is also unable to find a tag, use a default tagger.

t0 = nltk.DefaultTagger('NN')
t1 = nltk.RegexpTagger(patterns, backoff = t0 )
t2 = nltk.UnigramTagger(train_sents, backoff=t1)
t3 = nltk.BigramTagger(train_sents, backoff=t2)
t4 = nltk.TrigramTagger(train_sents, backoff=t3)


# Now let's check the accuracy of the ensemble of taggers
evalResult = t4.evaluate(test_sents)
print 'Accuracy is: %4.2f %%' % (100.0 * evalResult)


# What's the accuracy of the tagger if we simply tagged everything to be an 'NN'?
evalResult = t0.evaluate(test_sents)
print 'Accuracy is: %4.2f %%' % (100.0 * evalResult)

# After tagging, we must perform chunking, which will allow us to group tags of similar nature together
# We had created a PoS tagged text from an article earlier. Let's use that

print pos_preProcess

# Let's try and create a chunking which puts together all Named Entities under the same tag

print nltk.ne_chunk(pos_preProcess)

# Let's also try and see if we can build a Regular Expression Chunker to tag more than just named entity

grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
      {<NN+>}
      {<DT><JJ>?<NN>}
      {<DT><JJ>?<NNP>}
      {<DT|PP\$>?<JJ>*<NNP>}
      
"""

cp = nltk.RegexpParser(grammar)
print cp.parse(pos_preProcess)

# Similar to tagging, what if you want to train your own chunker? Let's try to train and evaluate our chunker with an already available Corpora

from nltk.corpus import conll2000
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

# We first create a Chunker class. You can create your own class of chunker.

class ChunkParser(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = nltk.TrigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag) in zip(sentence, chunktags)]
        from nltk.chunk.util import conlltags2tree
        return conlltags2tree(conlltags)

# Now we train the chunker with the training data set
NPChunker = ChunkParser(train_sents)

# Testing it on an unseen dataset
print NPChunker.evaluate(test_sents)

# This chunker does reasonably well, achieving an overall f-measure score of 84.6%.
 
# Let's take a look at what the chunker has learnt

print NPChunker.parse(conll2000.tagged_sents()[500])

# Check out the IOB chunking predicted by the chunker
postags = sorted(set(pos for sent in test_sents for (word,pos) in sent.leaves()))
print NPChunker.tagger.tag(postags)




