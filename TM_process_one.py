# -*- coding: utf-8 -*-


# Set folder path to the directory where the files are located
folder_path = 'C:\\Users\\A0136039H\\Documents\\Python Scripts\\All Articles'

# Start Processing for one article

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

# create your own list of stopwords
st = ['on','at','the','why','God','blah']
print st

# Convert the stopword free text into tokens to enable further processing
tokens = word_tokenize(text_nostop)
print tokens

len(tokens)
len(text_nostop)

# Now, let's do some Stemming!
# There are different stemmers available in Python. Let's take a look at a few

# The most popular stemmer
porter = stem.porter.PorterStemmer()
text_porter=" ".join([porter.stem(t) for t in tokens])
print text_porter

# The Lancaster Stemmer - developed at Lancaster University
lancaster = stem.lancaster.LancasterStemmer()
text_lancaster=" ".join([porter.stem(t) for t in tokens])
print text_lancaster

# The snowball stemmer -  which supports 13 non-English languages as well!

snowball = stem.snowball.EnglishStemmer()
text_snowball=" ".join([porter.stem(t) for t in tokens])
print text_snowball


# Now, for Lemmatization, which converts each word to it's corresponding lemma, use the Lemmatizer provided by nltk
wnl = nltk.WordNetLemmatizer()
text_lem=" ".join([wnl.lemmatize(t) for t in tokens])
print text_lem



# For bigrams and trigrams, we must use the tokenized form of the text
tokens_lem = word_tokenize(text_lem)
my_bigrams = nltk.bigrams(tokens_lem)
my_trigrams = nltk.trigrams(tokens_lem)
print my_bigrams
print my_trigrams

# Let's now apply this processing to all articles
