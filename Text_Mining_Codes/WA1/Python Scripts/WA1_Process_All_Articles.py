# -*- coding: utf-8 -*-
"""
@author: akshay
"""

#Now to repeat the previous processing for all the files 

# Set folder path to where the processed files must reside
folder_processed='C:\\Users\\aksha\\Desktop\\TA\\Day2\\Data\\p'

# Check if destination folder exists. If it does not, create it
if not os.path.exists(folder_processed):
    os.makedirs(folder_processed)

# Create an empty list to contain text for each document so that we can create a TDM out of it later
tot_text=[]

# Create a loop to read in each file, process it and store it in the processed folder as JSON files
for data_file in sorted(os.listdir(folder_path)):
    print data_file
    data =  open(os.path.join(folder_path, data_file), "r")
    jdata = json.load(data)
    url=jdata['URL']
    text=jdata['Text']
    ntext=unicodedata.normalize('NFKD', text).encode('ascii','ignore')
    text_nopunc=ntext.translate(string.maketrans("",""), string.punctuation)
    text_lower=text_nopunc.lower()
    stop = stopwords.words('english')
    text_nostop=" ".join(filter(lambda word: word not in stop, text_lower.split()))
    tokens = word_tokenize(text_nostop)
    wnl = nltk.WordNetLemmatizer()
    text_lem=" ".join([wnl.lemmatize(t) for t in tokens])
    tot_text.append(text_lem)
    tokens_lem = word_tokenize(text_lem)
    my_bigrams = nltk.bigrams(tokens_lem)
    my_trigrams = nltk.trigrams(tokens_lem)
    out_data = {'URL': unicodedata.normalize('NFKD', url).encode('ascii','ignore'), 'Text': ntext, 'Text_NoPunctuation': text_nopunc, 'Text_Lower': text_lower,
            'Text_NoStopwords':text_nostop, 'Text_Lemmatized': text_lem, 'Text_Bigrams': my_bigrams, 'Text_Trigrams': my_trigrams}
    json_out_data=json.dumps(out_data, sort_keys=True)
    with open(str(os.path.join(folder_processed, (data_file[:-5]) + "_Processed " + ".json")), 'w') as f:
        json.dump(json_out_data, f)

# Let's now create a TDM
