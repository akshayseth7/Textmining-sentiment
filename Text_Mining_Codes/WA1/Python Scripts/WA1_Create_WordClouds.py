# -*- coding: utf-8 -*-
"""
@author: akshay
"""

# Create Word Clouds
# Extract data published by the Foreign Desk and compare it with the data published by all the other desks

# Proceed first without any cleaning and slowly see the results of cleaning on the Word CLouds

# First let's create the Word Cloud function

def create_cloud(text, name_of_cloud, additional_stop_list):
    text_nopunc=text.translate(string.maketrans("",""), string.punctuation)
    text_lower=text_nopunc.lower()
    stop = stopwords.words('english')
    stop.extend(additional_stop_list)
    text_nostop=" ".join(filter(lambda word: word not in stop, text_lower.split()))
    tokens = word_tokenize(text_nostop)
    wnl = nltk.WordNetLemmatizer()
    text_lem=" ".join([wnl.lemmatize(t) for t in tokens])
    tokens_lem = word_tokenize(text_lem)
    my_bigrams = nltk.bigrams(tokens_lem)
    bigram_merged=list()
    i=0
    for line in my_bigrams:
        bigram_merged.append(line[0]+' ' + line[1])
    counts = collections.Counter(bigram_merged)
    max_words = 180
    final = counts.most_common(max_words)
    max_count = max(final, key=operator.itemgetter(1))[1]
    final = [(name, count / float(max_count))for name, count in final]
    max_word_size = 80
    tags = make_tags(final, maxsize=max_word_size)
    width = 1280
    height = 800
    layout = 3
    background_color = (255, 255, 255)
    create_tag_image(tags, name_of_cloud+'.png', size=(width, height), layout=layout, fontname='Crimson Text', background = background_color)

# Then simply extract Foreign desk data and proceed to create Word Clouds without cleaning

# Read in the csv file

amnesty = pd.read_csv('C:\\Users\\aksha\\Desktop\\TA\\Day2\\Data\\amnesty-related.csv')

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

# Create the word clouds without any cleaning and any additional stopword list provided
create_cloud(foreign_unclean_abstract, "Unclean_Foreign", [])
create_cloud(non_foreign_unclean_abstract, "Unclean_NonForeign", [] )


# To make more meaningful Word Clouds, let's clean the data further and add more stopwords to the stopwords list

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
    
# Create an additional Stopwords list
add=['say', 'year', 'international', 'may', 'u', 'amnesty', 'ago', 'photo', 'human', 'right', 'rights','photos', 'states','state','prime','min','oped','article','articles']

# Create the word clouds after cleaning and additional stopword list provided
create_cloud(foreign_abstract, "Clean_Foreign", add)
create_cloud(non_foreign_abstract, "Clean_NonForeign", add) 

# Congratulations, you have now created your first WordClouds in Python
