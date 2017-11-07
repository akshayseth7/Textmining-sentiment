# -*- coding: utf-8 -*-


# Load in the files created, one each for Mexican and Thai restaurants
mex_rev_file = "C:\\Users\\A0136039H\\Downloads\\NMSM\\Day1\\Data\\mexican_restaurant_reviews.json"
thai_rev_file = "C:\\Users\\A0136039H\\Downloads\\NMSM\\Day1\\Data\\thai_restaurant_reviews.json"

# Create an empty list, to which we will add all the reviews for Mexican restaurants
mex_rev = []

# Add to the above list all reviews for Mexican restaurants
with open(mex_rev_file) as mex_reviews:
    for review in mex_reviews:
        review = json.loads(review)
        text = review['text'].encode('utf8')      
        mex_rev.append(text)
        
# Create one large text string with all the reviews stored in the Mexican Review list above created
mex_rev_text=''.join(mex_rev)

# Create an empty list, to which we will add all the reviews for Thai restaurants

thai_rev = []

# Add to the above list all reviews for Thai restaurants

with open(thai_rev_file) as thai_reviews:
    for review in thai_reviews:
        review = json.loads(review)
        text = review['text'].encode('utf8')      
        thai_rev.append(text)        

# Create an empty list, to which we will add all the reviews for Thai restaurants

thai_rev_text=''.join(thai_rev)

# Load in the "pickled" stopwords file I used earlier to remove stopwords
stop = pickle.load(open('stopwords.pickle', 'rb'))

# If you want to remove any stopwords from this list, you can use the following code
if "not" in stop: stop.remove("not")
       

# Create your Wordcloud function. You can re-use this code to create your own wordclouds
def create_cloud(text, name_of_cloud, additional_stop_list, max_words, width, height, bigram=True):
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
    if bigram:
        bigram_merged=list()
        i=0
        for line in my_bigrams:
            bigram_merged.append(line[0]+' ' + line[1])
        counts = collections.Counter(bigram_merged)
    else:
        counts = collections.Counter(tokens_lem)
    max_words = max_words
    final = counts.most_common(max_words)
    max_count = max(final, key=operator.itemgetter(1))[1]
    final = [(name, count / float(max_count))for name, count in final]
    max_word_size = 200
    tags = make_tags(final, maxsize=max_word_size)
    width = width
    height = height
    layout = 3
    background_color = (255, 255, 255)
    create_tag_image(tags, name_of_cloud+'.png', size=(width, height), layout=layout, fontname='Crimson Text', background = background_color)
    
# Write a function to get reviews for different ratings. A rating of 1 would mean a poor review and a rating of 5 would mean
# an excellent review
def get_review(file_name, rating):
    with open(file_name) as low_reviews:
        low_reviews_list = []
        for low_review in low_reviews:
            low_review = json.loads(low_review)
            if low_review['stars'] == rating:
                text = low_review['text'].encode('utf8')      
                low_reviews_list.append(text)
    return low_reviews_list

# Get lists of reviews for each rating. 
low_rev_mex = get_low_review(mex_rev_file, 1)
high_rev_mex = get_low_review(mex_rev_file, 5)    

# Create one large text string with all the reviews stored in the above created lists
low_rev_text=''.join(low_rev_mex)
high_rev_text=''.join(high_rev_mex)
