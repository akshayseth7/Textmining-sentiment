# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:28:15 2015

@author: aditya
"""

import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from nltk.cluster import GAAClusterer
from gensim import corpora, models, similarities 
from nltk.stem.snowball import SnowballStemmer

#import three lists: titles, links and wikipedia synopses
titles = open('title_list.txt').read().split('\n')
#ensures that only the first 100 are read in
titles = titles[:100]

links = open('link_list_imdb.txt').read().split('\n')
links = links[:100]

synopses_wiki = open('synopses_list_wiki.txt').read().split('\n BREAKS HERE')
synopses_wiki = synopses_wiki[:100]

synopses_clean_wiki = []
for text in synopses_wiki:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean_wiki.append(text)

synopses_wiki = synopses_clean_wiki
    
    
genres = open('genres_list.txt').read().split('\n')
genres = genres[:100]

print(str(len(titles)) + ' titles')
print(str(len(links)) + ' links')
print(str(len(synopses_wiki)) + ' synopses')
print(str(len(genres)) + ' genres')

synopses_imdb = open('synopses_list_imdb.txt').read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]

synopses_clean_imdb = []

for text in synopses_imdb:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean_imdb.append(text)

synopses_imdb = synopses_clean_imdb

synopses = []

for i in range(len(synopses_wiki)):
    item = synopses_wiki[i] + synopses_imdb[i]
    synopses.append(item)
    
ranks = []

for i in range(0,len(titles)):
    ranks.append(i)

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')



# Create a TF-IDF Vectorizer by ignoring words with Document Frequency above 0.9 and below 0.05

tfidf_vectorizer1 = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.05, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))


# Creating the tf-idf matrix
%time tfidf_matrix1 = tfidf_vectorizer1.fit_transform(synopses)

print(tfidf_matrix1.shape)

# Fitting a K-Means Model and evaluating its performance

from sklearn.cluster import KMeans
num_clusters = 4
km1 = KMeans(n_clusters=num_clusters, random_state=23)
%time km1.fit(tfidf_matrix1)
clusters1 = km1.labels_.tolist()

# Evaluating the performance of the clustering algorithm. We check the silhouette score of the KMeans. -1 
# represents extremely poor clusters while +1 represents extremely good clusters

array1 = np.array(clusters1)
silhouette_score(tfidf_matrix1, array1, metric='euclidean', sample_size=None, random_state=None)

# We find that the silhouette score is ver close to zero. Let's try creating another matrix
# with slightly less extreme exclusion principles. We exclude only those words who have 
# Document Frequency above 0.8 and below 0.20
tfidf_vectorizer2 = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.20, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

# Creating the if-idf matrix
%time tfidf_matrix2 = tfidf_vectorizer2.fit_transform(synopses)

# Fitting a K-Means Model and evaluating its performance

num_clusters = 4
km2 = KMeans(n_clusters=num_clusters, random_state=23)
%time km2.fit(tfidf_matrix2)
clusters2 = km2.labels_.tolist()

# Evaluating the performance of the clustering algorithm. We check the silhouette score of the KMeans. -1 
# represents extremely poor clusters while +1 represents extremely good clusters
array2 = np.array(clusters2)
silhouette_score(tfidf_matrix2, array2, metric='euclidean', sample_size=None, random_state=None)

# There's a sharp increase in the silhouette score of the cluster! Let's see what these clusters are made up of

# Creating the function which will "get features" from the tf-idf
terms = tfidf_vectorizer2.get_feature_names()

# Creating a similarity matrix which will then be used to create a hierarchical cluster and make a dendrogram
# dist will store all the distances
dist = 1 - cosine_similarity(tfidf_matrix2)

# Now to view the movies in each cluster

# Create a dataframe which will hold all the movies and their cluster memberships
films = { 'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters2, 'genre': genres }
frame = pd.DataFrame(films, index = [clusters2] , columns = ['rank', 'title', 'cluster', 'genre'])

# Check the number of members of each cluster
frame['cluster'].value_counts()

# Check the average rank of movies in each cluster
grouped = frame['rank'].groupby(frame['cluster'])
grouped.mean()

# Viewing the top words of each cluster and the cluster members
from __future__ import print_function

print("Top terms per cluster:")
print()
order_centroids = km2.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :12]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print()
    print()

# Now let's try and plot them on a graph and see how they visualize

# We need a two-dimensional plane to view the plots. Hence, we need to reduce the dimensionality of 
# our distance matrix to two dimensions. Among the many ways to do dimensionality reduction
# MDS and SVD, particularly SVD are the most popular in document clustering. We will try both

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]


# Define the cluster names
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a'}

# Define the cluster names
cluster_names = {0: 'Musical, Romance, Drama', 
                 1: 'War, Soldier, Army', 
                 2: 'Police, Crime Noir, Suspense', 
                 3: 'Family, Love, Home'}



#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters2, title=titles)) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

    
    
plt.show() #show the plot


# You can also use svd to do a similar plot

# We need a two-dimensional plane to view the plots. Hence, we need to reduce the dimensionality of 
# our distance matrix to two dimensions. Among the many ways to do dimensionality reduction
# MDS and SVD, particularly SVD are the most popular in document clustering. 

svd = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=50, random_state=None, tol=0.0)
pos = svd.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]

# Define the cluster colors  
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a'}

# Define the cluster names
cluster_names = {0: 'Musical, Romance, Drama', 
                 1: 'War, Soldier, Army', 
                 2: 'Police, Crime Noir, Suspense', 
                 3: 'Family, Love, Home'}



#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters2, title=titles)) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

    
    
plt.show() #show the plot

# We also show a hierarchical clustering based on Ward's method
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

# Agglomorative clustering in nltk can be achieved by the nltk.gaac (Group Average Agglomerative Clusterer)

clusterer = GAAClusterer(4)
clusters_agg = clusterer.cluster(tfidf_matrix2.toarray(), True)
array3 = np.array(clusters_agg)
# EValuating the nltk Agglomerative clustering
silhouette_score(tfidf_matrix2, array3, metric='cosine', sample_size=None, random_state=None)

# What happens if we apply dimensionality reduction to the tf-idf and use the dimensionally reduced
# feature space for our clustering?

svd = TruncatedSVD(2)
lsa = make_pipeline(svd, Normalizer(copy=False))
X = lsa.fit_transform(tfidf_matrix2)
km_svd = KMeans(n_clusters=num_clusters, random_state=23)
km_svd.fit(X)

clusters_svd = km_svd.labels_.tolist()
array_svd = np.array(clusters_svd)
silhouette_score(X, array_svd, metric='euclidean', sample_size=None, random_state=None)

# We see a sharp increase in cluster performance


#strip any proper names from a text...unfortunately right now this is yanking the first word from a sentence too.
import string

def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

#remove proper names
%time preprocess = [strip_proppers(doc) for doc in synopses]

#tokenize
%time tokenized_text = [tokenize_and_stem(text) for text in preprocess]

stopwords.append("kill")
#remove stop words
%time texts = [[word for word in text if word not in stopwords] for text in tokenized_text]

#create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(texts)

#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.8)

#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]
%time lda = models.LdaModel(corpus, num_topics=4, id2word=dictionary, update_every=4, chunksize=10000, passes=100)
lda.show_topics()