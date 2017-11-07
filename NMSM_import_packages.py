# -*- coding: utf-8 -*-


#Import all packages required for this exercise

import nltk
import os
import json
import unicodedata
import string
import io
import pandas as pd
import textmining
import sklearn
import numpy as np
import rauth
import time
import pickle
global collections
import collections
global operator
import operator
global create_tag_image
global make_tags
global LAYOUTS
global get_tag_counts
from nltk import *
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import stem
from sklearn.feature_extraction.text import CountVectorizer
from pytagcloud import create_tag_image, make_tags, LAYOUTS
from pytagcloud.lang.counter import get_tag_counts


# Check NLTK Version. Must be 2.0.5 to go with Python 2.7

print('The nltk version is {}.'.format(nltk.__version__))

# Download everything from nltk
nltk.download()

# Now let's try to process one article first
