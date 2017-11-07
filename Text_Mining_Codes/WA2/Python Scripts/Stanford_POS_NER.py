# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 17:46:00 2015

@author: aditya
"""

import nltk
from nltk import *
import string
from nltk.tokenize import sent_tokenize

# Check if stanford.py file exists
import distutils.sysconfig
print distutils.sysconfig.get_python_lib()+'/nltk/tag/'
# Go to the above path and check


# Import POSTagger and NERTagger from nltk.tag.stanford
from nltk.tag.stanford import POSTagger
from nltk.tag.stanford import NERTagger

# Set JAVAHOME variable as below
import os
java_path = "C:\\Program Files\\Java\\jre1.8.0_60\\bin\\java.exe"
os.environ['JAVAHOME'] = java_path


# Point to the directory holding the models and the directory holding the jar file for the stanford pos tagger. An example is shown below
st=POSTagger("C:\\Users\\TEMP.ISS.000\\Desktop\\stanford-postagger-full-2014-08-27\\models\\english-bidirectional-distsim.tagger", "C:\\Users\\TEMP.ISS.000\\Desktop\\stanford-postagger-full-2014-08-27\\stanford-postagger.jar")
st.tag('What is the airspeed of an unladen swallow ? Airspeed is immaterial. It is the determination that counts'.split())


# Point to the directory holding the models and the directory holding the jar file for the stanford pos tagger. An example is shown below
english_nertagger = NERTagger('C:\\Users\\TEMP.ISS.000\\Desktop\\stanford-ner-2015-04-20\\classifiers\\english.all.3class.distsim.crf.ser.gz', 'C:\\Users\\TEMP.ISS.000\\Desktop\\stanford-ner-2015-04-20\\stanford-ner.jar')

text = 'Text Mining is taught at the Institute of Systems Science in National University of Singapore. Aditya, Zhenzhen and Mun Kew teach on the module. Mun Kew is also the Deputy Director of ISS. Chan Meng is the Director of ISS'

# Stanford NER tagger tags only the first sentence it comes across. It stops tagging after that. Hence, it is essential to 
# tokenize the text into sentences and show them to the tagger one at a time

sentences = sent_tokenize(text)

for sentence in sentences:
    NETag = english_nertagger.tag(word_tokenize(sentence))
    print NETag
    
# Check out the 7 class tagged english.muc.7class.distsim.crf.ser.gz model for more flexibility.
# It tags Currency, Location, Percentages along with Persons, Organizations etc.
english_nertagger = NERTagger('C:\\Users\\aditya\\Downloads\\stanford-ner-2015-04-20\\classifiers\\english.muc.7class.distsim.crf.ser.gz', 'C:\\Users\\aditya\\Downloads\\stanford-ner-2015-04-20\\stanford-ner.jar')

text = 'Text Mining is taught at the Institute of Systems Science in National University of Singapore. Aditya, Zhenzhen and Mun Kew teach on the module. Mun Kew is also the Deputy Director of ISS. Chan Meng is the Director of ISS. Singapore is a great place to stay. Places close by are very pretty. Places like Bali, Lombok and Bangkok and attractive tourist destnations. The GDP of Singapore is $470 Billion and per capita GDP is $85,427. The GDP of Singapore declined in the second quarter of 2015 by 4.6% in comparison to the previous quarter '

sentences = sent_tokenize(text)

for sentence in sentences:
    NETag = english_nertagger.tag(word_tokenize(sentence))
    print NETag



