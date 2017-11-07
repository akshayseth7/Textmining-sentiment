# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 17:20:11 2015

@author: aditya
"""
#The IEER corpus is marked up for a variety of Named Entities. 
# A Named Entity (more strictly, a Named Entity mention) is a name of an entity belonging to a specified class. 
# For example, the Named Entity classes in IEER include PERSON, LOCATION, ORGANIZATION, DATE and so on. 
# Within NLTK, Named Entities are represented as subtrees within a chunk structure: the class name is treated as node label, while the entity mention itself appears as the leaves of the subtree.
# This is illustrated below, where we have show an extract of the chunk representation of document NYT_19980315.064:

from nltk.corpus import ieer
docs = ieer.parsed_docs('NYT_19980315')
tree = docs[1].text
print tree 

# The CoNLL2002 Dutch and Spanish data is treated similarly, although in this case, the strings are also POS tagged.

from nltk.corpus import conll2002
for doc in conll2002.chunked_sents('ned.train')[27]:
    print doc
    

####### RELATIONSHIP EXTRACTION #######

# Relation Extraction standardly consists of identifying specified relations between Named Entities. 
# For example, assuming that we can recognize ORGANIZATIONs and LOCATIONs in text, we might want to also recognize pairs (o, l)
# of these kinds of entities such that o is located in l.
# The sem.relextract module provides some tools to help carry out a simple version of this task. 
# The mk_pairs() function splits a chunk document into a list of two-member lists, each of which consists of a (possibly empty) 
# string followed by a Tree (i.e., a Named Entity):



from nltk.sem import relextract
from string import join
pairs = relextract.mk_pairs(tree)
for s, tree in pairs:
    print '("...%s", %s)' % (join(s[-5:]),tree)
    
# The function mk_reldicts() processes triples of these pairs, i.e., pairs of the form ((string1, Tree1), (string2, Tree2), (string3, Tree3)) 
# and outputs a dictionary (a reldict) in which Tree1 is the subject of the relation, string2 is the filler and Tree3 is the object of the relation. 
# string1 and string3 are stored as left and right context respectively.
    
reldicts = relextract.mk_reldicts(pairs)
for k, v in reldicts[18].items():
    print k, '=>', v 

for k, v in reldicts[19].items():
    print k, '=>', v 

for r in reldicts[18:20]:
     print '=' * 20
     print r['subjtext']
     print r['filler']
     print r['objtext']
     
# The function relextract() allows us to filter the reldicts according to the classes of the subject and object named entities.
# In addition, we can specify that the filler text has to match a given regular expression, as illustrated in the next example. 
# Here, we are looking for pairs of entities in the IN relation, where IN has signature <ORG, LOC>.
import re
IN = re.compile(r'.*\bin\b(?!\b.+ing\b)')
for fileid in ieer.fileids():
    for doc in ieer.parsed_docs(fileid):
        for rel in relextract.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern = IN):
            print relextract.show_raw_rtuple(rel)

# The next example illustrates a case where we want to find out all the roles that a PERSON can occupy in an ORGANIZATION.

roles = """
 (.*(
 analyst|
 chair(wo)?man|
 commissioner|
 counsel|
 director|
 economist|
 editor|
 executive|
 foreman|
 governor|
 head|
 lawyer|
 leader|
 librarian).*)|
 manager|
 partner|
 president|
 producer|
 professor|
 researcher|
 spokes(wo)?man|
 writer|
 ,\sof\sthe?\s*  # "X, of (the) Y"
 """

ROLES = re.compile(roles, re.VERBOSE)

for fileid in ieer.fileids():
     for doc in ieer.parsed_docs(fileid):
         for rel in relextract.extract_rels('PER', 'ORG', doc, corpus='ieer', pattern=ROLES):
             print relextract.show_raw_rtuple(rel) 