# -*- coding: utf-8 -*-


import re
import string
import os
import sys

# Requires eGenix.com mx Base Distribution
# http://www.egenix.com/products/python/mxBase/
try:
    from mx.DateTime import *
except ImportError:
    print """
Requires eGenix.com mx Base Distribution
http://www.egenix.com/products/python/mxBase/"""

# Predefined strings.
reg_curr1 = '[$] *([0-9]+(,[0-9]+)*(.?\d*)) '
reg_curr2 = 'one.*?dollars|two.*?dollars|three.*?dollars|four.*?dollars|five.*?dollars|six.*?dollars|seven.*?dollars|eight.*?dollars|nine.*?dollars|ten.*?dollars| \
          eleven.*?dollars|twelve.*?dollars|thirteen.*?dollars|fourteen.*?dollars|fifteen.*?dollars|sixteen.*?dollars|seventeen.*?dollars| \
          eighteen.*?dollars|nineteen.*?dollars|twenty.*?dollars|thirty.*?dollars|forty.*?dollars|fifty.*?dollars|sixty.*?dollars|seventy.*?dollars|eighty.*?dollars| \
          ninety.*?dollars|hundred.*?dollars'
reg_curr3 = '([$] *([0-9]+(,[0-9]+)*(.?\d*)) *thousand)|([$] *([0-9]+(,[0-9]+)*(.?\d*)) *million)|([$] *([0-9]+(,[0-9]+)*(.?\d*)) *billion)|([$] *([0-9]+(,[0-9]+)*(.?\d*)) *trillion)|([$] *([0-9]+(,[0-9]+)*(.?\d*))m)|\
([$] *([0-9]+(,[0-9]+)*(.?\d*))b)|([$] *([0-9]+(,[0-9]+)*(.?\d*))M)|([$] *([0-9]+(,[0-9]+)*(.?\d*))B)'


reg1 = re.compile(reg_curr1, re.IGNORECASE)
reg2 = re.compile(reg_curr2, re.IGNORECASE)
reg3 = re.compile(reg_curr3, re.IGNORECASE)

def tag(text):

    dollar_found = []

    found = reg1.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for dollar in found:
        dollar_found.append(dollar)

    found = reg2.findall(text)
    for dollar in found:
        dollar_found.append(dollar)
        
    found = reg3.findall(text)
    temp=[]    
    for element in found:
        temp.append(max(element, key=len))
    found = temp
    for dollar in found:
        dollar_found.append(dollar)
        
    # Tag only dollar expressions which haven't been tagged.
    for dollar in dollar_found:
        text = re.sub(re.escape(dollar) + '(?!</ADI_DOLLARS>)', '<ADI_DOLLARS>' + dollar + '</ADI_DOLLARS>', text)

    return text
