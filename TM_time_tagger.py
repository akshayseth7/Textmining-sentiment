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
numbers = "(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten| \
          eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \
          eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
          ninety|hundred|thousand)"
day = "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
week_day = "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
month = "(january|february|march|april|may|june|july|august|september| \
          october|november|december)"
dmy = "(year|day|week|month)"
rel_day = "(today|yesterday|tomorrow|tonight|tonite)"
exp1 = "(before|after|earlier|later|ago)"
exp2 = "(this|next|last)"
iso = "\d+[/-]\d+[/-]\d+ \d+:\d+:\d+\.\d+"
year = "((?<=\s)\d{4}|^\d{4})"
ist = "(\d{1,2}st|\d{1,2}nd|\d{1,2}rd|\d{1,2}th)(\sof\s)"

ism = "((\d{1,2}th\sof\sjanuary|\d{1,2}th\sof\sfebruary|\d{1,2}th\sof\smarch|\d{1,2}th\sof\sapril|\d{1,2}th\sof\smay|\d{1,2}th\sof\sjune| \
\d{1,2}th\sof\sjuly|\d{1,2}th\sof\saugust| \
\d{1,2}th\sof\sseptember|\d{1,2}th\sof\soctober|\d{1,2}th\sof\snovember|\d{1,2}th\sof\sdecember)|(\d{1,2}st\sof\sjanuary| \
\d{1,2}st\sof\sfebruary|\d{1,2}st\sof\smarch|\d{1,2}st\sof\sapril|\d{1,2}st\sof\smay|\d{1,2}st\sof\sjune|\d{1,2}st\sof\sjuly| \
\d{1,2}st\sof\saugust|\d{1,2}st\sof\sseptember|\d{1,2}st\sof\soctober|\d{1,2}st\sof\snovember|\d{1,2}st\sof\sdecember)| \
(\d{1,2}nd\sof\sjanuary|\d{1,2}nd\sof\sfebruary|\d{1,2}nd\sof\smarch|\d{1,2}nd\sof\sapril|\d{1,2}nd\sof\smay|\d{1,2}nd\sof\sjune| \
\d{1,2}nd\sof\sjuly|\d{1,2}nd\sof\saugust|\d{1,2}nd\sof\sseptember|\d{1,2}nd\sof\soctober|\d{1,2}nd\sof\snovember|\d{1,2}nd\sof\sdecember) | \
(\d{1,2}rd\sof\sjanuary|\d{1,2}rd\sof\sfebruary|\d{1,2}rd\sof\smarch|\d{1,2}rd\sof\sapril|\d{1,2}rd\sof\smay|\d{1,2}rd\sof\sjune| \
\d{1,2}rd\sof\sjuly|\d{1,2}rd\sof\saugust|\d{1,2}rd\sof\sseptember|\d{1,2}rd\sof\soctober|\d{1,2}rd\sof\snovember|\d{1,2}rd\sof\sdecember))"

ist = "((\d{1,2}th\sjanuary|\d{1,2}th\sfebruary|\d{1,2}th\smarch|\d{1,2}th\sapril|\d{1,2}th\smay|\d{1,2}th\sjune| \
\d{1,2}th\sof\sjuly|\d{1,2}th\saugust| \
\d{1,2}th\sseptember|\d{1,2}th\soctober|\d{1,2}th\snovember|\d{1,2}th\sdecember)|(\d{1,2}st\sjanuary| \
\d{1,2}st\sfebruary|\d{1,2}st\smarch|\d{1,2}st\sapril|\d{1,2}st\smay|\d{1,2}st\sjune|\d{1,2}st\sjuly| \
\d{1,2}st\saugust|\d{1,2}st\sseptember|\d{1,2}st\soctober|\d{1,2}st\snovember|\d{1,2}st\sdecember)| \
(\d{1,2}nd\sjanuary|\d{1,2}nd\sfebruary|\d{1,2}nd\smarch|\d{1,2}nd\sapril|\d{1,2}nd\smay|\d{1,2}nd\sjune| \
\d{1,2}nd\sjuly|\d{1,2}nd\saugust|\d{1,2}nd\sseptember|\d{1,2}nd\soctober|\d{1,2}nd\snovember|\d{1,2}nd\sdecember) | \
(\d{1,2}rd\sjanuary|\d{1,2}rd\sfebruary|\d{1,2}rd\smarch|\d{1,2}rd\sapril|\d{1,2}rd\smay|\d{1,2}rd\sjune| \
\d{1,2}rd\sjuly|\d{1,2}rd\saugust|\d{1,2}rd\sseptember|\d{1,2}rd\soctober|\d{1,2}rd\snovember|\d{1,2}rd\sdecember))"

regxp1 = "((\d+|(" + numbers + "[-\s]?)+) " + dmy + "s? " + exp1 + ")"
regxp2 = "(" + exp2 + " (" + dmy + "|" + week_day + "|" + month + "))"
regexp4 = "((\d+|(" + ist + "[-\s]?)+) " + month  + ")"
regxp5 = "(" + ist + " (" + month + "))"

reg1 = re.compile(regxp1, re.IGNORECASE)
reg2 = re.compile(regxp2, re.IGNORECASE)
reg3 = re.compile(rel_day, re.IGNORECASE)
reg4 = re.compile(iso)
reg5 = re.compile(year)
reg6 = re.compile(ism, re.IGNORECASE)
reg7 = re.compile(ist, re.IGNORECASE)
reg8 = re.compile(day, re.IGNORECASE)


def tag(text):

    # Initialization
    timex_found = []

    # re.findall() finds all the substring matches, keep only the full
    # matching string. Captures expressions such as 'number of days' ago, etc.
    found = reg1.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)

    # Variations of this thursday, next year, etc
    found = reg2.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)

    # today, tomorrow, etc
    found = reg3.findall(text)
    for timex in found:
        timex_found.append(timex)

    # ISO
    found = reg4.findall(text)
    for timex in found:
        timex_found.append(timex)

    # Year
    found = reg5.findall(text)
    for timex in found:
        timex_found.append(timex)
    
    # eg. 22nd of February, 23rd March etc.
    found = reg6.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)
    
    # IST
    found = reg7.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)
        
    #Day
    found = reg8.findall(text)
    for timex in found:
        timex_found.append(timex)
        
    # Tag only temporal expressions which haven't been tagged.
    for timex in timex_found:
        text = re.sub(timex + '(?!</ADI_TIME>)', '<ADI_TIME>' + timex + '</ADI_TIME>', text)

    return text
