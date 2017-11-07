# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 12:25:53 2015

@author: aditya
"""

# Save the time_tagger.py in the Python working directory (Documents/Python Scripts mostly). Easier for the import that way
import time_tagger

# Create a sample text to check tagging
text = "I will meet you next week like I met you two weeks ago. The date that I had today has not been as exciting as the date we had last year on the 22nd of November 2015. I remember your birthday is on the 16th of August which is a special day for me. The moment the date reads 1st of January, I look forward to spending another good year with you. Friday is the best day at work because it is so much fun. Sunday evenings are the worst because they remind you that Monday is not too far."

# Tag Temporal Expressions in the text
time_tags = time_tagger.tag(text)

# Print Tagged Texts
print time_tags

# Extract all expressions tagged as temporal
time_info = re.compile('<ADI_TIME>(.*?)</ADI_TIME>', re.DOTALL).findall(time_tags)
print time_info

