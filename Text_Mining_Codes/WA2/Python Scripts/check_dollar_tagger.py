# -*- coding: utf-8 -*-
"""
@author: akshay
"""
# Save the dollar_tagger.py in the Python working directory (Documents/Python Scripts mostly). Easier for the import that way
import dollar_tagger

# Create a sample text to check tagging
text = "I have one billion dollars in my account. Yesterday, I was given $ 1,000,000.24 by my mother. Did not make much difference to my bank balance. However, if you could give me Two hundred and fifty million dollars, that might make a lot of difference. I also have $4.4M in my account since last October"

# Tag Dollar Expressions in the text
dollar_tag = dollar_tagger.tag(text)

# Print Tagged Texts
print dollar_tag

# Extract all expressions tagged as currency
curr_info = re.compile('<ADI_DOLLARS>(.*?)</ADI_DOLLARS>', re.DOTALL).findall(dollar_tag)
print curr_info

