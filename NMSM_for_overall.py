# -*- coding: utf-8 -*-



# To create overall summary visualizations

import pandasql
from pandasql import *
import pandas as pd


# Read in the csv file in question
test_sql = pd.read_csv('test.csv', encoding='latin-1')

# Initiate pysqldf to enable sql-like querying in python
pysqldf = lambda q: sqldf(q, globals())

# Write your query, counting positive and negative reviews by restaurant
q  = """
SELECT
m.restaurant_id, m.Sentiment, count(*) as Freq
FROM
    test_sql m
GROUP BY
    m.restaurant_id, m.Sentiment;
"""

# Create a dataframe with the result of the query
for_overall = pysqldf(q)

# Write to file
for_overall.to_csv("for_overall.csv")
