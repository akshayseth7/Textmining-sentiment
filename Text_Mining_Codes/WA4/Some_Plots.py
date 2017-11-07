# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 23:46:30 2015

@author: aditya
"""
import ggplot
import pandasql
from pandasql import *
from ggplot import *

# Read in the csv file
amnesty = pd.read_csv("amnesty-related.csv")


# We are interested in visualizing the primary subject and also seeing what has 
# been the general trend of the subjects across year

col_list = ['year', 'primary_subject']
include = ['UNITED STATES INTERNATIONAL RELATIONS','POLITICS AND GOVERNMENT','ATOMIC WEAPONS','WAR CRIME']

# Create a dataframe with the columns of interest
timedf = amnesty[col_list]

# Filter out only the subjects of interest
timedf = timedf[timedf.primary_subject.isin(include)]

# Create a query to find frequency of each subject of interest

pysqldf = lambda q: sqldf(q, globals())

q  = """
SELECT
 m.primary_subject, count(*) as Freq
FROM
  timedf m
GROUP BY
  m.primary_subject;

"""
# Create a data frame for making the bar plots
df_bar = pysqldf(q)

# Create the bar plots using ggplot
ggplot(aes(x="primary_subject", y="Freq"), df_bar) + geom_bar(stat='identity') + theme(axis_text_x  = element_text(angle = 90, hjust = 1))



# Similarly, create a dataframe for plotting Time Series
q  = """
SELECT
  m.year
  , m.primary_subject, count(*) as Freq
FROM
  timedf m
GROUP BY
  m.year, m.primary_subject;

"""


df_ts = pysqldf(q)

# Create a Time Series plot
ggplot(aes(x='year', y='Freq', colour='primary_subject'), data=df_ts) + geom_line(size = 2)