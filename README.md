# Reddit Sentiment Analysis

This program collects data from 
r/wallstreetbets using a reddit API
and analyzes it using VADER in order
to create a sentiment score that is 
plotted against time. This can be a
very useful tool for anyone wanting 
to analyze and draw comparisons between 
the sentiment on the subreddit and 
market performance. 

### **Program Files:**
 - **Raw_Data:** This program will collect
the previous 200 posts for analysis
as specified by the user. The number of 
posts to be extracted is specified by setting
the limit as a function parameter within the 
get_reddit_posts() function (it is set 
in this case as 100). This can also be explicitly 
overwritten when calling the main() function,
as is done here where the limit is again 
given as 200.
 - **Avg_Data:** This program attempts at
collecting more data than the previous one,
and averaging the sentiment score per day.
Results are then saved in **reddit_posts.csv**. Note that 
if there are duplicate results present the
program drops the duplicates and keeps the 
last result.

### **Limitations:**

There is a limit on the amount of data that
Reddit will allow for extraction. In order 
to avoid hitting the rate limits, the Avg_Data.py
program collects data in bathes, and also has
a sleep duration between batch collections 
within the get_reddit_posts_for_date() function.
The user can try experimenting with those for
varying results.

### **Required Libraries:**
- praw
- pandas 
- datetime 
- from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
- matplotlib.pyplot 
- seaborn 
- time
- os

