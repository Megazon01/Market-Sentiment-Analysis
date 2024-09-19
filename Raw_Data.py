import praw
import pandas as pd
import datetime as dt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns


def initialize_reddit_client():
    """
    Initialize and return a Reddit API client using PRAW.

    Returns:
        praw.Reddit: An instance of the Reddit client.
    """
    reddit = praw.Reddit(client_id='custom',
                         client_secret='custom',
                         user_agent='custom')
    return reddit


def get_reddit_posts(reddit, subreddit_name, limit=100):
    """
    Extract posts from a specific subreddit.

    Args:
        reddit (praw.Reddit): The Reddit API client.
        subreddit_name (str): The name of the subreddit to extract posts from.
        limit (int): The maximum number of posts to retrieve (default is 100).

    Returns:
        pd.DataFrame: A DataFrame containing the extracted posts with columns 'Title', 'Text', and 'Date'.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = subreddit.hot(limit=limit)

    posts_data = []
    for post in posts:
        posts_data.append([post.title, post.selftext, post.created_utc])

    posts_df = pd.DataFrame(posts_data, columns=['Title', 'Text', 'Date'])
    posts_df['Date'] = posts_df['Date'].apply(lambda x: dt.datetime.fromtimestamp(x))
    return posts_df


def analyze_sentiment(text):
    """
    Perform sentiment analysis on a given text using VADER.

    Args:
        text (str): The text to analyze.

    Returns:
        float: The compound sentiment score of the text.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']


def analyze_posts_sentiment(posts_df):
    """
    Perform sentiment analysis on posts and add sentiment scores to the DataFrame.

    Args:
        posts_df (pd.DataFrame): DataFrame containing post data with 'Title' and 'Text' columns.

    Returns:
        pd.DataFrame: The original DataFrame with added columns for sentiment scores: 'Sentiment', 'Text_Sentiment', and 'Overall_Sentiment'.
    """
    posts_df['Sentiment'] = posts_df['Title'].apply(analyze_sentiment)
    posts_df['Text_Sentiment'] = posts_df['Text'].apply(analyze_sentiment)
    posts_df['Overall_Sentiment'] = (posts_df['Sentiment'] + posts_df['Text_Sentiment']) / 2
    return posts_df


def plot_sentiment(posts_df):
    """
    Plot sentiment analysis results over time.

    Args:
        posts_df (pd.DataFrame): DataFrame containing post data with 'Date' and 'Overall_Sentiment' columns.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=posts_df['Date'], y=posts_df['Overall_Sentiment'], label="Sentiment")
    plt.title('WallStreetBets Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.show()


def main():
    """
    Main function to run the Reddit post extraction and sentiment analysis process.
    """
    reddit = initialize_reddit_client()
    posts_df = get_reddit_posts(reddit, 'wallstreetbets', limit=200)  # Extract 200 posts
    posts_df = analyze_posts_sentiment(posts_df)

    # Display first few rows of data
    print(posts_df.head())

    # Plot sentiment analysis
    plot_sentiment(posts_df)


if __name__ == '__main__':
    main()
