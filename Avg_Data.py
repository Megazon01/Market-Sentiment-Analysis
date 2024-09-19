import praw
import pandas as pd
import datetime as dt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os


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


def get_reddit_posts_for_date(reddit, subreddit_name, start_date, end_date, batch_size=100):
    """
    Extract posts from a specific subreddit within a given date range.

    Args:
        reddit (praw.Reddit): The Reddit API client.
        subreddit_name (str): The name of the subreddit to extract posts from.
        start_date (datetime.date): The start date for extracting posts.
        end_date (datetime.date): The end date for extracting posts.
        batch_size (int): The number of posts to collect before saving and pausing (default is 100).

    Returns:
        pd.DataFrame: A DataFrame containing the collected posts with columns 'Title', 'Text', and 'Date'.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    last_post_time = dt.datetime.now()

    # Fetch posts in batches
    for submission in subreddit.new(limit=None):
        post_date = dt.datetime.fromtimestamp(submission.created_utc).date()
        if post_date < start_date:
            break  # Stop if we have reached posts older than the start date
        if start_date <= post_date <= end_date:
            posts_data.append([submission.title, submission.selftext, post_date])

            # Fetch posts in batches of `batch_size`
            if len(posts_data) % batch_size == 0:
                print(f"Collected {len(posts_data)} posts...")
                # Save the collected data periodically
                save_posts_to_csv(pd.DataFrame(posts_data, columns=['Title', 'Text', 'Date']))
                # Pause between batches to avoid rate limiting
                time.sleep(2)

        # Pause between requests to avoid hitting rate limits
        time.sleep(1)

    # Save any remaining data
    if posts_data:
        save_posts_to_csv(pd.DataFrame(posts_data, columns=['Title', 'Text', 'Date']))

    posts_df = pd.DataFrame(posts_data, columns=['Title', 'Text', 'Date'])
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


def analyze_daily_sentiment(posts_df):
    """
    Analyze the sentiment for each post and group by day to calculate daily average sentiment.

    Args:
        posts_df (pd.DataFrame): DataFrame containing post data with 'Title', 'Text', and 'Date' columns.

    Returns:
        pd.DataFrame: DataFrame with daily average sentiment scores.
    """
    # Perform sentiment analysis on both title and text
    posts_df['Sentiment_Title'] = posts_df['Title'].apply(analyze_sentiment)
    posts_df['Sentiment_Text'] = posts_df['Text'].apply(analyze_sentiment)

    # Calculate the overall sentiment score as the mean of title and text sentiment
    posts_df['Overall_Sentiment'] = (posts_df['Sentiment_Title'] + posts_df['Sentiment_Text']) / 2

    # Group by 'Date' to calculate the average sentiment for each day
    daily_sentiment = posts_df.groupby('Date').agg({'Overall_Sentiment': 'mean'}).reset_index()
    return daily_sentiment


def save_posts_to_csv(posts_df, filename='reddit_posts.csv'):
    """
    Save the collected posts to a CSV file. Append new data to the existing file if it already exists.

    Args:
        posts_df (pd.DataFrame): DataFrame containing post data to save.
        filename (str): The name of the CSV file to save the data to (default is 'reddit_posts.csv').
    """
    if os.path.exists(filename):
        # If file exists, append new data, avoiding duplicates
        existing_data = pd.read_csv(filename)
        posts_df = pd.concat([existing_data, posts_df]).drop_duplicates(subset=['Title', 'Text'], keep='last')
    posts_df.to_csv(filename, index=False)
    print(f"Data saved to {filename}.")


def load_posts_from_csv(filename='reddit_posts.csv'):
    """
    Load existing posts from a CSV file.

    Args:
        filename (str): The name of the CSV file to load data from (default is 'reddit_posts.csv').

    Returns:
        pd.DataFrame: DataFrame containing the loaded post data, or an empty DataFrame if the file does not exist.
    """
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return pd.DataFrame(columns=['Title', 'Text', 'Date'])  # Return empty DataFrame if file does not exist


def plot_sentiment(daily_sentiment):
    """
    Visualize the daily average sentiment over time using a line plot.

    Args:
        daily_sentiment (pd.DataFrame): DataFrame containing daily sentiment scores with 'Date' and 'Overall_Sentiment' columns.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=daily_sentiment['Date'], y=daily_sentiment['Overall_Sentiment'], label="Average Sentiment")
    plt.title('WallStreetBets Average Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.show()


def main():
    """
    Main function to run sentiment analysis on Reddit posts from the WallStreetBets subreddit.
    """
    reddit = initialize_reddit_client()

    # Set the date range (e.g., last 50 days)
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=50)

    # Load existing posts data
    stored_posts_df = load_posts_from_csv()

    # Extract new posts from the desired date range
    posts_df = get_reddit_posts_for_date(reddit, 'wallstreetbets', start_date, end_date)

    # Combine new and existing posts
    if not posts_df.empty:
        posts_df = pd.concat([stored_posts_df, posts_df]).drop_duplicates(subset=['Title', 'Text'], keep='last')

    # Perform sentiment analysis and calculate daily average sentiment
    daily_sentiment = analyze_daily_sentiment(posts_df)

    # Save the new data to CSV
    save_posts_to_csv(posts_df)

    # Plot the sentiment analysis
    plot_sentiment(daily_sentiment)


if __name__ == '__main__':
    main()
