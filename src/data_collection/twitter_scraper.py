"""
Twitter data collection module
Fetches tweets related to stocks and financial topics
"""
import tweepy
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import re
from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterDataCollector:
    def __init__(self):
        self.logger = logger
        self.api = None
        self.client = None
        
        # Initialize Twitter API
        self._initialize_api()
    
    def _initialize_api(self):
        """Initialize Twitter API connections"""
        try:
            # Twitter API v2 client
            if Config.TWITTER_BEARER_TOKEN:
                self.client = tweepy.Client(
                    bearer_token=Config.TWITTER_BEARER_TOKEN,
                    consumer_key=Config.TWITTER_API_KEY,
                    consumer_secret=Config.TWITTER_API_SECRET,
                    access_token=Config.TWITTER_ACCESS_TOKEN,
                    access_token_secret=Config.TWITTER_ACCESS_TOKEN_SECRET,
                    wait_on_rate_limit=True
                )
                
                # Twitter API v1.1 for additional features
                auth = tweepy.OAuthHandler(Config.TWITTER_API_KEY, Config.TWITTER_API_SECRET)
                auth.set_access_token(Config.TWITTER_ACCESS_TOKEN, Config.TWITTER_ACCESS_TOKEN_SECRET)
                self.api = tweepy.API(auth, wait_on_rate_limit=True)
                
                self.logger.info("Twitter API initialized successfully")
            else:
                self.logger.warning("Twitter API credentials not configured")
                
        except Exception as e:
            self.logger.error(f"Error initializing Twitter API: {str(e)}")
    
    def search_tweets(self, query: str, max_results: int = 100, days_back: int = 7) -> List[Dict]:
        """
        Search for tweets using Twitter API v2
        
        Args:
            query: Search query
            max_results: Maximum number of tweets to return
            days_back: Number of days to look back
        
        Returns:
            List of tweet data
        """
        if not self.client:
            self.logger.warning("Twitter client not available")
            return []
        
        try:
            # Calculate date range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            # Search tweets
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=min(max_results, 100),  # API limit per request
                start_time=start_time,
                end_time=end_time,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations', 'lang']
            ).flatten(limit=max_results)
            
            tweet_data = []
            for tweet in tweets:
                if tweet.lang == 'en':  # Only English tweets
                    tweet_info = {
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'author_id': tweet.author_id,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'quote_count': tweet.public_metrics['quote_count'],
                        'query': query
                    }
                    tweet_data.append(tweet_info)
            
            self.logger.info(f"Collected {len(tweet_data)} tweets for query: {query}")
            return tweet_data
            
        except Exception as e:
            self.logger.error(f"Error searching tweets for {query}: {str(e)}")
            return []
    
    def get_stock_tweets(self, symbol: str, max_results: int = 100, days_back: int = 7) -> List[Dict]:
        """
        Get tweets specifically about a stock symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            max_results: Maximum number of tweets
            days_back: Number of days to look back
        
        Returns:
            List of tweet data
        """
        # Create comprehensive search query for the stock
        queries = [
            f"${symbol}",  # Cashtag
            f"{symbol} stock",
            f"{symbol} shares",
            f"{symbol} earnings",
            f"{symbol} price"
        ]
        
        all_tweets = []
        tweets_per_query = max_results // len(queries)
        
        for query in queries:
            tweets = self.search_tweets(query, tweets_per_query, days_back)
            all_tweets.extend(tweets)
        
        # Remove duplicates based on tweet ID
        seen_ids = set()
        unique_tweets = []
        for tweet in all_tweets:
            if tweet['id'] not in seen_ids:
                seen_ids.add(tweet['id'])
                unique_tweets.append(tweet)
        
        # Limit to max_results
        return unique_tweets[:max_results]
    
    def get_financial_tweets(self, keywords: List[str], max_results: int = 100, days_back: int = 7) -> List[Dict]:
        """
        Get tweets about general financial topics
        
        Args:
            keywords: List of financial keywords
            max_results: Maximum number of tweets
            days_back: Number of days to look back
        
        Returns:
            List of tweet data
        """
        all_tweets = []
        tweets_per_keyword = max_results // len(keywords)
        
        for keyword in keywords:
            tweets = self.search_tweets(keyword, tweets_per_keyword, days_back)
            all_tweets.extend(tweets)
        
        # Remove duplicates
        seen_ids = set()
        unique_tweets = []
        for tweet in all_tweets:
            if tweet['id'] not in seen_ids:
                seen_ids.add(tweet['id'])
                unique_tweets.append(tweet)
        
        return unique_tweets[:max_results]
    
    def get_trending_financial_topics(self, woeid: int = 1) -> List[str]:
        """
        Get trending financial topics from Twitter
        
        Args:
            woeid: Where On Earth ID (1 = worldwide)
        
        Returns:
            List of trending financial topics
        """
        if not self.api:
            return []
        
        try:
            trends = self.api.get_place_trends(woeid)[0]['trends']
            
            # Filter for financial keywords
            financial_keywords = [
                'stock', 'market', 'trading', 'finance', 'investment',
                'earnings', 'dividend', 'NYSE', 'NASDAQ', 'SPY', 'bull', 'bear'
            ]
            
            financial_trends = []
            for trend in trends:
                trend_name = trend['name'].lower()
                if any(keyword in trend_name for keyword in financial_keywords):
                    financial_trends.append(trend['name'])
            
            return financial_trends[:10]  # Return top 10
            
        except Exception as e:
            self.logger.error(f"Error getting trending topics: {str(e)}")
            return []
    
    def clean_tweet_text(self, text: str) -> str:
        """
        Clean tweet text by removing URLs, mentions, etc.
        
        Args:
            text: Raw tweet text
        
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_tweets_dataframe(self, symbols: List[str], days_back: int = 7, max_per_symbol: int = 50) -> pd.DataFrame:
        """
        Get tweets for multiple symbols and return as DataFrame
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back
            max_per_symbol: Maximum tweets per symbol
        
        Returns:
            DataFrame with all tweets
        """
        all_tweets = []
        
        for symbol in symbols:
            tweets = self.get_stock_tweets(symbol, max_per_symbol, days_back)
            for tweet in tweets:
                tweet['symbol'] = symbol
                tweet['cleaned_text'] = self.clean_tweet_text(tweet['text'])
                all_tweets.append(tweet)
        
        if not all_tweets:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_tweets)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_at', ascending=False)
        
        return df
    
    def calculate_engagement_score(self, tweet_data: Dict) -> float:
        """
        Calculate engagement score for a tweet
        
        Args:
            tweet_data: Tweet data dictionary
        
        Returns:
            Engagement score
        """
        likes = tweet_data.get('like_count', 0)
        retweets = tweet_data.get('retweet_count', 0)
        replies = tweet_data.get('reply_count', 0)
        quotes = tweet_data.get('quote_count', 0)
        
        # Weighted engagement score
        score = (likes * 1) + (retweets * 2) + (replies * 1.5) + (quotes * 2)
        return score

# Example usage
if __name__ == "__main__":
    collector = TwitterDataCollector()
    
    # Test single stock tweets
    if collector.client:
        tweets = collector.get_stock_tweets("AAPL", max_results=10, days_back=1)
        print(f"Collected {len(tweets)} tweets for AAPL")
        
        for tweet in tweets[:3]:
            print(f"Tweet: {tweet['text'][:100]}...")
            print(f"Engagement: {collector.calculate_engagement_score(tweet)}")
            print("---")
        
        # Test multiple stocks
        symbols = ["AAPL", "GOOGL"]
        df = collector.get_tweets_dataframe(symbols, days_back=1, max_per_symbol=5)
        
        if not df.empty:
            print(f"\nDataFrame with {len(df)} tweets created")
            print(df[['symbol', 'created_at', 'like_count', 'retweet_count']].head())
    else:
        print("Twitter API not configured")
