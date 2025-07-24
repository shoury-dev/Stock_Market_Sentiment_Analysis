"""
Reddit data collection module
Scrapes financial discussions from relevant subreddits
"""
import praw
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import re
from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditDataCollector:
    def __init__(self):
        self.logger = logger
        self.reddit = None
        
        # Initialize Reddit API
        self._initialize_reddit()
    
    def _initialize_reddit(self):
        """Initialize Reddit API connection"""
        try:
            if all([Config.REDDIT_CLIENT_ID, Config.REDDIT_CLIENT_SECRET, Config.REDDIT_USER_AGENT]):
                self.reddit = praw.Reddit(
                    client_id=Config.REDDIT_CLIENT_ID,
                    client_secret=Config.REDDIT_CLIENT_SECRET,
                    user_agent=Config.REDDIT_USER_AGENT
                )
                
                self.logger.info("Reddit API initialized successfully")
            else:
                self.logger.warning("Reddit API credentials not configured")
                
        except Exception as e:
            self.logger.error(f"Error initializing Reddit API: {str(e)}")
    
    def get_financial_subreddits(self) -> List[str]:
        """
        Get list of popular financial subreddits
        
        Returns:
            List of subreddit names
        """
        return [
            'stocks',
            'investing',
            'SecurityAnalysis',
            'ValueInvesting',
            'StockMarket',
            'financialindependence',
            'personalfinance',
            'wallstreetbets',
            'pennystocks',
            'dividends'
        ]
    
    def search_posts_by_symbol(self, symbol: str, subreddit_names: List[str] = None, 
                              limit: int = 50, time_filter: str = 'week') -> List[Dict]:
        """
        Search for posts mentioning a specific stock symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            subreddit_names: List of subreddits to search
            limit: Maximum number of posts per subreddit
            time_filter: Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
        
        Returns:
            List of post data
        """
        if not self.reddit:
            self.logger.warning("Reddit API not available")
            return []
        
        if subreddit_names is None:
            subreddit_names = self.get_financial_subreddits()
        
        all_posts = []
        
        for subreddit_name in subreddit_names:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for the symbol in the subreddit
                search_terms = [symbol, f"${symbol}", f"{symbol} stock"]
                
                for term in search_terms:
                    posts = subreddit.search(term, sort='relevance', time_filter=time_filter, limit=limit//len(search_terms))
                    
                    for post in posts:
                        post_data = {
                            'id': post.id,
                            'title': post.title,
                            'selftext': post.selftext,
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'created_utc': datetime.fromtimestamp(post.created_utc),
                            'subreddit': subreddit_name,
                            'author': str(post.author) if post.author else '[deleted]',
                            'url': post.url,
                            'permalink': f"https://reddit.com{post.permalink}",
                            'symbol': symbol,
                            'search_term': term
                        }
                        all_posts.append(post_data)
                
                self.logger.info(f"Collected posts from r/{subreddit_name} for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error searching r/{subreddit_name} for {symbol}: {str(e)}")
                continue
        
        # Remove duplicates based on post ID
        seen_ids = set()
        unique_posts = []
        for post in all_posts:
            if post['id'] not in seen_ids:
                seen_ids.add(post['id'])
                unique_posts.append(post)
        
        self.logger.info(f"Collected {len(unique_posts)} unique posts for {symbol}")
        return unique_posts
    
    def get_hot_posts(self, subreddit_names: List[str] = None, limit: int = 25) -> List[Dict]:
        """
        Get hot posts from financial subreddits
        
        Args:
            subreddit_names: List of subreddits
            limit: Number of posts per subreddit
        
        Returns:
            List of post data
        """
        if not self.reddit:
            return []
        
        if subreddit_names is None:
            subreddit_names = self.get_financial_subreddits()
        
        all_posts = []
        
        for subreddit_name in subreddit_names:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                hot_posts = subreddit.hot(limit=limit)
                
                for post in hot_posts:
                    post_data = {
                        'id': post.id,
                        'title': post.title,
                        'selftext': post.selftext,
                        'score': post.score,
                        'upvote_ratio': post.upvote_ratio,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'subreddit': subreddit_name,
                        'author': str(post.author) if post.author else '[deleted]',
                        'url': post.url,
                        'permalink': f"https://reddit.com{post.permalink}"
                    }
                    all_posts.append(post_data)
                
                self.logger.info(f"Collected {limit} hot posts from r/{subreddit_name}")
                
            except Exception as e:
                self.logger.error(f"Error getting hot posts from r/{subreddit_name}: {str(e)}")
                continue
        
        return all_posts
    
    def get_post_comments(self, post_id: str, limit: int = 10) -> List[Dict]:
        """
        Get comments for a specific post
        
        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments
        
        Returns:
            List of comment data
        """
        if not self.reddit:
            return []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Remove "more comments"
            
            comments = []
            for comment in submission.comments.list()[:limit]:
                comment_data = {
                    'id': comment.id,
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc),
                    'author': str(comment.author) if comment.author else '[deleted]',
                    'post_id': post_id
                }
                comments.append(comment_data)
            
            return comments
            
        except Exception as e:
            self.logger.error(f"Error getting comments for post {post_id}: {str(e)}")
            return []
    
    def extract_stock_mentions(self, text: str) -> List[str]:
        """
        Extract stock symbol mentions from text
        
        Args:
            text: Text to analyze
        
        Returns:
            List of stock symbols found
        """
        # Common stock symbol patterns
        patterns = [
            r'\$([A-Z]{1,5})',  # $AAPL format
            r'\b([A-Z]{2,5})\b(?=\s+stock|\s+shares|\s+ticker)',  # AAPL stock format
        ]
        
        symbols = []
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            symbols.extend(matches)
        
        # Filter out common false positives
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        return [symbol for symbol in symbols if symbol not in false_positives]
    
    def get_subreddit_posts_dataframe(self, symbols: List[str], days_back: int = 7, 
                                     max_per_symbol: int = 25) -> pd.DataFrame:
        """
        Get Reddit posts for multiple symbols and return as DataFrame
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back
            max_per_symbol: Maximum posts per symbol
        
        Returns:
            DataFrame with all posts
        """
        all_posts = []
        
        # Determine time filter based on days_back
        if days_back <= 1:
            time_filter = 'day'
        elif days_back <= 7:
            time_filter = 'week'
        elif days_back <= 30:
            time_filter = 'month'
        else:
            time_filter = 'year'
        
        for symbol in symbols:
            posts = self.search_posts_by_symbol(
                symbol, 
                limit=max_per_symbol, 
                time_filter=time_filter
            )
            all_posts.extend(posts)
        
        if not all_posts:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_posts)
        
        # Filter by date if needed
        if days_back <= 30:  # More precise filtering for shorter periods
            cutoff_date = datetime.now() - timedelta(days=days_back)
            df = df[df['created_utc'] >= cutoff_date]
        
        df = df.sort_values('created_utc', ascending=False)
        
        return df
    
    def calculate_post_engagement(self, post_data: Dict) -> float:
        """
        Calculate engagement score for a Reddit post
        
        Args:
            post_data: Post data dictionary
        
        Returns:
            Engagement score
        """
        score = post_data.get('score', 0)
        comments = post_data.get('num_comments', 0)
        upvote_ratio = post_data.get('upvote_ratio', 0.5)
        
        # Weighted engagement score
        engagement = (score * upvote_ratio) + (comments * 2)
        return max(0, engagement)  # Ensure non-negative
    
    def get_trending_stocks_from_wsb(self, limit: int = 100) -> Dict[str, int]:
        """
        Get trending stock mentions from WallStreetBets
        
        Args:
            limit: Number of posts to analyze
        
        Returns:
            Dictionary with stock symbols and mention counts
        """
        if not self.reddit:
            return {}
        
        try:
            wsb = self.reddit.subreddit('wallstreetbets')
            hot_posts = wsb.hot(limit=limit)
            
            stock_mentions = {}
            
            for post in hot_posts:
                # Combine title and selftext
                text = f"{post.title} {post.selftext}"
                symbols = self.extract_stock_mentions(text)
                
                for symbol in symbols:
                    stock_mentions[symbol] = stock_mentions.get(symbol, 0) + 1
            
            # Sort by mention count
            sorted_mentions = dict(sorted(stock_mentions.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_mentions
            
        except Exception as e:
            self.logger.error(f"Error getting trending stocks from WSB: {str(e)}")
            return {}

# Example usage
if __name__ == "__main__":
    collector = RedditDataCollector()
    
    if collector.reddit:
        # Test single symbol search
        posts = collector.search_posts_by_symbol("AAPL", limit=5, time_filter='week')
        print(f"Found {len(posts)} posts for AAPL")
        
        for post in posts[:2]:
            print(f"Title: {post['title']}")
            print(f"Score: {post['score']}, Comments: {post['num_comments']}")
            print(f"Engagement: {collector.calculate_post_engagement(post)}")
            print("---")
        
        # Test multiple symbols DataFrame
        symbols = ["AAPL", "GOOGL"]
        df = collector.get_subreddit_posts_dataframe(symbols, days_back=7, max_per_symbol=3)
        
        if not df.empty:
            print(f"\nDataFrame with {len(df)} posts created")
            print(df[['symbol', 'subreddit', 'score', 'num_comments']].head())
        
        # Test trending stocks
        trending = collector.get_trending_stocks_from_wsb(limit=20)
        print(f"\nTrending stocks: {dict(list(trending.items())[:5])}")
    else:
        print("Reddit API not configured")
