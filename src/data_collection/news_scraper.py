"""
News scraping module for financial news
Collects news from various financial news sources
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from newsapi import NewsApiClient
from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsDataCollector:
    def __init__(self):
        self.logger = logger
        self.news_api = None
        
        # Initialize NewsAPI if key is available
        if Config.NEWS_API_KEY:
            self.news_api = NewsApiClient(api_key=Config.NEWS_API_KEY)
    
    def get_news_from_api(self, query: str, days_back: int = 7, language: str = 'en') -> List[Dict]:
        """
        Fetch news articles using NewsAPI
        
        Args:
            query: Search query (stock symbol or company name)
            days_back: Number of days to look back
            language: Language of articles
        
        Returns:
            List of news articles
        """
        if not self.news_api:
            self.logger.warning("NewsAPI key not configured")
            return []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Fetch articles
            articles = self.news_api.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language=language,
                sort_by='relevancy',
                page_size=100
            )
            
            news_data = []
            for article in articles['articles']:
                news_item = {
                    'title': article['title'],
                    'description': article['description'],
                    'content': article['content'],
                    'url': article['url'],
                    'source': article['source']['name'],
                    'published_at': article['publishedAt'],
                    'query': query
                }
                news_data.append(news_item)
            
            self.logger.info(f"Fetched {len(news_data)} articles for query: {query}")
            return news_data
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {query}: {str(e)}")
            return []
    
    def scrape_reuters_finance(self, symbol: str) -> List[Dict]:
        """
        Scrape financial news from Reuters
        
        Args:
            symbol: Stock symbol
        
        Returns:
            List of news articles
        """
        try:
            url = f"https://www.reuters.com/finance/stocks/{symbol.upper()}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find news articles (this is a simplified example - actual selectors may vary)
            news_items = soup.find_all('div', class_='story-content')
            
            for item in news_items[:10]:  # Limit to 10 articles
                title_elem = item.find('h3') or item.find('h2')
                desc_elem = item.find('p')
                link_elem = item.find('a')
                
                if title_elem:
                    article = {
                        'title': title_elem.get_text().strip(),
                        'description': desc_elem.get_text().strip() if desc_elem else '',
                        'url': link_elem.get('href') if link_elem else '',
                        'source': 'Reuters',
                        'published_at': datetime.now().isoformat(),
                        'symbol': symbol
                    }
                    articles.append(article)
            
            self.logger.info(f"Scraped {len(articles)} articles from Reuters for {symbol}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error scraping Reuters for {symbol}: {str(e)}")
            return []
    
    def scrape_marketwatch_news(self, symbol: str) -> List[Dict]:
        """
        Scrape news from MarketWatch
        
        Args:
            symbol: Stock symbol
        
        Returns:
            List of news articles
        """
        try:
            url = f"https://www.marketwatch.com/investing/stock/{symbol.lower()}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find news articles
            news_sections = soup.find_all('div', class_='article__content')
            
            for section in news_sections[:10]:
                title_elem = section.find('h3') or section.find('h2')
                summary_elem = section.find('p')
                link_elem = section.find('a')
                
                if title_elem:
                    article = {
                        'title': title_elem.get_text().strip(),
                        'description': summary_elem.get_text().strip() if summary_elem else '',
                        'url': link_elem.get('href') if link_elem else '',
                        'source': 'MarketWatch',
                        'published_at': datetime.now().isoformat(),
                        'symbol': symbol
                    }
                    articles.append(article)
            
            self.logger.info(f"Scraped {len(articles)} articles from MarketWatch for {symbol}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error scraping MarketWatch for {symbol}: {str(e)}")
            return []
    
    def get_financial_news_aggregate(self, symbols: List[str], days_back: int = 7) -> pd.DataFrame:
        """
        Aggregate news from multiple sources for multiple symbols
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back
        
        Returns:
            DataFrame with all news articles
        """
        all_articles = []
        
        for symbol in symbols:
            # Get news from NewsAPI
            if self.news_api:
                api_articles = self.get_news_from_api(symbol, days_back)
                all_articles.extend(api_articles)
            
            # Scrape from Reuters
            reuters_articles = self.scrape_reuters_finance(symbol)
            all_articles.extend(reuters_articles)
            
            # Scrape from MarketWatch
            marketwatch_articles = self.scrape_marketwatch_news(symbol)
            all_articles.extend(marketwatch_articles)
        
        if not all_articles:
            self.logger.warning("No articles found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_articles)
        
        # Remove duplicates based on title
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # Sort by published date
        df['published_at'] = pd.to_datetime(df['published_at'])
        df = df.sort_values('published_at', ascending=False)
        
        self.logger.info(f"Collected {len(df)} unique articles for {len(symbols)} symbols")
        return df
    
    def search_financial_keywords(self, keywords: List[str], days_back: int = 7) -> pd.DataFrame:
        """
        Search for articles containing specific financial keywords
        
        Args:
            keywords: List of keywords to search for
            days_back: Number of days to look back
        
        Returns:
            DataFrame with articles containing the keywords
        """
        all_articles = []
        
        for keyword in keywords:
            if self.news_api:
                articles = self.get_news_from_api(keyword, days_back)
                all_articles.extend(articles)
        
        if not all_articles:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_articles)
        df = df.drop_duplicates(subset=['title'], keep='first')
        df['published_at'] = pd.to_datetime(df['published_at'])
        df = df.sort_values('published_at', ascending=False)
        
        return df

# Example usage
if __name__ == "__main__":
    collector = NewsDataCollector()
    
    # Test single symbol news collection
    symbols = ["AAPL", "GOOGL"]
    news_df = collector.get_financial_news_aggregate(symbols, days_back=3)
    
    if not news_df.empty:
        print(f"Collected {len(news_df)} articles")
        print(news_df[['title', 'source', 'published_at']].head())
    else:
        print("No articles found")
    
    # Test keyword search
    keywords = ["stock market", "inflation", "fed rates"]
    keyword_news = collector.search_financial_keywords(keywords, days_back=1)
    
    if not keyword_news.empty:
        print(f"Found {len(keyword_news)} articles with keywords")
    else:
        print("No keyword articles found")
