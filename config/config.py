import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
    
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/sentiment_analysis.db')
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Streamlit
    STREAMLIT_SERVER_PORT = int(os.getenv('STREAMLIT_SERVER_PORT', 8501))
    
    # Data Collection Settings
    MAX_TWEETS_PER_QUERY = 100
    MAX_REDDIT_POSTS = 50
    NEWS_SOURCES = ['reuters', 'bloomberg', 'cnbc', 'marketwatch']
    
    # Sentiment Analysis Settings
    SENTIMENT_THRESHOLD_POSITIVE = 0.1
    SENTIMENT_THRESHOLD_NEGATIVE = -0.1
    
    # Stock Symbols for Analysis
    DEFAULT_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NFLX']
    
    # File Paths
    DATA_DIR = 'data'
    RAW_DATA_DIR = 'data/raw'
    PROCESSED_DATA_DIR = 'data/processed'
    MODELS_DIR = 'data/models'
