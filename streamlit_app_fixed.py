import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.data_collection.stock_data import StockDataCollector
from src.data_collection.news_scraper import NewsDataCollector
from src.data_collection.twitter_scraper import TwitterDataCollector
from src.data_collection.reddit_scraper import RedditDataCollector
from src.sentiment_analysis.preprocessor import TextPreprocessor
from src.sentiment_analysis.analyzer import SentimentAnalyzer
from config.config import Config

# Page configuration
st.set_page_config(
    page_title="Stock Market Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = pd.DataFrame()

# Header
st.markdown('<h1 class="main-header">Stock Market Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
st.sidebar.header("Configuration")

# Stock symbol input
symbols_input = st.sidebar.text_input(
    "Stock Symbols (comma-separated)", 
    value="AAPL,TSLA,GOOGL",
    help="Enter stock symbols separated by commas (e.g., AAPL,TSLA,GOOGL)"
)

# Date range
end_date = st.sidebar.date_input("End Date", datetime.now().date())
start_date = st.sidebar.date_input("Start Date", (datetime.now() - timedelta(days=30)).date())

# Data source selection
st.sidebar.subheader("Data Sources")
use_news = st.sidebar.checkbox("News Articles", value=True)
use_twitter = st.sidebar.checkbox("Twitter/X", value=True)
use_reddit = st.sidebar.checkbox("Reddit", value=True)

# Sentiment analysis method
sentiment_method = st.sidebar.selectbox(
    "Sentiment Analysis Method",
    ["ensemble", "vader", "textblob", "finbert"],
    help="Choose the sentiment analysis method"
)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    days_back = st.number_input("Days to Look Back", min_value=1, max_value=30, value=7)
    max_news_articles = st.number_input("Max News Articles", min_value=10, max_value=200, value=50)

@st.cache_resource
def load_data_collectors():
    """Load and cache data collectors"""
    collectors = {
        'stock': StockDataCollector(),
        'news': NewsDataCollector(),
        'twitter': TwitterDataCollector(),
        'reddit': RedditDataCollector()
    }
    
    return collectors

def analyze_stocks(symbols, start_date, end_date, use_news, use_twitter, use_reddit, 
                  sentiment_method, days_back):
    """Main analysis function"""
    
    # Load collectors
    collectors = load_data_collectors()
    
    # Initialize components
    preprocessor = TextPreprocessor()
    analyzer = SentimentAnalyzer()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Collect stock data
    status_text.text("Collecting stock data...")
    progress_bar.progress(10)
    
    stock_data = {}
    for symbol in symbols:
        data = collectors['stock'].get_stock_data(symbol, "1mo")
        if not data.empty:
            stock_data[symbol] = data
    
    # Step 2: Collect sentiment data
    all_sentiment_data = []
    progress_step = 80 / (len([use_news, use_twitter, use_reddit]) + 1)
    current_progress = 10
    
    for symbol in symbols:
        if use_news:
            status_text.text(f"Collecting news for {symbol}...")
            current_progress += progress_step
            progress_bar.progress(min(int(current_progress), 90))
            
            news_data = collectors['news'].get_news_from_api(symbol, days_back)
            for item in news_data:
                all_sentiment_data.append({
                    'symbol': symbol,
                    'text': f"{item.get('title', '')} {item.get('description', '')}",
                    'source': 'news',
                    'timestamp': item.get('published_at', datetime.now()),
                    'url': item.get('url', '')
                })
        
        if use_twitter and collectors['twitter'].client:
            status_text.text(f"Collecting tweets for {symbol}...")
            current_progress += progress_step
            progress_bar.progress(min(int(current_progress), 90))
            
            tweets = collectors['twitter'].get_stock_tweets(symbol, max_results=50, days_back=days_back)
            for tweet in tweets:
                all_sentiment_data.append({
                    'symbol': symbol,
                    'text': tweet.get('text', ''),
                    'source': 'twitter',
                    'timestamp': tweet.get('created_at', datetime.now()),
                    'engagement': collectors['twitter'].calculate_engagement_score(tweet)
                })
        
        if use_reddit and collectors['reddit'].reddit:
            status_text.text(f"Collecting Reddit posts for {symbol}...")
            current_progress += progress_step
            progress_bar.progress(min(int(current_progress), 90))
            
            posts = collectors['reddit'].search_posts_by_symbol(symbol, limit=25, time_filter='week')
            for post in posts:
                all_sentiment_data.append({
                    'symbol': symbol,
                    'text': f"{post.get('title', '')} {post.get('selftext', '')}",
                    'source': 'reddit',
                    'timestamp': datetime.fromtimestamp(post.get('created_utc', 0)),
                    'score': post.get('score', 0)
                })
    
    # Step 3: Analyze sentiment
    status_text.text("Analyzing sentiment...")
    progress_bar.progress(90)
    
    if all_sentiment_data:
        sentiment_df = pd.DataFrame(all_sentiment_data)
        
        # Preprocess text
        sentiment_df['processed_text'] = sentiment_df['text'].apply(
            lambda x: preprocessor.preprocess(x) if isinstance(x, str) else ''
        )
        
        # Analyze sentiment
        sentiment_df = analyzer.analyze_dataframe(
            sentiment_df, 'processed_text', method=sentiment_method
        )
    else:
        sentiment_df = pd.DataFrame()
    
    # Complete
    status_text.text("Analysis complete!")
    progress_bar.progress(100)
    
    return stock_data, sentiment_df

def display_results(stock_data, sentiment_data):
    """Display analysis results"""
    
    if not stock_data and sentiment_data.empty:
        st.warning("No data found for the selected symbols and time range.")
        return
    
    # Overview metrics
    st.subheader("Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stocks Analyzed", len(stock_data))
    
    with col2:
        total_texts = len(sentiment_data) if not sentiment_data.empty else 0
        st.metric("Texts Analyzed", total_texts)
    
    with col3:
        if not sentiment_data.empty and 'sentiment_label' in sentiment_data.columns:
            positive_pct = (sentiment_data['sentiment_label'] == 'positive').mean() * 100
            st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
        else:
            st.metric("Positive Sentiment", "N/A")
    
    with col4:
        if not sentiment_data.empty and 'sentiment_polarity' in sentiment_data.columns:
            avg_sentiment = sentiment_data['sentiment_polarity'].mean()
            st.metric("Avg Sentiment Score", f"{avg_sentiment:.3f}")
        else:
            st.metric("Avg Sentiment Score", "N/A")
    
    # Individual stock analysis
    for symbol in stock_data.keys():
        st.subheader(f"{symbol} Analysis")
        
        # Stock data
        stock_df = stock_data[symbol]
        
        # Create price chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_df.index,
                y=stock_df['Close'],
                mode='lines',
                name=f'{symbol} Close Price',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title=f"{symbol} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Key metrics
            current_price = stock_df['Close'].iloc[-1]
            previous_price = stock_df['Close'].iloc[0]
            price_change = current_price - previous_price
            pct_change = (price_change / previous_price) * 100
            
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("Price Change", f"${price_change:.2f}", f"{pct_change:.2f}%")
            st.metric("Volume", f"{stock_df['Volume'].iloc[-1]:,.0f}")
            st.metric("High", f"${stock_df['High'].max():.2f}")
            st.metric("Low", f"${stock_df['Low'].min():.2f}")
        
        # Sentiment analysis for this symbol
        if not sentiment_data.empty:
            symbol_sentiment = sentiment_data[sentiment_data['symbol'] == symbol]
            
            if not symbol_sentiment.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment distribution
                    if 'sentiment_label' in symbol_sentiment.columns:
                        sentiment_counts = symbol_sentiment['sentiment_label'].value_counts()
                        
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title=f"{symbol} Sentiment Distribution",
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Sentiment by source
                    if 'source' in symbol_sentiment.columns:
                        source_sentiment = symbol_sentiment.groupby(['source', 'sentiment_label']).size().unstack(fill_value=0)
                        
                        fig = px.bar(
                            source_sentiment,
                            title=f"{symbol} Sentiment by Source",
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment timeline
                if 'timestamp' in symbol_sentiment.columns and 'sentiment_polarity' in symbol_sentiment.columns:
                    # Group by date
                    symbol_sentiment['date'] = pd.to_datetime(symbol_sentiment['timestamp']).dt.date
                    daily_sentiment = symbol_sentiment.groupby('date')['sentiment_polarity'].mean().reset_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=daily_sentiment['date'],
                        y=daily_sentiment['sentiment_polarity'],
                        mode='lines+markers',
                        name='Daily Sentiment',
                        line=dict(color='orange')
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Sentiment Timeline",
                        xaxis_title="Date",
                        yaxis_title="Sentiment Score",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")

# Main application
if st.sidebar.button("Analyze Stocks", type="primary"):
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    if not symbols:
        st.error("Please enter at least one stock symbol")
    elif start_date >= end_date:
        st.error("Start date must be before end date")
    else:
        # Run analysis
        stock_data, sentiment_data = analyze_stocks(
            symbols, start_date, end_date, use_news, use_twitter, 
            use_reddit, sentiment_method, days_back
        )
        
        # Store in session state
        st.session_state.stock_data = stock_data
        st.session_state.sentiment_data = sentiment_data
        st.session_state.analysis_complete = True
        
        # Display results
        display_results(stock_data, sentiment_data)

elif st.session_state.analysis_complete:
    # Display previous results
    display_results(st.session_state.stock_data, st.session_state.sentiment_data)

else:
    # Welcome screen
    st.info("👈 Configure your analysis in the sidebar and click 'Analyze Stocks' to get started!")
    
    st.subheader("Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Collection:**
        - **News Analysis**: Scraping financial news from multiple sources
        - **Social Media Monitoring**: Analyzing Twitter discussions
        - **Reddit Insights**: Mining financial subreddits for opinions
        - **Real-time Stock Data**: Yahoo Finance integration
        - **Visual Analytics**: Correlating sentiment with stock price movements
        """)
    
    with col2:
        st.markdown("""
        **Sentiment Analysis:**
        - **Multiple Methods**: VADER, TextBlob, FinBERT ensemble
        - **Financial Domain**: Specialized models for financial text
        - **Real-time Processing**: Live sentiment scoring
        - **Source Attribution**: Track sentiment by data source
        - **Trend Analysis**: Historical sentiment patterns
        """)
    
    st.subheader("Supported Data Sources")
    
    sources_data = {
        'Source': ['News APIs', 'Twitter/X', 'Reddit', 'Yahoo Finance'],
        'Type': ['News Articles', 'Social Media', 'Discussion Forums', 'Stock Data'],
        'Real-time': ['✅', '✅', '✅', '✅'],
        'Historical': ['✅', '✅', '✅', '✅']
    }
    
    st.dataframe(pd.DataFrame(sources_data), use_container_width=True)
    
    st.subheader("Example Analysis")
    st.markdown("""
    Try analyzing popular stocks like:
    - **AAPL** (Apple) - Consumer technology
    - **TSLA** (Tesla) - Electric vehicles
    - **GOOGL** (Alphabet) - Technology/AI
    - **MSFT** (Microsoft) - Cloud computing
    - **AMZN** (Amazon) - E-commerce/Cloud
    - **NVDA** (NVIDIA) - AI/Semiconductors
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Data from Yahoo Finance, NewsAPI, Twitter, Reddit*")
