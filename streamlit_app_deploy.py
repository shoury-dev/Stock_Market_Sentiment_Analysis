import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

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
</style>
""", unsafe_allow_html=True)

# Initialize VADER analyzer
@st.cache_resource
def load_vader_analyzer():
    return SentimentIntensityAnalyzer()

vader_analyzer = load_vader_analyzer()

# Helper functions
def get_stock_data(symbol, period="1mo"):
    """Get stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    if not text or not isinstance(text, str):
        return {"compound": 0, "label": "neutral"}
    
    scores = vader_analyzer.polarity_scores(text)
    
    if scores['compound'] >= 0.05:
        label = "positive"
    elif scores['compound'] <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    
    return {
        "compound": scores['compound'],
        "positive": scores['pos'],
        "negative": scores['neg'],
        "neutral": scores['neu'],
        "label": label
    }

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    if not text or not isinstance(text, str):
        return {"polarity": 0, "label": "neutral"}
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "polarity": polarity,
            "subjectivity": blob.sentiment.subjectivity,
            "label": label
        }
    except:
        return {"polarity": 0, "subjectivity": 0, "label": "neutral"}

def get_news_sentiment(symbol, num_articles=5):
    """Get news sentiment for a stock symbol"""
    # Simulated news headlines for demo (replace with actual news API)
    sample_headlines = [
        f"{symbol} reports strong quarterly earnings beating expectations",
        f"Analysts upgrade {symbol} stock rating to buy",
        f"{symbol} announces new product line expansion",
        f"Market volatility affects {symbol} trading volume",
        f"{symbol} CEO discusses future growth strategy"
    ]
    
    sentiments = []
    for headline in sample_headlines[:num_articles]:
        vader_result = analyze_sentiment_vader(headline)
        textblob_result = analyze_sentiment_textblob(headline)
        
        sentiments.append({
            "headline": headline,
            "vader_score": vader_result["compound"],
            "textblob_score": textblob_result["polarity"],
            "vader_label": vader_result["label"],
            "textblob_label": textblob_result["label"]
        })
    
    return sentiments

# Header
st.markdown('<h1 class="main-header">Stock Market Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

# Stock symbol input
symbols_input = st.sidebar.text_input(
    "Stock Symbols (comma-separated)", 
    value="AAPL,TSLA,GOOGL",
    help="Enter stock symbols separated by commas"
)

# Date range
end_date = st.sidebar.date_input("End Date", datetime.now().date())
start_date = st.sidebar.date_input("Start Date", (datetime.now() - timedelta(days=30)).date())

# Analysis options
sentiment_method = st.sidebar.selectbox(
    "Sentiment Analysis Method",
    ["VADER", "TextBlob", "Both"],
    help="Choose the sentiment analysis method"
)

# Analysis button
if st.sidebar.button("Analyze Stocks", type="primary"):
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    if not symbols:
        st.error("Please enter at least one stock symbol")
    else:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Collect data for each symbol
        all_results = {}
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Analyzing {symbol}...")
            progress_bar.progress((i + 1) / len(symbols))
            
            # Get stock data
            stock_data = get_stock_data(symbol, "1mo")
            
            if stock_data.empty:
                st.warning(f"Could not fetch data for {symbol}")
                continue
            
            # Get sentiment data
            news_sentiments = get_news_sentiment(symbol)
            
            # Calculate average sentiments
            if news_sentiments:
                avg_vader = sum(s["vader_score"] for s in news_sentiments) / len(news_sentiments)
                avg_textblob = sum(s["textblob_score"] for s in news_sentiments) / len(news_sentiments)
                
                # Determine overall sentiment
                if sentiment_method == "VADER":
                    overall_score = avg_vader
                elif sentiment_method == "TextBlob":
                    overall_score = avg_textblob
                else:  # Both
                    overall_score = (avg_vader + avg_textblob) / 2
                
                if overall_score > 0.1:
                    overall_label = "positive"
                elif overall_score < -0.1:
                    overall_label = "negative"
                else:
                    overall_label = "neutral"
            else:
                avg_vader = 0
                avg_textblob = 0
                overall_score = 0
                overall_label = "neutral"
            
            all_results[symbol] = {
                "stock_data": stock_data,
                "news_sentiments": news_sentiments,
                "avg_vader": avg_vader,
                "avg_textblob": avg_textblob,
                "overall_score": overall_score,
                "overall_label": overall_label
            }
        
        status_text.text("Analysis complete!")
        progress_bar.progress(1.0)
        
        # Display results
        if all_results:
            st.subheader("Analysis Results")
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Stocks Analyzed", len(all_results))
            
            with col2:
                avg_sentiment = sum(r["overall_score"] for r in all_results.values()) / len(all_results)
                st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
            
            with col3:
                positive_count = sum(1 for r in all_results.values() if r["overall_label"] == "positive")
                st.metric("Positive Stocks", f"{positive_count}/{len(all_results)}")
            
            with col4:
                total_news = sum(len(r["news_sentiments"]) for r in all_results.values())
                st.metric("News Articles", total_news)
            
            # Individual stock results
            for symbol, result in all_results.items():
                st.subheader(f"{symbol} Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Stock price chart
                    stock_data = result["stock_data"]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name=f'{symbol} Close Price',
                        line=dict(color='blue')
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Stock Price (Last 30 Days)",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Key metrics
                    current_price = stock_data['Close'].iloc[-1]
                    previous_price = stock_data['Close'].iloc[0]
                    price_change = current_price - previous_price
                    pct_change = (price_change / previous_price) * 100
                    
                    st.metric("Current Price", f"${current_price:.2f}")
                    st.metric("Price Change", f"${price_change:.2f}", f"{pct_change:.2f}%")
                    st.metric("Sentiment Score", f"{result['overall_score']:.3f}")
                    
                    # Sentiment label with color
                    label = result['overall_label']
                    if label == "positive":
                        st.markdown(f"<p class='positive'>📈 {label.title()}</p>", unsafe_allow_html=True)
                    elif label == "negative":
                        st.markdown(f"<p class='negative'>📉 {label.title()}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p class='neutral'>➡️ {label.title()}</p>", unsafe_allow_html=True)
                
                # Sentiment breakdown
                if result["news_sentiments"]:
                    st.subheader(f"{symbol} News Sentiment Analysis")
                    
                    # Create sentiment distribution chart
                    sentiment_labels = [s["vader_label"] for s in result["news_sentiments"]]
                    sentiment_counts = pd.Series(sentiment_labels).value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Show news headlines
                        st.write("**Sample Headlines:**")
                        for news in result["news_sentiments"][:3]:
                            sentiment_color = {
                                'positive': '🟢',
                                'negative': '🔴',
                                'neutral': '🟡'
                            }.get(news["vader_label"], '⚪')
                            
                            st.write(f"{sentiment_color} {news['headline']}")
                            st.write(f"   Score: {news['vader_score']:.3f}")
                
                st.markdown("---")

else:
    # Welcome screen
    st.info("👈 Configure your analysis in the sidebar and click 'Analyze Stocks' to get started!")
    
    st.subheader("Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Stock Analysis:**
        - Real-time stock price data
        - Historical price charts
        - Price change calculations
        - Volume and volatility metrics
        """)
    
    with col2:
        st.markdown("""
        **🎭 Sentiment Analysis:**
        - VADER sentiment scoring
        - TextBlob polarity analysis
        - News headline analysis
        - Ensemble sentiment methods
        """)
    
    st.subheader("How to Use")
    st.markdown("""
    1. **Enter Stock Symbols**: Add comma-separated symbols (e.g., AAPL,TSLA,GOOGL)
    2. **Choose Date Range**: Select analysis period
    3. **Select Method**: Pick sentiment analysis approach
    4. **Click Analyze**: Get comprehensive analysis results
    5. **View Results**: Explore charts, metrics, and sentiment breakdowns
    """)
    
    st.subheader("Popular Stocks to Try")
    st.markdown("""
    - **AAPL** (Apple) - Consumer technology
    - **TSLA** (Tesla) - Electric vehicles  
    - **GOOGL** (Alphabet) - Technology/AI
    - **MSFT** (Microsoft) - Cloud computing
    - **AMZN** (Amazon) - E-commerce/Cloud
    - **NVDA** (NVIDIA) - AI/Semiconductors
    - **META** (Meta) - Social media
    - **NFLX** (Netflix) - Streaming
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Data from Yahoo Finance • Powered by VADER & TextBlob*")
