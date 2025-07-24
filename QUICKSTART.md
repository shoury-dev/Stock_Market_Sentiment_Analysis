# 🚀 Stock Market Sentiment Analyzer - Quick Start Guide

Welcome to your comprehensive Stock Market Sentiment Analyzer! This guide will help you get up and running quickly.

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection for API calls and data collection

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
# Install Python requirements
pip install -r requirements.txt

# Or run the setup script
python setup.py
```

### 2. Configure API Keys

1. Copy `.env.example` to `.env`
2. Edit `.env` and add your API keys:

```bash
# Twitter API Keys (optional but recommended)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here

# Reddit API Keys (optional but recommended)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=StockSentimentAnalyzer/1.0

# News API Key (optional)
NEWS_API_KEY=your_news_api_key_here

# Financial Data API (optional, yfinance works without this)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

### 3. Get API Keys (Optional but Recommended)

#### Twitter API v2
1. Go to [developer.twitter.com](https://developer.twitter.com)
2. Apply for a developer account
3. Create a new app and get your Bearer Token

#### Reddit API
1. Go to [reddit.com/prefs/apps](https://reddit.com/prefs/apps)
2. Create a new app (script type)
3. Note your client ID and secret

#### News API
1. Go to [newsapi.org](https://newsapi.org)
2. Sign up for a free account
3. Get your API key

**Note**: The tool works without API keys using simulated data and yfinance for stock prices!

## 🎯 Quick Start

### Option 1: Streamlit Web App (Recommended)

```bash
# Start the web application
streamlit run streamlit_app_fixed.py
```

Then open your browser to the displayed URL (usually http://localhost:8501)

### Option 2: Jupyter Notebook Exploration

```bash
# Start Jupyter
jupyter notebook

# Open the exploration notebook
notebooks/stock_sentiment_analysis_exploration.ipynb
```

### Option 3: Python Scripts

```python
# Example usage in Python
from src.data_collection.stock_data import StockDataCollector
from src.sentiment_analysis.analyzer import SentimentAnalyzer

# Collect stock data
collector = StockDataCollector()
stock_data = collector.get_stock_data("AAPL", "1mo")

# Analyze sentiment
analyzer = SentimentAnalyzer()
sentiment = analyzer.textblob_sentiment("AAPL stock is performing great!")
print(sentiment)
```

## 🎛️ Usage Examples

### Basic Stock Analysis
1. Open Streamlit app
2. Select stocks: AAPL, GOOGL, MSFT
3. Choose time range: 1 Week
4. Select data sources: News, Twitter, Reddit
5. Click "Start Analysis"

### Advanced Analysis
1. Use the Jupyter notebook for detailed exploration
2. Customize analysis parameters
3. Run correlation studies
4. Build custom prediction models

## 📊 Features Overview

### Data Collection
- **Stock Prices**: Real-time and historical via yfinance
- **Financial News**: Multiple news sources and APIs
- **Social Media**: Twitter tweets and Reddit discussions
- **Preprocessing**: Advanced text cleaning and NLP

### Sentiment Analysis
- **Multiple Methods**: VADER, TextBlob, FinBERT
- **Ensemble Scoring**: Combined sentiment for accuracy
- **Financial Focus**: Domain-specific sentiment analysis
- **Confidence Metrics**: Reliability scoring

### Analysis & Visualization
- **Correlation Analysis**: Sentiment vs price movements
- **Statistical Testing**: Significance and lag analysis
- **Interactive Charts**: Time series and scatter plots
- **Prediction Models**: ML-based trend prediction

### Dashboard
- **Real-time Updates**: Live data feeds
- **Multi-symbol Analysis**: Compare multiple stocks
- **Customizable Views**: Flexible time ranges and sources
- **Export Options**: Download results and charts

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Rate Limits**
   - Check your API quotas
   - Reduce analysis frequency
   - Use simulated data mode

3. **Missing Data**
   - Verify internet connection
   - Check API keys are valid
   - Try different time ranges

4. **Performance Issues**
   - Reduce number of stocks analyzed
   - Limit time range
   - Use fewer data sources

### Getting Help

1. Check the Jupyter notebook for detailed examples
2. Review the module documentation in `src/`
3. Check the logs for error messages
4. Ensure all API keys are properly configured

## 📈 Sample Analysis Workflow

1. **Setup**: Configure APIs and install dependencies
2. **Data Collection**: Gather stock prices and sentiment data
3. **Preprocessing**: Clean and prepare text data
4. **Sentiment Analysis**: Apply multiple NLP models
5. **Correlation Study**: Analyze sentiment-price relationships
6. **Visualization**: Create charts and dashboards
7. **Prediction**: Build and test ML models
8. **Interpretation**: Draw insights and conclusions

## 🎯 Next Steps

1. **Explore**: Try different stocks and time periods
2. **Customize**: Modify analysis parameters
3. **Extend**: Add new data sources or models
4. **Deploy**: Scale for production use
5. **Research**: Investigate advanced strategies

## ⚠️ Important Notes

- **Educational Purpose**: This is for learning and research
- **Not Financial Advice**: Don't base real trades on this alone
- **Risk Management**: Always use proper risk controls
- **Compliance**: Ensure regulatory compliance for any trading

---

**Happy Analyzing!** 🚀📊

Need help? Check the documentation in each module or run the Jupyter notebook for detailed examples.
