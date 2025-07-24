# Stock Market Sentiment Analyzer

A comprehensive sentiment analysis tool that scrapes financial news and social media to predict stock market trends.

## Tech Stack
- **Backend**: Python, Flask
- **Frontend**: Streamlit
- **NLP**: NLTK, TextBlob, VADER, Transformers
- **Data Sources**: Twitter API, Reddit API, News APIs
- **Data Storage**: SQLite/PostgreSQL
- **Visualization**: Plotly, Matplotlib
- **Stock Data**: yfinance, Alpha Vantage

## Features
- 📰 Financial news scraping from multiple sources
- 🐦 Twitter sentiment analysis
- 📊 Reddit financial discussions analysis
- 📈 Stock price correlation with sentiment
- 📊 Interactive visualizations
- 🔮 Basic trend prediction
- 🌐 Web interface for easy interaction

## Project Structure
```
├── src/
│   ├── data_collection/
│   │   ├── news_scraper.py
│   │   ├── twitter_scraper.py
│   │   ├── reddit_scraper.py
│   │   └── stock_data.py
│   ├── sentiment_analysis/
│   │   ├── analyzer.py
│   │   ├── preprocessor.py
│   │   └── models.py
│   ├── prediction/
│   │   ├── correlation.py
│   │   └── trend_predictor.py
│   ├── visualization/
│   │   ├── charts.py
│   │   └── dashboard.py
│   └── api/
│       └── flask_app.py
├── streamlit_app.py
├── config/
│   └── config.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── notebooks/
├── tests/
├── requirements.txt
└── .env.example
```

## Setup Instructions

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up API keys in `.env` file
4. Run the Streamlit app: `streamlit run streamlit_app.py`
5. Or run the Flask API: `python src/api/flask_app.py`

## API Keys Required
- Twitter API v2
- Reddit API
- News API
- Alpha Vantage (for stock data)

## Usage
1. Select stocks to analyze
2. Choose data sources (news, Twitter, Reddit)
3. Set time range for analysis
4. View sentiment analysis results
5. Examine correlation with stock price movements
6. Get basic trend predictions
