"""
Stock data collection module
Fetches real-time and historical stock data
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCollector:
    def __init__(self):
        self.logger = logger
    
    def get_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """
        Fetch stock data for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            DataFrame with stock data
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                self.logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            data['Symbol'] = symbol
            data.reset_index(inplace=True)
            
            self.logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            period: Time period
        
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        stock_data = {}
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, period)
            if not data.empty:
                stock_data[symbol] = data
        
        return stock_data
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a stock
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Extract key information
            key_info = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
            return key_info
            
        except Exception as e:
            self.logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}
    
    def calculate_price_change(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price changes and percentage changes
        
        Args:
            data: DataFrame with stock data
        
        Returns:
            DataFrame with additional price change columns
        """
        if data.empty:
            return data
        
        data = data.copy()
        data['Price_Change'] = data['Close'].diff()
        data['Price_Change_Pct'] = data['Close'].pct_change() * 100
        data['Daily_Return'] = (data['Close'] - data['Open']) / data['Open'] * 100
        
        return data
    
    def get_recent_performance(self, symbol: str, days: int = 30) -> Dict:
        """
        Get recent performance metrics for a stock
        
        Args:
            symbol: Stock symbol
            days: Number of days to analyze
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                return {}
            
            # Calculate performance metrics
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            total_return = ((end_price - start_price) / start_price) * 100
            
            volatility = data['Close'].pct_change().std() * 100
            max_price = data['Close'].max()
            min_price = data['Close'].min()
            avg_volume = data['Volume'].mean()
            
            performance = {
                'symbol': symbol,
                'period_days': days,
                'start_price': round(start_price, 2),
                'end_price': round(end_price, 2),
                'total_return_pct': round(total_return, 2),
                'volatility_pct': round(volatility, 2),
                'max_price': round(max_price, 2),
                'min_price': round(min_price, 2),
                'avg_volume': int(avg_volume)
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error calculating performance for {symbol}: {str(e)}")
            return {}

# Example usage
if __name__ == "__main__":
    collector = StockDataCollector()
    
    # Test with a single stock
    data = collector.get_stock_data("AAPL", "1mo")
    print(f"Fetched {len(data)} records for AAPL")
    
    # Test with multiple stocks
    stocks = ["AAPL", "GOOGL", "MSFT"]
    all_data = collector.get_multiple_stocks(stocks)
    
    for symbol, df in all_data.items():
        print(f"{symbol}: {len(df)} records")
    
    # Test stock info
    info = collector.get_stock_info("AAPL")
    print(f"AAPL Info: {info}")
    
    # Test performance
    performance = collector.get_recent_performance("AAPL", 30)
    print(f"AAPL Performance: {performance}")
