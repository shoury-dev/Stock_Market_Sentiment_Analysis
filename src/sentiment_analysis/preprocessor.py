"""
Text preprocessing module for sentiment analysis
Cleans and prepares text data for analysis
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import pandas as pd
from typing import List
import logging

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        self.logger = logger
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            
        # Financial stopwords to remove
        self.financial_stopwords = {
            'stock', 'share', 'price', 'market', 'trading', 'trade',
            'buy', 'sell', 'hold', 'investment', 'money', 'dollar'
        }
        
        # Financial terms to keep (don't remove these)
        self.keep_terms = {
            'bullish', 'bearish', 'bull', 'bear', 'long', 'short',
            'call', 'put', 'option', 'gain', 'loss', 'profit',
            'revenue', 'earnings', 'dividend', 'growth', 'decline'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove user mentions and hashtags (social media)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove stock symbols in $SYMBOL format but keep the symbol
        text = re.sub(r'\$([A-Z]+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation while preserving sentence structure
        
        Args:
            text: Input text
        
        Returns:
            Text without punctuation
        """
        # Keep some punctuation that might be important for sentiment
        keep_punct = {'!', '?'}
        
        # Remove other punctuation
        translator = str.maketrans('', '', ''.join(set(string.punctuation) - keep_punct))
        text = text.translate(translator)
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except:
            # Fallback to simple split if NLTK fails
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords while preserving important financial terms
        
        Args:
            tokens: List of tokens
        
        Returns:
            Filtered tokens
        """
        filtered_tokens = []
        
        for token in tokens:
            # Keep important financial terms
            if token.lower() in self.keep_terms:
                filtered_tokens.append(token)
            # Remove stopwords and common financial noise
            elif (token.lower() not in self.stop_words and 
                  token.lower() not in self.financial_stopwords and
                  len(token) > 2):
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their root form
        
        Args:
            tokens: List of tokens
        
        Returns:
            Lemmatized tokens
        """
        try:
            lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
            return lemmatized
        except:
            # Return original tokens if lemmatization fails
            return tokens
    
    def preprocess_text(self, text: str, remove_punct: bool = True, 
                       remove_stops: bool = True, lemmatize: bool = True) -> str:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Raw text
            remove_punct: Whether to remove punctuation
            remove_stops: Whether to remove stopwords
            lemmatize: Whether to lemmatize
        
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        if not text:
            return ""
        
        # Remove punctuation if requested
        if remove_punct:
            text = self.remove_punctuation(text)
        
        # Tokenize
        tokens = self.tokenize_text(text)
        
        # Remove stopwords if requested
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize if requested
        if lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str, 
                           output_column: str = 'processed_text') -> pd.DataFrame:
        """
        Preprocess text in a DataFrame column
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_column: Name of output column
        
        Returns:
            DataFrame with processed text column
        """
        if text_column not in df.columns:
            self.logger.error(f"Column '{text_column}' not found in DataFrame")
            return df
        
        df = df.copy()
        
        # Apply preprocessing
        df[output_column] = df[text_column].apply(
            lambda x: self.preprocess_text(x) if pd.notna(x) else ""
        )
        
        # Remove rows with empty processed text
        df = df[df[output_column].str.strip() != ""]
        
        self.logger.info(f"Preprocessed {len(df)} text entries")
        return df
    
    def extract_financial_entities(self, text: str) -> dict:
        """
        Extract financial entities and keywords from text
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with extracted entities
        """
        text_lower = text.lower()
        
        # Financial sentiment words
        positive_words = [
            'profit', 'gain', 'growth', 'increase', 'rise', 'up', 'bull', 'bullish',
            'surge', 'rally', 'strong', 'positive', 'good', 'excellent', 'beat',
            'outperform', 'upgrade', 'buy', 'recommend'
        ]
        
        negative_words = [
            'loss', 'decline', 'decrease', 'fall', 'down', 'bear', 'bearish',
            'crash', 'drop', 'weak', 'negative', 'bad', 'poor', 'miss',
            'underperform', 'downgrade', 'sell', 'avoid'
        ]
        
        # Count sentiment words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Extract stock symbols
        stock_symbols = re.findall(r'\b[A-Z]{2,5}\b', text)
        
        # Extract monetary amounts
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        
        # Extract percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        
        return {
            'positive_words': positive_count,
            'negative_words': negative_count,
            'stock_symbols': stock_symbols,
            'monetary_amounts': amounts,
            'percentages': percentages,
            'sentiment_ratio': positive_count / max(negative_count, 1)
        }
    
    def prepare_for_sentiment_analysis(self, texts: List[str]) -> List[str]:
        """
        Prepare a list of texts for sentiment analysis
        
        Args:
            texts: List of raw texts
        
        Returns:
            List of preprocessed texts
        """
        processed_texts = []
        
        for text in texts:
            # For sentiment analysis, we want to preserve some punctuation
            # and not remove all stopwords as they can affect sentiment
            processed = self.preprocess_text(
                text, 
                remove_punct=False,  # Keep punctuation for sentiment
                remove_stops=False,  # Keep stopwords for sentiment
                lemmatize=False      # Don't lemmatize for sentiment
            )
            processed_texts.append(processed)
        
        return processed_texts

# Example usage
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Test single text
    sample_text = "AAPL stock is going to the moon! 🚀 Great earnings beat expectations. $AAPL +5.2% today!"
    
    print("Original:", sample_text)
    print("Cleaned:", preprocessor.clean_text(sample_text))
    print("Preprocessed:", preprocessor.preprocess_text(sample_text))
    print("Entities:", preprocessor.extract_financial_entities(sample_text))
    
    # Test DataFrame processing
    df = pd.DataFrame({
        'text': [
            "TSLA earnings were amazing! Stock up 15%",
            "Bad news for GOOGL, revenue miss",
            "Market rally continues, SPY hitting new highs"
        ]
    })
    
    processed_df = preprocessor.preprocess_dataframe(df, 'text')
    print("\nDataFrame processing:")
    print(processed_df[['text', 'processed_text']])
