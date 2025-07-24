"""
Comprehensive sentiment analysis module
Supports multiple sentiment analysis methods and models
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import List, Dict, Union, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.logger = logger
        
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize FinBERT (financial sentiment model)
        self.finbert = None
        self._initialize_finbert()
        
        # Initialize general sentiment model
        self.general_sentiment = None
        self._initialize_general_sentiment()
    
    def _initialize_finbert(self):
        """Initialize FinBERT model for financial sentiment analysis"""
        try:
            # FinBERT model specifically trained on financial text
            model_name = "ProsusAI/finbert"
            self.finbert = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=-1  # Use CPU
            )
            self.logger.info("FinBERT model initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize FinBERT: {str(e)}")
    
    def _initialize_general_sentiment(self):
        """Initialize general sentiment analysis model"""
        try:
            # RoBERTa model trained on sentiment analysis
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.general_sentiment = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=-1  # Use CPU
            )
            self.logger.info("General sentiment model initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize general sentiment model: {str(e)}")
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with sentiment scores
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert to positive/negative/neutral
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'method': 'textblob',
                'polarity': polarity,
                'subjectivity': subjectivity,
                'label': label,
                'confidence': abs(polarity)
            }
        except Exception as e:
            self.logger.error(f"TextBlob sentiment analysis failed: {str(e)}")
            return {'method': 'textblob', 'polarity': 0, 'subjectivity': 0, 'label': 'neutral', 'confidence': 0}
    
    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with sentiment scores
        """
        try:
            scores = self.vader.polarity_scores(text)
            
            # VADER returns: pos, neu, neg, compound
            compound = scores['compound']  # -1 to 1
            
            # Determine label based on compound score
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'method': 'vader',
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg'],
                'compound': compound,
                'label': label,
                'confidence': abs(compound)
            }
        except Exception as e:
            self.logger.error(f"VADER sentiment analysis failed: {str(e)}")
            return {'method': 'vader', 'positive': 0, 'neutral': 1, 'negative': 0, 'compound': 0, 'label': 'neutral', 'confidence': 0}
    
    def finbert_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment using FinBERT (financial sentiment model)
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with sentiment scores
        """
        if not self.finbert:
            return {'method': 'finbert', 'label': 'neutral', 'confidence': 0, 'error': 'Model not available'}
        
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            result = self.finbert(text)[0]
            
            # FinBERT returns: positive, negative, neutral
            label = result['label'].lower()
            confidence = result['score']
            
            # Convert to standardized format
            if label == 'positive':
                polarity = confidence
            elif label == 'negative':
                polarity = -confidence
            else:
                polarity = 0
            
            return {
                'method': 'finbert',
                'label': label,
                'confidence': confidence,
                'polarity': polarity
            }
        except Exception as e:
            self.logger.error(f"FinBERT sentiment analysis failed: {str(e)}")
            return {'method': 'finbert', 'label': 'neutral', 'confidence': 0, 'polarity': 0, 'error': str(e)}
    
    def general_sentiment_analysis(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment using general sentiment model
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with sentiment scores
        """
        if not self.general_sentiment:
            return {'method': 'general', 'label': 'neutral', 'confidence': 0, 'error': 'Model not available'}
        
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            result = self.general_sentiment(text)[0]
            
            label = result['label'].lower()
            confidence = result['score']
            
            # Map labels to standard format
            if 'pos' in label or label == 'label_2':
                label = 'positive'
                polarity = confidence
            elif 'neg' in label or label == 'label_0':
                label = 'negative'
                polarity = -confidence
            else:
                label = 'neutral'
                polarity = 0
            
            return {
                'method': 'general',
                'label': label,
                'confidence': confidence,
                'polarity': polarity
            }
        except Exception as e:
            self.logger.error(f"General sentiment analysis failed: {str(e)}")
            return {'method': 'general', 'label': 'neutral', 'confidence': 0, 'polarity': 0, 'error': str(e)}
    
    def analyze_sentiment_comprehensive(self, text: str) -> Dict[str, Dict]:
        """
        Run all sentiment analysis methods on a text
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with results from all methods
        """
        if not isinstance(text, str) or not text.strip():
            return {}
        
        results = {}
        
        # Run all methods
        results['textblob'] = self.textblob_sentiment(text)
        results['vader'] = self.vader_sentiment(text)
        results['finbert'] = self.finbert_sentiment(text)
        results['general'] = self.general_sentiment_analysis(text)
        
        # Calculate ensemble score
        results['ensemble'] = self._calculate_ensemble_score(results)
        
        return results
    
    def _calculate_ensemble_score(self, results: Dict[str, Dict]) -> Dict[str, Union[str, float]]:
        """
        Calculate ensemble sentiment score from multiple methods
        
        Args:
            results: Results from different sentiment analysis methods
        
        Returns:
            Ensemble sentiment score
        """
        scores = []
        weights = {'textblob': 0.2, 'vader': 0.3, 'finbert': 0.3, 'general': 0.2}
        
        total_weight = 0
        weighted_sum = 0
        
        for method, weight in weights.items():
            if method in results and 'polarity' in results[method]:
                polarity = results[method]['polarity']
                if polarity is not None and not np.isnan(polarity):
                    weighted_sum += polarity * weight
                    total_weight += weight
        
        if total_weight > 0:
            ensemble_polarity = weighted_sum / total_weight
        else:
            ensemble_polarity = 0
        
        # Determine label
        if ensemble_polarity > 0.05:
            label = 'positive'
        elif ensemble_polarity < -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'method': 'ensemble',
            'polarity': ensemble_polarity,
            'label': label,
            'confidence': abs(ensemble_polarity)
        }
    
    def analyze_batch(self, texts: List[str], method: str = 'ensemble') -> List[Dict]:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of texts to analyze
            method: Sentiment analysis method to use
        
        Returns:
            List of sentiment results
        """
        results = []
        
        for text in texts:
            if method == 'ensemble':
                result = self.analyze_sentiment_comprehensive(text)
                if 'ensemble' in result:
                    results.append(result['ensemble'])
                else:
                    results.append({'method': 'ensemble', 'polarity': 0, 'label': 'neutral', 'confidence': 0})
            elif method == 'textblob':
                results.append(self.textblob_sentiment(text))
            elif method == 'vader':
                results.append(self.vader_sentiment(text))
            elif method == 'finbert':
                results.append(self.finbert_sentiment(text))
            elif method == 'general':
                results.append(self.general_sentiment_analysis(text))
            else:
                results.append({'method': method, 'polarity': 0, 'label': 'neutral', 'confidence': 0, 'error': 'Unknown method'})
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str, 
                         method: str = 'ensemble') -> pd.DataFrame:
        """
        Analyze sentiment for texts in a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to analyze
            method: Sentiment analysis method
        
        Returns:
            DataFrame with sentiment analysis results
        """
        if text_column not in df.columns:
            self.logger.error(f"Column '{text_column}' not found in DataFrame")
            return df
        
        df = df.copy()
        
        # Get texts to analyze
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Analyze sentiment
        self.logger.info(f"Analyzing sentiment for {len(texts)} texts using {method}")
        results = self.analyze_batch(texts, method)
        
        # Add results to DataFrame
        df[f'sentiment_label'] = [r.get('label', 'neutral') for r in results]
        df[f'sentiment_polarity'] = [r.get('polarity', 0) for r in results]
        df[f'sentiment_confidence'] = [r.get('confidence', 0) for r in results]
        
        # Add additional columns based on method
        if method == 'vader':
            df[f'sentiment_positive'] = [r.get('positive', 0) for r in results]
            df[f'sentiment_negative'] = [r.get('negative', 0) for r in results]
            df[f'sentiment_neutral'] = [r.get('neutral', 0) for r in results]
            df[f'sentiment_compound'] = [r.get('compound', 0) for r in results]
        elif method == 'textblob':
            df[f'sentiment_subjectivity'] = [r.get('subjectivity', 0) for r in results]
        
        self.logger.info(f"Sentiment analysis completed using {method}")
        return df
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """
        Get summary statistics of sentiment analysis results
        
        Args:
            df: DataFrame with sentiment results
        
        Returns:
            Summary statistics
        """
        if 'sentiment_label' not in df.columns:
            return {}
        
        total_count = len(df)
        if total_count == 0:
            return {}
        
        # Count sentiment labels
        sentiment_counts = df['sentiment_label'].value_counts()
        
        summary = {
            'total_texts': total_count,
            'positive_count': sentiment_counts.get('positive', 0),
            'negative_count': sentiment_counts.get('negative', 0),
            'neutral_count': sentiment_counts.get('neutral', 0),
            'positive_percentage': (sentiment_counts.get('positive', 0) / total_count) * 100,
            'negative_percentage': (sentiment_counts.get('negative', 0) / total_count) * 100,
            'neutral_percentage': (sentiment_counts.get('neutral', 0) / total_count) * 100
        }
        
        # Calculate average scores if available
        if 'sentiment_polarity' in df.columns:
            summary['average_polarity'] = df['sentiment_polarity'].mean()
            summary['polarity_std'] = df['sentiment_polarity'].std()
        
        if 'sentiment_confidence' in df.columns:
            summary['average_confidence'] = df['sentiment_confidence'].mean()
        
        return summary

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test single text
    sample_texts = [
        "AAPL stock is soaring! Amazing earnings beat expectations!",
        "Terrible news for TSLA, stock price crashing down",
        "GOOGL remains stable, no major changes in price"
    ]
    
    print("Testing individual methods:")
    for text in sample_texts:
        print(f"\nText: {text}")
        
        # Test each method
        textblob_result = analyzer.textblob_sentiment(text)
        print(f"TextBlob: {textblob_result['label']} ({textblob_result['polarity']:.3f})")
        
        vader_result = analyzer.vader_sentiment(text)
        print(f"VADER: {vader_result['label']} ({vader_result['compound']:.3f})")
        
        # Comprehensive analysis
        comprehensive = analyzer.analyze_sentiment_comprehensive(text)
        if 'ensemble' in comprehensive:
            ensemble = comprehensive['ensemble']
            print(f"Ensemble: {ensemble['label']} ({ensemble['polarity']:.3f})")
    
    # Test DataFrame analysis
    print("\nTesting DataFrame analysis:")
    df = pd.DataFrame({'text': sample_texts})
    
    # Analyze with ensemble method
    df_with_sentiment = analyzer.analyze_dataframe(df, 'text', method='ensemble')
    print(df_with_sentiment[['text', 'sentiment_label', 'sentiment_polarity']].to_string())
    
    # Get summary
    summary = analyzer.get_sentiment_summary(df_with_sentiment)
    print(f"\nSentiment Summary: {summary}")
