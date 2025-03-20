import re
from collections import Counter
import numpy as np

class TextAnalyzer:
    def __init__(self):
        # Initialize patterns for AI-generated text detection
        self.ai_patterns = [
            r'\b(furthermore|moreover|additionally)\b',
            r'\b(in conclusion|to summarize|in summary)\b',
            r'\b(it is worth noting|it should be noted)\b',
            r'\b(in terms of|with respect to|regarding)\b',
            r'\b(significantly|substantially|considerably)\b'
        ]
        
        # Common filler words that might indicate AI generation
        self.filler_words = {
            'essentially', 'basically', 'fundamentally',
            'generally', 'typically', 'usually',
            'primarily', 'predominantly', 'mainly',
            'effectively', 'efficiently', 'successfully'
        }
    
    def analyze_text(self, text):
        """Analyze text for potential AI generation markers."""
        try:
            # Perform multiple analyses
            pattern_matches = self._analyze_patterns(text)
            complexity_score = self._analyze_complexity(text)
            repetition_score = self._analyze_repetition(text)
            filler_score = self._analyze_filler_words(text)
            
            # Calculate overall probability
            ai_probability = self._calculate_ai_probability(
                pattern_matches,
                complexity_score,
                repetition_score,
                filler_score
            )
            
            return {
                'ai_generated_probability': ai_probability,
                'analysis_details': {
                    'pattern_matches': pattern_matches,
                    'complexity_score': float(complexity_score),
                    'repetition_score': float(repetition_score),
                    'filler_word_score': float(filler_score)
                },
                'analysis_method': 'linguistic_analysis',
                'analysis_complete': True
            }
        except Exception as e:
            return {
                'error': str(e),
                'analysis_complete': False
            }
    
    def _analyze_patterns(self, text):
        """Analyze text for common AI writing patterns."""
        patterns_found = {}
        for pattern in self.ai_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                patterns_found[pattern] = matches
        return patterns_found
    
    def _analyze_complexity(self, text):
        """Analyze text complexity using various metrics."""
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 0.0
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Combine metrics into a complexity score
        complexity_score = (avg_word_length * 0.5) + (avg_sentence_length * 0.1)
        return float(complexity_score)
    
    def _analyze_repetition(self, text):
        """Analyze text for unusual repetition patterns."""
        words = text.lower().split()
        word_counts = Counter(words)
        
        # Calculate repetition score based on word frequency
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        unique_words = len(word_counts)
        repetition_score = 1 - (unique_words / total_words)
        return float(repetition_score)
    
    def _analyze_filler_words(self, text):
        """Analyze text for common filler words."""
        words = text.lower().split()
        filler_count = sum(1 for word in words if word in self.filler_words)
        
        # Calculate filler word ratio
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        filler_score = filler_count / total_words
        return float(filler_score)
    
    def _calculate_ai_probability(self, pattern_matches, complexity_score, repetition_score, filler_score):
        """Calculate overall AI generation probability."""
        # Calculate pattern score
        pattern_score = min(1.0, len(pattern_matches) / 10)
        
        # Normalize complexity score
        normalized_complexity = min(1.0, complexity_score / 10)
        
        # Weight the different factors
        weights = {
            'patterns': 0.4,
            'complexity': 0.2,
            'repetition': 0.2,
            'filler': 0.2
        }
        
        # Calculate weighted average
        ai_probability = (
            pattern_score * weights['patterns'] +
            normalized_complexity * weights['complexity'] +
            repetition_score * weights['repetition'] +
            filler_score * weights['filler']
        )
        
        return float(ai_probability)
    
    def get_detailed_analysis(self, text):
        """Get a detailed analysis of the text with multiple metrics."""
        basic_analysis = self.analyze_text(text)
        
        # Add more detailed metrics
        words = text.split()
        sentences = text.split('.')
        
        additional_metrics = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'average_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'unique_words_ratio': len(set(words)) / len(words) if words else 0
        }
        
        return {**basic_analysis, 'detailed_metrics': additional_metrics} 